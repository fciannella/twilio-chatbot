# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

"""Speech planner service for managing real-time speech interactions and VAD-based interruptions.

This module provides the SpeechPlanner class which handles:
- Voice Activity Detection (VAD) processing
- Speech interaction management
- Interruption handling based on VAD signals
- Coordination of speech prediction and transcription frames
"""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import yaml
from langchain_core.messages.base import BaseMessageChunk
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pydantic import BaseModel, Field

from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService


class SpeechPlanner(NvidiaLLMService):
    """Speech planner that manages speech interactions and interruptions based on VAD and predictions."""

    class InputParams(BaseModel):
        """Parameters for controlling NVIDIA LLM behavior."""

        frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
        presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
        seed: int | None = Field(default=None, ge=0)
        temperature: float | None = Field(default=None, ge=0.0, le=2.0)
        top_k: int | None = Field(default=None, ge=0)
        top_p: float | None = Field(default=None, ge=0.0, le=1.0)
        max_tokens: int | None = Field(default=None, ge=1)
        max_completion_tokens: int | None = Field(default=None, ge=1)
        extra: dict[str, Any] | None = Field(default=None)

    def __init__(
        self,
        *,
        prompt_file: str,
        model: str = "nvdev/google/gemma-2b-it",
        api_key: str = None,
        base_url: str | None = None,
        context: OpenAILLMContext = None,
        params: InputParams | None = None,
        context_window: int = 1,  # Number of previous conversation turns to consider for the current conversation.
        **kwargs,
    ):
        """Initialize the speech planner.

        Args:
            prompt_file: Path to YAML file containing prompts
            model: Name of the NVIDIA LLM model. Defaults to "nvdev/google/gemma-2b-it"
            api_key: API key for authentication
            base_url: Base URL for the API
            params: Input parameters for the service
            context: Context manager for conversation history
            context_window: Number of previous conversation turns to consider. Defaults to 1
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        if params is None:
            params = SpeechPlanner.InputParams()
        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "seed": params.seed,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "max_completion_tokens": params.max_completion_tokens,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self.set_model_name(model)
        self._client = self.create_client(api_key=api_key, base_url=base_url, **kwargs)
        self.context = context
        self.context_window = context_window
        with open(prompt_file) as file:
            self.prompts = yaml.safe_load(file)
        self.last_processed_frame = None
        self.last_frame = None
        self.last_complete_interim_frame = None
        self.user_speaking = None
        self.latest_bot_started_speaking_frame_timestamp = None
        self.current_prediction = None
        self._current_task = None

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create a client for the NVIDIA LLM service."""
        return ChatNVIDIA(
            base_url=base_url,
            model=self.model_name,
            api_key=api_key,
        )

    async def get_chat_completions(self, messages) -> AsyncIterator[BaseMessageChunk]:
        """Get chat completions from the LLM model.

        Args:
            messages: The input messages to process

        Returns:
            AsyncIterator[BaseMessageChunk]: Stream of response chunks from the model
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": messages,
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }
        params.update(self._settings["extra"])
        chunks = self._client.astream(input=messages, config=params)
        return chunks

    async def _stream_chat_completions(self, prompt: str) -> AsyncIterator[BaseMessageChunk]:
        """Stream chat completions for a given prompt.

        Args:
            prompt (str): The prompt to send to the model

        Returns:
            AsyncIterator[BaseMessageChunk]: Stream of response chunks
        """
        logger.debug(f"Generating chat: {prompt}")

        chunks = await self.get_chat_completions(prompt)
        return chunks

    def get_chat_history(self) -> list:
        """Retrieves a subset of the conversation history for context in speech planning.

        This method calculates how many recent conversation turns to include based on
        the configured context_window size. It ensures we start with a user message
        for proper conversation flow.

        The method will:
        1. Return empty list if no messages exist
        2. Calculate the starting point based on context_window size
        3. Include messages from either:
           - The last N user-assistant pairs (where N = context_window)
           - Or slightly more if needed to start with a user message

        Returns:
            list: A slice of conversation history messages, starting with a user message
        """
        chat_history = []
        messages = self.context.get_messages()

        if len(messages) == 0:
            return chat_history

        # Calculate how many conversation turns to include
        # Each turn consists of 2 messages (user + assistant)
        conversation_turns_to_include = min(
            self.context_window,  # Max turns specified in config
            (len(messages) - 2) / 2,  # Available complete turns
        )
        start_position = max(0, int(conversation_turns_to_include * 2))

        # Try to start from a user message
        if -len(messages) <= -start_position < len(messages) and messages[-start_position]["role"] == "user":
            chat_history = messages[-start_position:]
        # If the above doesn't work, try starting one message later
        elif (
            -len(messages) <= (-start_position + 1) < len(messages) and messages[-start_position + 1]["role"] == "user"
        ):
            chat_history = messages[-start_position + 1 :]

        return chat_history

    async def _cancel_current_task(self):
        """Cancel the current prediction task if it exists and is running."""
        if self._current_task is not None:
            if not (self._current_task.done() or self._current_task.cancelled()):
                logger.debug("Speech Planner: Cancelling previous task")
                await self.cancel_task(self._current_task)
            self._current_task = None

    async def _process_complete_context(self, frame: TranscriptionFrame):
        """Process a transcription frame to determine if it represents a complete utterance.

        This method uses the LLM to analyze the transcription and determine if it's
        a complete thought/sentence. If complete, it triggers appropriate interruption
        frames and forwards the transcription.

        Args:
            frame (TranscriptionFrame): The transcription frame to process
        """
        try:
            base_prompt = self.prompts["prompts"]["completion_prompt"]
            chat_history = self.get_chat_history()
            transcript = frame.text
            prompt = ""
            if self.prompts["configurations"]["using_chat_history"]:
                prompt = base_prompt.format(transcript=transcript, chat_history=chat_history)
            else:
                prompt = base_prompt.format(transcript=transcript)
            chunk_stream = await self._stream_chat_completions(prompt)
            pred = ""
            async for chunk in chunk_stream:
                if not chunk.content:
                    continue
                try:
                    pred += chunk.content
                except Exception as e:
                    logger.debug(
                        f"Failed to append chunk content: {e}, chunk: {chunk}, setting prediction to '<silent>'"
                    )
                    pred = "<silent>"
                pred = pred.strip()
            logger.debug(
                f"""Speech Planner : Smart EOU Detection 
                    \n\t Transcript: {transcript} \t Prompt: {prompt} \t Prediction: {pred}"""
            )

            def preprocess_pred(x):
                """Maps LLM speech classification labels to EOU detection states.

                Args:
                    x (str): LLM prediction containing Label1-4 classifications

                Returns:
                    str: "Complete" (Label1/3/4) or "Incomplete" (Label2/unrecognized)

                Note: Handles "Label1" and "Label 1" formats. Defaults to "Incomplete".
                """
                if (
                    "Label1" in x
                    or "Label 1" in x
                    or "Label3" in x
                    or "Label 3" in x
                    or "Label4" in x
                    or "Label 4" in x
                ):
                    return "Complete"
                else:
                    return "Incomplete"

            pred = preprocess_pred(pred)
        except Exception as e:
            logger.warning(f"Disabling Smart EOU detection due to error: {e}", exc_info=True)
            pred = "Complete"
        self.current_prediction = pred
        if pred == "Complete":
            # send transcript frame downwards if it is complete
            if isinstance(frame, InterimTranscriptionFrame):
                self.last_complete_interim_frame = frame
            if len(frame.text) > 0:
                logger.debug(f"Speech Planner: Pushing Complete Transcript to LLM at {datetime.now()}")
                await self.push_frame(StartInterruptionFrame())
                await self.push_frame(StopInterruptionFrame())
                await self.push_frame(TranscriptionFrame(frame.text, frame.user_id, frame.timestamp, frame.language))
        return

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and manage speech interactions.

        Args:
            frame: The incoming frame to process.
            direction: The direction of the frame flow.
        """
        if isinstance(frame, TranscriptionFrame):
            self.last_frame = frame
            if self.current_prediction == "Complete":  # need to reset every time new transcript comes
                logger.debug("Speech Planner: Holding final frame")
                self.last_frame = None
            else:
                await self._cancel_current_task()
                # await self.push_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM) # Triggering Interruption
                logger.debug(
                    f"Speech Planner: Pushing final frame when prediction is {self.current_prediction} "
                    f"and user_speaking is {self.user_speaking}"
                )
                if not self.user_speaking:  # Utilising acoustic VAD signal
                    logger.debug("Speech Planner: Sent final after VAD signal go ahead")
                    await self.push_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)
                    await self.push_frame(frame, FrameDirection.DOWNSTREAM)
                    self.last_frame = None
            self.last_processed_frame = None
            self.last_complete_interim_frame = None
            self.current_prediction = None
        elif isinstance(frame, InterimTranscriptionFrame):
            self.last_frame = frame
            logger.debug(f"Speech Planner: Last Complete Interim Frame {self.last_complete_interim_frame}")
            if self.last_processed_frame is None or (self.last_processed_frame.text.strip() != frame.text.strip()):
                self.current_prediction = None  # predictions need to be reset every time new partial comes
                await self._cancel_current_task()
                if self.current_prediction == "Complete":
                    await self.push_frame(
                        StartInterruptionFrame(), FrameDirection.DOWNSTREAM
                    )  # Triggering Interruption
                if not self.user_speaking:  # Utilising acoustic VAD signal
                    logger.debug("Speech Planner: Sent interim after VAD signal go ahead")
                    self._current_task = self.create_task(self._process_complete_context(frame))
                    self.last_processed_frame = frame
                    self.last_frame = None
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.latest_bot_started_speaking_frame_timestamp = datetime.now()
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.latest_bot_started_speaking_frame_timestamp = None
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("Speech Planner: Setting user speaking to True")
            self.user_speaking = True
            self.last_frame = None
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("Speech Planner: Setting user speaking to False")
            self.user_speaking = False
            if self.last_frame is not None:
                if isinstance(self.last_frame, TranscriptionFrame):
                    logger.debug("Speech Planner: Sent final after VAD signal go ahead")
                    await self.push_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)
                    await self.push_frame(self.last_frame, FrameDirection.DOWNSTREAM)
                elif isinstance(self.last_frame, InterimTranscriptionFrame):
                    logger.debug("Speech Planner: Sent interim after VAD signal go ahead")
                    self._current_task = self.create_task(self._process_complete_context(self.last_frame))
                    self.last_processed_frame = self.last_frame
                self.last_frame = None
            await self.push_frame(frame, direction)
        elif not isinstance(frame, OpenAILLMContextFrame | LLMMessagesFrame | LLMUpdateSettingsFrame):
            await super().process_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
