#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pathlib import Path

from nvidia_pipecat.pipeline.ace_pipeline_runner import ACEPipelineRunner, PipelineMetadata

from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService

from nvidia_pipecat.transports.network.ace_fastapi_websocket import (
    ACETransport,
    ACETransportParams,
)
from nvidia_pipecat.transports.services.ace_controller.routers.websocket_router import router as websocket_router
from nvidia_pipecat.utils.logging import setup_default_ace_logging

from langgraph_llm_service import LangGraphLLMService


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def run_bot(transport: BaseTransport, handle_sigint: bool):
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))


    llm = LangGraphLLMService(
        base_url=os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024"),
        assistant="fraud-notification-agent",
        user_email=os.getenv("USER_EMAIL", "test@example.com"),
        stream_mode=os.getenv("LANGGRAPH_STREAM_MODE", "values"),
        debug_stream=os.getenv("LANGGRAPH_DEBUG_STREAM", "false").lower() == "true",
    )


    stt = RivaASRService(
        # server=os.getenv("RIVA_ASR_URL", "localhost:50051"), # default url is grpc.nvcf.nvidia.com:443
        api_key=os.getenv("RIVA_API_KEY"),
        function_id=os.getenv("NVIDIA_ASR_FUNCTION_ID", "52b117d2-6c15-4cfa-a905-a67013bee409"),
        language=os.getenv("RIVA_ASR_LANGUAGE", "en-US"),
        sample_rate=16000,
        model=os.getenv("RIVA_ASR_MODEL", "parakeet-1.1b-en-US-asr-streaming-silero-vad-asr-bls-ensemble"),
    )

    tts = RivaTTSService(
        # server=os.getenv("RIVA_TTS_URL", "localhost:50051"), # default url is grpc.nvcf.nvidia.com:443
        api_key=os.getenv("RIVA_API_KEY"),
        function_id=os.getenv("NVIDIA_TTS_FUNCTION_ID", "4e813649-d5e4-4020-b2be-2b918396d19d"),
        voice_id=os.getenv("RIVA_TTS_VOICE_ID", "Magpie-ZeroShot.Female-1"),
        model=os.getenv("RIVA_TTS_MODEL", "magpie_tts_ensemble-Magpie-ZeroShot"),
        language=os.getenv("RIVA_TTS_LANGUAGE", "en-US"),
        zero_shot_audio_prompt_file=(
            Path(os.getenv("ZERO_SHOT_AUDIO_PROMPT")) if os.getenv("ZERO_SHOT_AUDIO_PROMPT") else None
        ),
    )

    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    # )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly assistant making an outbound phone call. Your responses will be read aloud, "
                "so keep them concise and conversational. Avoid special characters or formatting. "
                "Begin by politely greeting the person and explaining why you're calling."
            ),
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the outbound conversation, waiting for the user to speak first
        logger.info("Starting outbound call conversation")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Outbound call ended")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    # transport = ACETransport(
    #     websocket=pipeline_metadata.websocket,
    #     params=ACETransportParams(
    #         vad_analyzer=SileroVADAnalyzer(),
    #     ),
    # )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    handle_sigint = runner_args.handle_sigint

    await run_bot(transport, handle_sigint)
