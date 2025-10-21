# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA Riva offline speech-to-text service implementation.

This module provides offline/batch transcription using NVIDIA Riva's ASR service.
Unlike the streaming RivaASRService, this service processes complete audio chunks
and returns full transcriptions, making it more reliable for certain use cases.
"""

import asyncio
import io
from collections.abc import AsyncGenerator

import riva.client
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced


@traceable
class RivaOfflineSTTService(STTService):
    """NVIDIA Riva Offline Speech Recognition service.

    Provides offline/batch speech recognition using Riva ASR models with support for:
        - Complete audio transcription (non-streaming)
        - Optional speaker diarization
        - Word-level timestamps
        - Higher reliability compared to streaming

    This service is ideal when:
        - You have complete audio chunks to process
        - Streaming issues are problematic
        - You need speaker diarization
        - Latency is less critical than accuracy
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        function_id: str = "4f9cd9c1-e8bf-4d7c-81ed-9525a1a3351a",
        version_id: str = "6f3f6b2e-86dc-4c7c-b1fa-601e3c2a2e84",
        language: Language | None = Language.EN_US,
        model: str = "parakeet-1.1b-en-US-asr-offline-asr-bls-ensemble",
        profanity_filter: bool = False,
        automatic_punctuation: bool = False,
        verbatim_transcripts: bool = False,
        enable_word_time_offsets: bool = True,
        enable_diarization: bool = False,
        max_speakers: int = 8,
        sample_rate: int = 16000,
        audio_channel_count: int = 1,
        max_alternatives: int = 1,
        use_ssl: bool = False,
        **kwargs,
    ):
        """Initializes the Riva Offline STT service.

        Args:
            api_key: NVIDIA API key for cloud access.
            server: Riva server address.
            function_id: NVCF function identifier.
            version_id: NVCF function version identifier.
            language: Language for recognition.
            model: ASR model name.
            profanity_filter: Enable profanity filtering.
            automatic_punctuation: Enable automatic punctuation.
            verbatim_transcripts: Enable verbatim transcripts.
            enable_word_time_offsets: Enable word-level timing information.
            enable_diarization: Enable speaker diarization.
            max_speakers: Maximum number of speakers to identify (used with diarization).
            sample_rate: Audio sample rate in Hz.
            audio_channel_count: Number of audio channels.
            max_alternatives: Maximum number of alternatives.
            use_ssl: Enable SSL connection.
            **kwargs: Additional arguments for STTService.

        Usage:
            If server is not set then it defaults to "grpc.nvcf.nvidia.com:443" and uses NVCF hosted models.
            Update function_id and version_id to use a different NVCF model. API key is required for NVCF hosted models.
            For using locally deployed Riva Speech Server, set server to "localhost:50051" and
            follow the quick start guide to setup the server.
        """
        super().__init__(**kwargs)
        
        self._profanity_filter = profanity_filter
        self._automatic_punctuation = automatic_punctuation
        self._verbatim_transcripts = verbatim_transcripts
        self._enable_word_time_offsets = enable_word_time_offsets
        self._enable_diarization = enable_diarization
        self._max_speakers = max_speakers
        self._language_code = language
        self._sample_rate = sample_rate
        self._model = model
        self._audio_channel_count = audio_channel_count
        self._max_alternatives = max_alternatives
        
        self.set_model_name(model)

        # Audio buffering for offline recognition
        self._audio_buffer = bytearray()
        self._is_speaking = False

        # Build metadata for authentication
        metadata = [
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ]
        
        if version_id:
            metadata.append(["function-version-id", version_id])

        if server == "grpc.nvcf.nvidia.com:443":
            use_ssl = True

        try:
            auth = riva.client.Auth(None, use_ssl, server, metadata)
            self._asr_service = riva.client.ASRService(auth)
            logger.info(f"Initialized Riva Offline STT Service (model: {model})")
        except Exception as e:
            logger.error(
                "In order to use Riva Offline STT Service, you will either need a locally "
                "deployed Riva Speech Server with ASR models (Follow riva quick start guide at "
                "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html and "
                "edit the config file to deploy which model you want to use and set the server url to "
                "localhost:50051), or you can set the NVIDIA_API_KEY environment "
                "variable to connect with NVCF hosted models."
            )
            raise Exception(f"Failed to initialize Riva ASR service: {e}") from e

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: True as this service supports metric generation.
        """
        return True

    async def process_frame(self, frame: Frame, direction):
        """Override to handle VAD signals and buffer audio.
        
        This method intercepts:
        - UserStartedSpeakingFrame: Mark as speaking (keep buffering)
        - UserStoppedSpeakingFrame: Process buffered audio
        - InputAudioRawFrame: Always buffer (to catch audio before VAD signals)
        """
        from pipecat.frames.frames import InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
        
        # Handle VAD signals
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug(f"User started speaking - buffer has {len(self._audio_buffer)} bytes")
            self._is_speaking = True
            # Don't clear buffer - keep any audio that arrived before VAD signal
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            duration = len(self._audio_buffer) / (self._sample_rate * 2)
            logger.info(f"User stopped speaking - processing {len(self._audio_buffer)} bytes ({duration:.2f}s) of buffered audio")
            self._is_speaking = False
            
            # Process the buffered audio
            if len(self._audio_buffer) > 0:
                # Send buffered audio for transcription
                buffered_audio = bytes(self._audio_buffer)
                async for result_frame in self.run_stt(buffered_audio):
                    if result_frame:
                        await self.push_frame(result_frame, direction)
                
                # Clear buffer after processing
                self._audio_buffer.clear()
            
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, InputAudioRawFrame):
            # Always buffer audio (even before VAD signals to catch beginning of speech)
            self._audio_buffer.extend(frame.audio)
            
            # Keep only last 10 seconds if not speaking (rolling buffer)
            if not self._is_speaking:
                max_pre_buffer = self._sample_rate * 2 * 10  # 10 seconds of audio
                if len(self._audio_buffer) > max_pre_buffer:
                    # Keep only the most recent audio
                    self._audio_buffer = self._audio_buffer[-max_pre_buffer:]
            else:
                # Log progress while speaking (every ~1 second)
                if len(self._audio_buffer) % 16000 == 0:
                    duration = len(self._audio_buffer) / (self._sample_rate * 2)
                    logger.debug(f"Buffering: {duration:.1f}s")
            # Don't call parent's process_frame to avoid immediate processing
            
        else:
            # Pass through all other frames
            await super().process_frame(frame, direction)

    async def _transcribe_audio(self, audio_data: bytes) -> tuple[str, dict]:
        """Internal method to perform offline transcription.

        Args:
            audio_data: Raw audio bytes to transcribe.

        Returns:
            A tuple of (transcript_text, metadata_dict) where metadata contains
            optional information like speakers, word timings, etc.
        """
        # Build recognition config
        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,  # Required for offline recognition
            language_code=self._language_code,
            max_alternatives=self._max_alternatives,
            profanity_filter=self._profanity_filter,
            enable_automatic_punctuation=self._automatic_punctuation,
            verbatim_transcripts=self._verbatim_transcripts,
            enable_word_time_offsets=self._enable_word_time_offsets,
            sample_rate_hertz=self._sample_rate,  # Also required
            audio_channel_count=self._audio_channel_count,  # Also required
        )

        # Add speaker diarization if enabled
        if self._enable_diarization:
            riva.client.asr.add_speaker_diarization_to_config(
                config,
                diarization_enable=True,
                diarization_max_speakers=self._max_speakers,
            )
            logger.debug(f"Speaker diarization enabled (max {self._max_speakers} speakers)")

        # Prepare audio data
        audio_buffer = io.BytesIO(audio_data).read()

        # Call Riva's offline recognition in a thread pool to avoid blocking
        def _sync_recognize():
            return self._asr_service.offline_recognize(audio_buffer, config)

        response = await asyncio.to_thread(_sync_recognize)

        # Parse response
        transcripts = []
        metadata = {
            "diarization_enabled": self._enable_diarization,
            "words": [] if self._enable_word_time_offsets else None,
            "speakers": [] if self._enable_diarization else None,
        }

        if self._enable_diarization:
            # Format with speaker labels
            for result in response.results:
                if result.alternatives and result.alternatives[0].words:
                    alt = result.alternatives[0]
                    current_speaker = None
                    current_segment = []

                    for word in alt.words:
                        speaker_tag = getattr(word, 'speaker_tag', None)

                        if speaker_tag is not None:
                            if current_speaker != speaker_tag:
                                # Speaker changed
                                if current_segment:
                                    transcripts.append(f"[Speaker {current_speaker}]: {' '.join(current_segment)}")
                                current_speaker = speaker_tag
                                current_segment = [word.word]
                            else:
                                current_segment.append(word.word)
                        else:
                            # No speaker tag
                            if current_speaker is not None and current_segment:
                                transcripts.append(f"[Speaker {current_speaker}]: {' '.join(current_segment)}")
                                current_speaker = None
                                current_segment = []
                            current_segment.append(word.word)

                    # Output final segment
                    if current_segment:
                        if current_speaker is not None:
                            transcripts.append(f"[Speaker {current_speaker}]: {' '.join(current_segment)}")
                        else:
                            transcripts.append(' '.join(current_segment))
        else:
            # Simple transcription without diarization
            for result in response.results:
                if result.alternatives:
                    transcripts.append(result.alternatives[0].transcript)

        # Combine all transcripts
        full_transcript = "\n".join(transcripts) if transcripts else ""

        logger.debug(f"Transcription complete: {len(full_transcript)} characters")
        return full_transcript, metadata

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="offline_asr")
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run offline speech-to-text recognition on buffered audio.

        Args:
            audio: The complete buffered audio data from a user utterance.

        Yields:
            TranscriptionFrame containing the full transcription, or ErrorFrame on failure.
        """
        if not audio or len(audio) == 0:
            logger.warning("Received empty audio data")
            yield None
            return

        audio_duration = len(audio) / (self._sample_rate * 2 * self._audio_channel_count)
        logger.info(f"Transcribing {len(audio)} bytes ({audio_duration:.2f}s) of buffered audio")

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        try:
            
            # Perform transcription
            transcript, metadata = await self._transcribe_audio(audio)

            await self.stop_ttfb_metrics()

            if transcript and transcript.strip():
                logger.debug(f"Offline transcription result: [{transcript}]")
                # Yield the transcription frame
                yield TranscriptionFrame(
                    text=transcript,
                    user_id="",
                    timestamp=time_now_iso8601(),
                    language=self._language_code,
                )
            else:
                logger.warning("Transcription returned empty result")
                yield None

        except Exception as e:
            # Handle grpc errors specially - they don't format well in f-strings
            # Use repr() which is safer for complex objects
            try:
                error_msg = repr(e)
            except Exception:
                error_msg = "Unknown error"
            
            # Log without using f-string with the exception object
            logger.error("Error during offline transcription", exc_info=True)
            yield ErrorFrame(f"Transcription failed: {error_msg}")
        finally:
            await self.stop_processing_metrics()

