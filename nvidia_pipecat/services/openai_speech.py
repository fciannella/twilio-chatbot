#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""OpenAI Realtime speech services (ASR/TTS).

Currently implements streaming Speech-to-Text (ASR) over OpenAI's Realtime WebSocket API
using server-side VAD and input transcription events. TTS class is provided as a stub.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import os
from collections.abc import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601

from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value is not None else default


@traceable
class OpenAIRealtimeASRService(STTService):
    """OpenAI Realtime ASR (Speech-to-Text) service.

    Streams 16 kHz mono PCM16 audio chunks to the OpenAI Realtime API and emits
    interim and final transcription frames.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        realtime_model: str | None = None,
        stt_model: str | None = None,
        sample_rate: int = 16000,
        audio_channel_count: int = 1,
        server_vad: bool = True,
        vad_threshold: float | None = 0.5,
        vad_prefix_padding_ms: int | None = 300,
        vad_silence_duration_ms: int | None = 600,
        ws_ping_interval: int = 20,
        ws_ping_timeout: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Configuration
        self._api_key = api_key or _get_env("OPENAI_API_KEY")
        if not self._api_key:
            logger.warning("OpenAIRealtimeASRService: OPENAI_API_KEY not set; service will fail to connect.")

        self._realtime_model = (
            realtime_model
            or _get_env("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
            or "gpt-4o-realtime-preview"
        )
        self._stt_model = stt_model or _get_env("OPENAI_STT_MODEL", "gpt-4o-transcribe") or "gpt-4o-transcribe"
        self._sample_rate = sample_rate
        self._audio_channel_count = audio_channel_count
        self._server_vad = server_vad
        self._vad_threshold = vad_threshold
        self._vad_prefix_padding_ms = vad_prefix_padding_ms
        self._vad_silence_duration_ms = vad_silence_duration_ms
        self._ws_ping_interval = ws_ping_interval
        self._ws_ping_timeout = ws_ping_timeout

        # Runtime state
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._ws = None
        self._ws_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None
        self._receiver_task: asyncio.Task | None = None
        self._partial_buffer: str = ""

        # Construct WS URL and headers
        self._ws_url = f"wss://api.openai.com/v1/realtime?model={self._realtime_model}"
        self._ws_headers = [
            ("Authorization", f"Bearer {self._api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]

    def can_generate_metrics(self) -> bool:
        return False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Start websocket main loop in background
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = self.create_task(self._ws_main())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._shutdown()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._shutdown()

    async def _shutdown(self):
        # Cancel sender/receiver first
        if self._sender_task and not self._sender_task.done():
            await self.cancel_task(self._sender_task)
        if self._receiver_task and not self._receiver_task.done():
            await self.cancel_task(self._receiver_task)
        self._sender_task = None
        self._receiver_task = None

        # Close websocket by cancelling _ws_task (which owns the connection)
        if self._ws_task and not self._ws_task.done():
            await self.cancel_task(self._ws_task)
        self._ws_task = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        # Enqueue audio for the sender task
        try:
            await self._audio_queue.put(audio)
        except Exception as e:
            logger.error(f"{self.name} failed to enqueue audio: {e}")
        # Nothing to yield immediately (streaming handled in background)
        yield None

    async def _ws_main(self):
        """Owns the websocket connection and spawns sender/receiver tasks."""
        try:
            # Late import to avoid hard dependency errors during initialization
            import websockets

            kwargs = {
                "extra_headers": self._ws_headers,
                "ping_interval": self._ws_ping_interval,
                "ping_timeout": self._ws_ping_timeout,
                "max_size": 10_000_000,
            }

            logger.info(f"Connecting to OpenAI Realtime WS model={self._realtime_model}")
            async with websockets.connect(self._ws_url, **kwargs) as ws:
                self._ws = ws
                await self._configure_session()

                # Spawn sender and receiver tasks bound to this ws
                self._sender_task = self.create_task(self._ws_sender())
                self._receiver_task = self.create_task(self._ws_receiver())

                done, pending = await asyncio.wait(
                    {self._sender_task, self._receiver_task},
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                for t in pending:
                    t.cancel()
        except asyncio.CancelledError:
            logger.debug("OpenAIRealtimeASRService websocket task cancelled")
            raise
        except Exception as e:
            logger.error(f"OpenAIRealtimeASRService websocket error: {e}")
        finally:
            self._ws = None
            self._sender_task = None
            self._receiver_task = None

    async def _configure_session(self):
        if not self._ws:
            return
        payload: dict = {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                # Server-side VAD and transcription
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": False,
                }
                if self._server_vad
                else None,
                "input_audio_transcription": {
                    "model": self._stt_model,
                },
            },
        }

        # Inject VAD knob values only when provided
        if self._server_vad and isinstance(payload.get("session", {}).get("turn_detection"), dict):
            vad_cfg = payload["session"]["turn_detection"]
            if self._vad_threshold is not None:
                vad_cfg["threshold"] = float(self._vad_threshold)
            if self._vad_prefix_padding_ms is not None:
                vad_cfg["prefix_padding_ms"] = int(self._vad_prefix_padding_ms)
            if self._vad_silence_duration_ms is not None:
                vad_cfg["silence_duration_ms"] = int(self._vad_silence_duration_ms)

        await self._ws.send(json.dumps(payload))

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="asr")
    async def _ws_sender(self):
        """Reads PCM16 bytes from queue and sends append events to the server."""
        try:
            while True:
                chunk: bytes = await self._audio_queue.get()
                if not self._ws:
                    continue
                try:
                    b64 = base64.b64encode(chunk).decode("ascii")
                    await self._ws.send(
                        json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": b64,
                        })
                    )
                except Exception as e:
                    logger.warning(f"Failed to send audio chunk: {e}")
        except asyncio.CancelledError:
            raise

    async def _push_interim(self, text: str):
        if not text:
            return
        # Debounce: only push if changed
        if self._partial_buffer.rstrip() == text.rstrip():
            return
        self._partial_buffer = text
        await self.push_frame(
            InterimTranscriptionFrame(text, "", time_now_iso8601(), None)
        )

    async def _push_final(self, text: str):
        if not text:
            return
        self._partial_buffer = ""
        await self.push_frame(
            TranscriptionFrame(text, "", time_now_iso8601(), None)
        )

    async def _ws_receiver(self):
        """Receives transcription events and emits frames."""
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except Exception:
                    continue

                etype = event.get("type", "")
                if etype == "conversation.item.input_audio_transcription.delta":
                    delta = event.get("delta", "")
                    if delta:
                        # Compose into cumulative partial text
                        text = (self._partial_buffer + delta) if self._partial_buffer else delta
                        await self._push_interim(text)

                elif etype == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "")
                    await self._push_final(text)

                elif etype == "error":
                    logger.error(f"OpenAI Realtime error: {event}")
                elif etype == "input_audio_buffer.speech_started":
                    try:
                        await self.push_frame(UserStartedSpeakingFrame())
                    except Exception:
                        pass
                elif etype == "input_audio_buffer.committed":
                    try:
                        await self.push_frame(UserStoppedSpeakingFrame())
                    except Exception:
                        pass
        except asyncio.CancelledError:
            raise


class OpenAIRealtimeTTSService(TTSService):
    """Stub for OpenAI Realtime TTS service."""

    def __init__(self, **kwargs):  # noqa: D401
        super().__init__(**kwargs)
        logger.warning("OpenAIRealtimeTTSService is a stub and not wired to Realtime yet.")

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:  # pragma: no cover - stub
        raise NotImplementedError("OpenAIRealtimeTTSService Realtime path not implemented.")


@traceable
class OpenAITTSService(TTSService):
    """OpenAI Text-to-Speech over REST streaming.

    Uses OpenAI's `audio.speech` streaming API to synthesize PCM audio.
    """

    OPENAI_SAMPLE_RATE = 24000

    _VALID_VOICES = {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
        "verse",
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        voice: str = "shimmer",
        model: str = "gpt-4o-mini-tts",
        sample_rate: int | None = None,
        instructions: str | None = None,
        speed: float | None = None,
        **kwargs,
    ):
        # Default to native 24 kHz to avoid quality loss from resampling unless caller overrides
        sr = sample_rate or self.OPENAI_SAMPLE_RATE
        if sr != self.OPENAI_SAMPLE_RATE:
            logger.info(
                f"OpenAI TTS renders at {self.OPENAI_SAMPLE_RATE} Hz; resampling to {sr} Hz for output."
            )

        super().__init__(sample_rate=sr, **kwargs)

        # Client is created lazily to avoid import errors if openai not installed at import-time
        self._api_key = api_key or _get_env("OPENAI_API_KEY")
        self._base_url = base_url or _get_env("OPENAI_BASE_URL")
        self._client = None

        # Allow env overrides for voice/speed
        env_voice = _get_env("OPENAI_TTS_VOICE")
        selected_voice = env_voice if env_voice else voice

        env_speed = _get_env("OPENAI_TTS_SPEED")
        try:
            selected_speed = float(env_speed) if env_speed is not None else speed
        except Exception:
            selected_speed = speed

        self._speed = selected_speed
        self.set_model_name(model)
        # Validate voice
        if selected_voice not in self._VALID_VOICES:
            logger.warning(f"Unknown OpenAI TTS voice '{selected_voice}', defaulting to 'alloy'")
            selected_voice = "alloy"
        self.set_voice(selected_voice)
        self._instructions = instructions
        self._ratecv_state = None
        # Carry a leftover byte to ensure PCM16 sample alignment across frames
        self._pcm_carry: bytes = b""

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "openai package is required for OpenAITTSService. Install 'openai'"
                ) from e
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # No warning: we resample to the configured sample rate

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="tts")
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        self._ensure_client()
        try:
            await self.start_ttfb_metrics()

            create_params = {
                "input": text,
                "model": self.model_name,
                "voice": self._voice_id,
                "response_format": "pcm",
            }
            if self._instructions:
                create_params["instructions"] = self._instructions
            if self._speed is not None:
                create_params["speed"] = self._speed

            # Stream response
            async with self._client.audio.speech.with_streaming_response.create(**create_params) as r:
                if getattr(r, "status_code", 200) != 200:
                    error_text = await r.text()
                    logger.error(f"{self} error getting audio (status: {r.status_code}, error: {error_text})")
                    yield ErrorFrame(f"OpenAI TTS error (status: {r.status_code}, error: {error_text})")
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                first_chunk = True
                chunk_size = getattr(self, "chunk_size", 4096)
                async for chunk in r.iter_bytes(chunk_size):
                    if not chunk:
                        continue
                    # Ensure 16-bit sample alignment by carrying last odd byte to next chunk
                    if self._pcm_carry:
                        chunk = self._pcm_carry + chunk
                        self._pcm_carry = b""
                    if len(chunk) % 2 == 1:
                        self._pcm_carry = chunk[-1:]
                        chunk = chunk[:-1]
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False
                    out_bytes = chunk
                    if self.sample_rate != self.OPENAI_SAMPLE_RATE:
                        try:
                            out_bytes, self._ratecv_state = audioop.ratecv(
                                chunk,
                                2,
                                1,
                                self.OPENAI_SAMPLE_RATE,
                                self.sample_rate,
                                self._ratecv_state,
                            )
                        except Exception as e:
                            logger.warning(f"TTS resample failed, passing through 24kHz: {e}")
                            out_bytes = chunk
                    yield TTSAudioRawFrame(out_bytes, self.sample_rate, 1)
                yield TTSStoppedFrame()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"{self} error generating TTS: {e}")
            # Yield an error frame so downstream can react
            try:
                yield ErrorFrame(str(e))
            except Exception:
                # If yielding after exception fails, just swallow
                pass


