# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA AI Services."""

# Re-export OpenAI speech services when available
try:  # pragma: no cover - optional import
    from .openai_speech import (
        OpenAIRealtimeASRService,
        OpenAIRealtimeTTSService,
        OpenAITTSService,
    )
except Exception:  # noqa: BLE001
    OpenAIRealtimeASRService = None  # type: ignore
    OpenAIRealtimeTTSService = None  # type: ignore
    OpenAITTSService = None  # type: ignore

__all__ = [
    "OpenAIRealtimeASRService",
    "OpenAIRealtimeTTSService",
    "OpenAITTSService",
]
