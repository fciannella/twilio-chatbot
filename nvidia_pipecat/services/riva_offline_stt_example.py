#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Example usage of RivaOfflineSTTService.

This script demonstrates how to use the offline Riva STT service to transcribe audio files.
"""

import argparse
import asyncio
import os
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

from nvidia_pipecat.services.riva_offline_stt import RivaOfflineSTTService
from pipecat.frames.frames import TranscriptionFrame

# Load environment variables
load_dotenv()


async def transcribe_audio_file(
    audio_path: Path,
    enable_diarization: bool = False,
    max_speakers: int = 8,
) -> str:
    """Transcribe an audio file using RivaOfflineSTTService.

    Args:
        audio_path: Path to the audio file (should be WAV, mono, 16kHz)
        enable_diarization: Whether to enable speaker diarization
        max_speakers: Maximum number of speakers to identify

    Returns:
        The transcribed text
    """
    # Get credentials from environment
    api_key = os.getenv('RIVA_BEARER_TOKEN')
    function_id = os.getenv('RIVA_FUNCTION_ID', '4f9cd9c1-e8bf-4d7c-81ed-9525a1a3351a')
    version_id = os.getenv('RIVA_FUNCTION_VERSION_ID', '6f3f6b2e-86dc-4c7c-b1fa-601e3c2a2e84')
    server = os.getenv('RIVA_URI', 'grpc.nvcf.nvidia.com:443')

    if not api_key:
        raise ValueError("RIVA_BEARER_TOKEN environment variable is required")

    logger.info(f"Initializing RivaOfflineSTTService...")
    
    # Create the service
    stt_service = RivaOfflineSTTService(
        api_key=api_key,
        server=server,
        function_id=function_id,
        version_id=version_id,
        enable_diarization=enable_diarization,
        max_speakers=max_speakers,
        automatic_punctuation=False,
        profanity_filter=False,
    )

    # Read audio file
    logger.info(f"Reading audio file: {audio_path}")
    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    logger.info(f"Processing {len(audio_data)} bytes of audio...")

    # Process audio through the service
    transcript_text = ""
    async for frame in stt_service.run_stt(audio_data):
        if frame and isinstance(frame, TranscriptionFrame):
            transcript_text = frame.text
            logger.info(f"Received transcription: {len(transcript_text)} characters")

    return transcript_text


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Riva Offline STT Service"
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file (preferably WAV, mono, 16kHz)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Maximum number of speakers (default: 8)",
    )
    args = parser.parse_args()

    if not args.audio_file.exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        return 1

    try:
        transcript = await transcribe_audio_file(
            args.audio_file,
            enable_diarization=args.diarize,
            max_speakers=args.max_speakers,
        )

        print("\n" + "=" * 80)
        print("TRANSCRIPTION:")
        print("=" * 80)
        print(transcript)
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))

