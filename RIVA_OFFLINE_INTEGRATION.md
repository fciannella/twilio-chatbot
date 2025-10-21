# Riva Offline STT Integration Guide

## ✅ Integration Complete!

The `RivaOfflineSTTService` has been successfully integrated into `bot.py` and is now the **default STT service**.

## What Changed

### Files Modified
1. **`bot.py`** - Added Riva Offline STT as the primary STT service
2. **`env.example`** - Added Riva configuration variables

### New Files Created
1. **`nvidia_pipecat/services/riva_offline_stt.py`** - Main service implementation
2. **`nvidia_pipecat/services/riva_offline_stt_example.py`** - Standalone example script
3. **`nvidia_pipecat/services/RIVA_OFFLINE_STT_README.md`** - Complete documentation

## Setup Instructions

### 1. Update Your `.env` File

Add your NVIDIA API key to `.env`:

```bash
# NVIDIA Riva Configuration
RIVA_API_KEY=your-nvidia-api-key-here

# These are already set with defaults in bot.py:
RIVA_OFFLINE_ASR_FUNCTION_ID=4f9cd9c1-e8bf-4d7c-81ed-9525a1a3351a
RIVA_OFFLINE_ASR_VERSION_ID=6f3f6b2e-86dc-4c7c-b1fa-601e3c2a2e84
RIVA_ASR_LANGUAGE=en-US
```

### 2. Run Your Bot

That's it! The bot will now use Riva Offline STT by default:

```bash
source .venv/bin/activate
python server.py
```

## Configuration in bot.py

The bot now has **three STT options** (lines 67-96):

### Option 1: Riva Offline STT (✅ Currently Active)
```python
stt = RivaOfflineSTTService(
    api_key=os.getenv("RIVA_API_KEY"),
    function_id=os.getenv("RIVA_OFFLINE_ASR_FUNCTION_ID", "4f9cd9c1-e8bf-4d7c-81ed-9525a1a3351a"),
    version_id=os.getenv("RIVA_OFFLINE_ASR_VERSION_ID", "6f3f6b2e-86dc-4c7c-b1fa-601e3c2a2e84"),
    language=os.getenv("RIVA_ASR_LANGUAGE", "en-US"),
    sample_rate=16000,
    enable_diarization=False,  # Set True to identify different speakers
    automatic_punctuation=False,
    profanity_filter=False,
)
```

**Benefits:**
- ✅ More reliable than streaming
- ✅ Works with complete audio chunks
- ✅ Supports speaker diarization
- ✅ Simpler implementation (less prone to connection issues)
- ✅ Same async interface as other STT services

**When to use:** Default choice for most use cases

### Option 2: Riva Streaming STT (Commented Out)
```python
# Option 2: Riva Streaming STT (Has some reliability issues)
# stt = RivaASRService(...)
```

**When to use:** Only if you specifically need real-time interim results

### Option 3: OpenAI STT (Commented Out)
```python
# Option 3: OpenAI STT
# stt = OpenAISTTService(...)
```

**When to use:** If you prefer OpenAI's transcription service

## Switching Between Services

To switch to a different STT service, simply:
1. Comment out the current `stt = ...` line
2. Uncomment your preferred service
3. Restart the bot

Example - switching to OpenAI:
```python
# Option 1: Riva Offline STT (Recommended - More reliable than streaming)
# stt = RivaOfflineSTTService(...)

# Option 3: OpenAI STT
stt = OpenAISTTService(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-transcribe",
)
```

## Advanced Configuration

### Enable Speaker Diarization

If you need to identify different speakers in the conversation:

```python
stt = RivaOfflineSTTService(
    api_key=os.getenv("RIVA_API_KEY"),
    enable_diarization=True,      # Enable speaker detection
    max_speakers=3,                # Expected number of speakers
)
```

Output will include speaker labels:
```
[Speaker 0]: Hello, how are you doing today?
[Speaker 1]: I'm doing great, thanks for asking!
```

### Enable Punctuation

```python
stt = RivaOfflineSTTService(
    api_key=os.getenv("RIVA_API_KEY"),
    automatic_punctuation=True,    # Add punctuation automatically
)
```

### Use Local Riva Server

If you have a locally deployed Riva server:

```python
stt = RivaOfflineSTTService(
    api_key=None,  # Not needed for local
    server="localhost:50051",
    use_ssl=False,
)
```

## How It Works

### Architecture Overview

```
Twilio Audio → Pipeline Input → RivaOfflineSTTService → LLM → TTS → Pipeline Output
                                         ↓
                                 Complete audio chunks
                                         ↓
                                 Riva offline_recognize()
                                         ↓
                                 TranscriptionFrame
```

### Processing Flow

1. **Audio Input**: Silero VAD detects when user stops speaking
2. **Complete Chunk**: Entire audio segment sent to Riva Offline STT
3. **Transcription**: Riva processes the complete audio chunk
4. **Frame Output**: Returns a single `TranscriptionFrame` with full text
5. **Pipeline Continues**: Text flows to LLM → TTS → Output

### Key Differences from Streaming

| Aspect | Streaming (RivaASRService) | Offline (RivaOfflineSTTService) |
|--------|---------------------------|----------------------------------|
| **Processing** | Processes audio as it arrives | Processes complete audio chunk |
| **Results** | Multiple interim + final | Single final result |
| **Complexity** | High (queues, threads) | Low (simple async call) |
| **Reliability** | Can have connection issues | More stable |
| **Latency** | Lower (real-time) | Slightly higher |

## Testing

### Test the Service Standalone

You can test transcription without running the full bot:

```bash
source .venv/bin/activate

# Basic transcription
python nvidia_pipecat/services/riva_offline_stt_example.py test_audio.wav

# With speaker diarization
python nvidia_pipecat/services/riva_offline_stt_example.py test_audio.wav --diarize
```

### Test with Bot

1. Start the server: `python server.py`
2. Connect via Twilio
3. Speak and wait for VAD to detect end of speech
4. Check logs for transcription results

## Troubleshooting

### Issue: "Missing module" or auth errors

**Solution**: Check your environment variables:
```bash
# Make sure these are set in .env
RIVA_API_KEY=your-key-here
```

### Issue: Empty transcription results

**Possible causes:**
- Audio chunk is too short or silent
- Audio format mismatch (should be PCM, 16kHz, mono)
- VAD not properly detecting speech boundaries

**Solution**: Check your pipeline audio parameters:
```python
params=PipelineParams(
    audio_in_sample_rate=8000,   # Match your transport
    audio_out_sample_rate=8000,
    enable_metrics=True,
)
```

### Issue: Slow response times

This is expected for offline processing. The service waits for complete audio chunks.

**If too slow:**
1. Adjust VAD sensitivity to detect end-of-speech faster
2. Consider using streaming service for lower latency
3. Check network latency to Riva servers

### Issue: "Import could not be resolved" warnings

These are linter warnings about installed packages. They don't affect functionality.

**To fix (optional):**
- Ensure your IDE's Python interpreter is set to `.venv/bin/python`
- Or ignore these warnings - they're harmless

## Performance Considerations

### Latency
- **First token**: Higher than streaming (~1-2s for complete audio processing)
- **Total processing**: Comparable to streaming for full utterances
- **Network dependent**: Cloud API calls add latency

### Reliability
- ✅ **More reliable** than streaming (no connection state to maintain)
- ✅ **Simpler error handling** (single request/response)
- ✅ **Better for batch processing** of recorded audio

### When Offline is Better
- Processing complete audio files or recordings
- When streaming has reliability issues
- When speaker diarization is needed
- Batch transcription tasks

### When Streaming is Better
- Real-time transcription with very low latency requirements
- Need to show interim/partial results to user
- Live captioning applications

## Next Steps

1. ✅ Bot is ready to use with Riva Offline STT
2. 🔧 Optionally enable diarization if needed
3. 📊 Monitor metrics and adjust configuration
4. 🔄 Switch services if different needs arise

## Documentation

- **Full API docs**: `nvidia_pipecat/services/RIVA_OFFLINE_STT_README.md`
- **Example usage**: `nvidia_pipecat/services/riva_offline_stt_example.py`
- **Source code**: `nvidia_pipecat/services/riva_offline_stt.py`

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Review the full documentation in `RIVA_OFFLINE_STT_README.md`
3. Test with the standalone example script first
4. Compare with streaming service behavior if needed

