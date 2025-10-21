"""
Example: How to integrate the fraud-notification-agent with Pipecat bot.py

This shows how to modify bot.py to use the fraud notification agent instead of
the default OpenAI LLM service.
"""

# In your bot.py, replace the OpenAI LLM with LangGraph LLM Service:

# BEFORE:
# from pipecat.services.openai.llm import OpenAILLMService
# llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

# AFTER:
from langgraph_llm_service import LangGraphLLMService

llm = LangGraphLLMService(
    base_url=os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024"),
    assistant="fraud-notification-agent",  # Your agent name
    user_email="fraud-system@bank.com",
    stream_mode="values",
    debug_stream=False,
)

# The rest of your bot.py remains the same:
# - STT (speech-to-text) setup
# - TTS (text-to-speech) setup  
# - Pipeline configuration
# - Transport setup

# IMPORTANT: Pass the customer's phone number in the configuration:

# In the bot() function, add configurable context:
async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    
    # ... existing transport setup ...
    
    # Extract customer phone from call data or environment
    customer_phone = os.getenv("CUSTOMER_PHONE", "+15551234567")
    
    # Create context with phone number
    context = OpenAILLMContext(
        messages=[],  # Will be populated by agent
        configurable={
            "phone": customer_phone,
            "thread_id": transport_data.get("call_id", "unknown")
        }
    )
    
    # ... rest of pipeline setup ...

# Full bot.py example modification:
"""
import os
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from langgraph_llm_service import LangGraphLLMService

async def run_bot(transport: BaseTransport, handle_sigint: bool, customer_phone: str):
    # Use LangGraph agent instead of OpenAI
    llm = LangGraphLLMService(
        base_url="http://127.0.0.1:2024",
        assistant="fraud-notification-agent",
        user_email="fraud@bank.com",
        stream_mode="values",
    )
    
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
    )
    
    # No initial messages - agent will start the conversation
    messages = []
    
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    
    # Configure with customer phone
    context.configurable = {
        "phone": customer_phone,
        "thread_id": str(uuid.uuid4())
    }
    
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    
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
        # Agent will initiate the conversation
        logger.info(f"Fraud notification call started for {customer_phone}")
    
    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)
"""

