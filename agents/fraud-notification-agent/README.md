# Fraud Notification Agent

A LangGraph-powered agent that makes outbound phone calls to notify customers about detected fraudulent activity on their accounts. The agent handles identity verification, explains the fraud incident, confirms the fraud with the customer, and provides next steps for resolution.

## Overview

This agent is designed for banks and financial institutions to proactively alert customers about suspicious activity, ensuring quick response times and minimizing fraud impact.

## Features

- **Simple Identity Verification**: Verifies customer by asking for first and last name only
- **Context Awareness**: Maintains full conversation history and remembers previous exchanges
- **Unclear Response Handling**: Politely handles gibberish or unclear responses by asking to repeat
- **Empathetic Communication**: Professional yet caring tone suitable for sensitive fraud discussions
- **Fraud Details**: Clearly explains detected fraud type, affected transactions, and actions taken
- **Customer Confirmation**: Asks customer to verify if transactions are legitimate or fraudulent
- **Resolution Support**: Guides customers through card replacement, charge disputes, and account security
- **Comprehensive Logging**: Tracks all interactions for compliance and quality assurance

## How to Use

### 1. Provide Phone Number

When the agent calls, it will have the customer's phone number from the outbound call setup. The agent will:
- Introduce itself as calling from the bank's fraud team
- Ask if it's a good time to discuss account security
- Request permission to verify identity

### 2. Identity Verification

The agent will:
- Ask: "For security purposes, can I please have your first and last name?"
- **ONLY asks for NAME** - no phone numbers, no codes, no other information
- Verify the name matches account records
- If unclear response or gibberish, politely ask to repeat
- Proceed with fraud details only after successful verification

**IMPORTANT**: 
- The agent will NOT share any fraud information until identity is verified
- The agent will NOT ask for phone verification or SMS codes
- Name verification is the ONLY identity check

### 3. Fraud Notification

After verification, the agent will:
- Explain what type of fraud was detected
- Describe the suspicious transactions (amounts, merchants, locations)
- Explain what actions have already been taken (card blocked, etc.)
- Provide the case ID for reference

### 4. Customer Confirmation

The agent will:
- Ask if the customer recognizes the transactions
- Listen to the customer's response
- Record whether it's confirmed fraud or a false positive

### 5. Next Steps

Depending on confirmation:
- **If Fraud**: Explain card replacement timeline, dispute process, provisional credit
- **If Legitimate**: Explain how to unblock the card and restore access
- Provide contact information for follow-up questions
- Offer fraud prevention tips

## Mock Data

The agent uses realistic mock data stored in `mock_data/`:

### Customers (`customers.json`)

Three test customers with different profiles:
- `+15551234567` (Alex Johnson): Visa ending in 4532
- `+447911123456` (Sarah Williams): Mastercard ending in 8791
- `+19175551234` (Michael Chen): Amex ending in 3421

### Fraud Cases (`fraud_cases.json`)

Three types of fraud scenarios:

1. **Unauthorized Transaction** (`+15551234567`):
   - Two large purchases totaling $2,140.50
   - Card temporarily blocked
   - High severity

2. **Card Skimming** (`+447911123456`):
   - Small gas station transaction flagged
   - Card on enhanced monitoring
   - Medium severity

3. **Account Takeover Attempt** (`+19175551234`):
   - Multiple failed logins from foreign IP
   - Account fully locked
   - Critical severity

### Customer Names

For testing purposes, verify identity with these names:
- `+15551234567` → Alex Johnson (can use "Alex" or "Alex Johnson")
- `+447911123456` → Sarah Williams (can use "Sarah" or "Sarah Williams")
- `+19175551234` → Michael Chen (can use "Michael" or "Michael Chen")

## Example Conversation

```
Agent: Hello, this is the fraud prevention team from your bank. We detected 
suspicious activity on your account and wanted to notify you immediately. 
Do you have a moment to discuss your account security?

Customer: Yes, what happened?

Agent: For security purposes, can I please have your first and last name?

Customer: Alex Johnson

Agent: Thank you, Alex. Your identity is confirmed. I'm calling because we 
detected two unauthorized transactions on your Visa ending in 4532. The first 
was for 1,250 dollars at Electronics Store XYZ, and the second was for 890 
dollars at an online marketplace. Both occurred today around 2 PM Pacific time.

Customer: I didn't make those purchases!

Agent: I understand this is concerning. Do you recognize these transactions, 
or should we treat them as fraudulent?

Customer: Definitely fraud. I was at work all day.

Agent: Thank you for confirming. We have already temporarily blocked your card 
to prevent further unauthorized charges. Here's what happens next: First, we'll 
send you a new card within 5 to 7 business days. Second, the disputed charges 
will be credited back to your account within 3 to 5 business days. Third, you 
should review your recent transactions for any other unfamiliar charges.

Customer: Do I need to file a report?

Agent: We've already filed the fraud report with case ID FRAUD-2025-00142. 
You don't need to do anything else right now. We'll email you a summary. 
If you have questions, you can reach our fraud hotline 24/7 at 1-800-FRAUD-HELP.

Customer: Thank you for letting me know.

Agent: You're welcome. We're here to protect your account. Is there anything 
else I can help you with today?

Customer: No, that's all.

Agent: Take care, and we'll have your new card to you soon.
```

### Example: Handling Unclear Response

```
Agent: For security purposes, can I please have your first and last name?

Customer: Um... what? Blah blah something...

Agent: I apologize, I didn't catch that clearly. Could you please repeat your 
first and last name?

Customer: Oh sorry, it's Alex Johnson

Agent: Thank you, Alex. Your identity is confirmed. Now, I need to inform you...
```

### Example: Context Awareness

```
Agent: We detected two unauthorized transactions totaling 2,140 dollars.

Customer: Two thousand?

Agent: Yes, that's correct - 2,140 dollars total. The first transaction was 
for 1,250 dollars and the second was for 890 dollars. Do you recognize these?

Customer: No, I didn't make those.

Agent: I understand. Since you don't recognize these transactions, we'll treat 
them as fraudulent...
```

## Available Tools

The agent has access to these tools:

1. **verify_name_tool**: Verifies customer identity by checking their name
2. **get_fraud_alert_tool**: Gets fraud case summary
3. **get_suspicious_transactions_tool**: Lists detailed transaction info
4. **confirm_fraud_tool**: Records customer's fraud confirmation
5. **get_next_steps_tool**: Provides resolution steps
6. **request_new_card_tool**: Orders replacement card (standard or expedited)
7. **dispute_charges_tool**: Files disputes for specific transactions
8. **mark_customer_notified_tool**: Marks case as customer-notified

## Configuration

### Environment Variables

- `REACT_MODEL`: OpenAI model to use (default: `gpt-4o`)
- `FRAUD_DEBUG`: Set to `1` to enable verbose logging
- `FRAUD_MAX_MSGS`: Max conversation history (default: 50) - higher value maintains better context

### LangGraph Setup

The agent is built using LangGraph's functional API:
- Uses `@entrypoint()` decorator for main agent function
- Uses `@task()` for LLM and tool calls
- Maintains conversation state across turns
- Supports streaming responses

## Integration with Pipecat

To use this agent with Pipecat for outbound calls:

1. Import the agent in your bot script:
```python
from agents.fraud_notification_agent.react_agent import agent
```

2. Use with LangGraphLLMService:
```python
from langgraph_llm_service import LangGraphLLMService

llm = LangGraphLLMService(
    base_url="http://127.0.0.1:2024",  # Your LangGraph server
    assistant="fraud-notification-agent",
    user_email="system@bank.com",
    stream_mode="values"
)
```

3. Configure the transport with customer phone number:
```python
# Pass phone number in configurable context
config = {
    "configurable": {
        "phone": "+15551234567",
        "thread_id": call_session_id
    }
}
```

## Deployment

### Local Development

1. Start LangGraph server:
```bash
langgraph dev
```

2. Test the agent:
```bash
langgraph studio
```

### Production

Deploy to LangGraph Cloud or Platform:
```bash
langgraph deploy
```

## Compliance & Security

### PCI DSS Compliance
- Never logs full card numbers (only last 4 digits)
- Verification codes are never exposed to LLM
- All transactions are encrypted in transit

### Call Recording
- All calls should be recorded for quality assurance
- Customers should be notified of recording per regulations

### Data Retention
- Conversation logs: 7 years (compliance requirement)
- Verification codes: Expire after use or 15 minutes
- Fraud case data: Per institution policy

## Fraud Types Supported

1. **Unauthorized Transaction**: Purchases made without customer knowledge
2. **Card Skimming**: Card data stolen at physical locations
3. **Account Takeover Attempt**: Unauthorized access attempts to online banking

## Next Steps Templates

The agent provides fraud-type-specific guidance:
- Immediate actions taken by the bank
- Follow-up actions for the customer
- Fraud prevention tips
- Contact information for support

## Testing

Use the mock data to test different scenarios:

```python
# Test unauthorized transaction scenario
phone = "+15551234567"
customer_name = "Alex Johnson"  # or just "Alex"

# Test card skimming scenario
phone = "+447911123456"
customer_name = "Sarah Williams"  # or just "Sarah"

# Test account takeover scenario
phone = "+19175551234"
customer_name = "Michael Chen"  # or just "Michael"
```

## Troubleshooting

### Agent won't share fraud details
- Ensure identity verification completed successfully
- Check that `verify_name_tool` returned `verified: true`

### Name verification fails
- Check phone number format (E.164: +1234567890)
- Verify customer exists in mock_data/customers.json
- Name matching is flexible (accepts first name, last name, or full name)
- Names are case-insensitive

## Support

For questions or issues:
- Review agent logs in `app.log`
- Enable debug mode: `export FRAUD_DEBUG=1`
- Check LangGraph server logs
- Review conversation history in LangGraph Studio

## License

BSD 2-Clause License (same as parent project)

