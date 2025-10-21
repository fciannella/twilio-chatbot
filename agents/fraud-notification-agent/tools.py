import os
import json
from typing import Any, Dict, List

from langchain_core.tools import tool

# Robust logic import that avoids cross-module leakage during hot reloads
try:
    from . import logic as fraud_logic  # type: ignore
except Exception:
    import importlib.util as _ilu
    _dir = os.path.dirname(__file__)
    _logic_path = os.path.join(_dir, "logic.py")
    _spec = _ilu.spec_from_file_location("fraud_agent_logic", _logic_path)
    fraud_logic = _ilu.module_from_spec(_spec)  # type: ignore
    assert _spec and _spec.loader
    _spec.loader.exec_module(fraud_logic)  # type: ignore


# --- Identity Verification Tools ---

@tool
def verify_name_tool(session_id: str, phone: str, provided_name: str) -> str:
    """Verify the customer's identity by checking their name.
    
    This is the first step in fraud notification - verify we're speaking to the right person.
    Args:
        session_id: Current session ID
        phone: Customer phone number
        provided_name: The name provided by the caller
    
    Returns verification status. Only proceed with fraud details if verified=true.
    """
    return json.dumps(fraud_logic.verify_name(session_id, phone, provided_name))


# --- Fraud Information Tools ---

@tool
def get_fraud_alert_tool(phone: str) -> str:
    """Get the fraud alert details for this customer.
    
    Returns case ID, fraud type, severity, affected card, and actions already taken.
    Use this to inform the customer about what happened.
    """
    return json.dumps(fraud_logic.get_fraud_alert(phone))


@tool
def get_suspicious_transactions_tool(phone: str) -> str:
    """Get detailed list of suspicious transactions.
    
    Returns transaction details including amounts, merchants, locations, and timestamps.
    Use when customer wants to know specific transaction details.
    """
    return json.dumps(fraud_logic.get_suspicious_transactions(phone))


@tool
def confirm_fraud_tool(phone: str, is_fraud: bool) -> str:
    """Record customer's confirmation about whether the activity is fraudulent.
    
    Args:
        phone: Customer phone number
        is_fraud: True if customer confirms fraud, False if transactions are legitimate
    
    IMPORTANT: Always ask the customer explicitly before calling this tool.
    Do NOT assume - let the customer tell you if they recognize the transactions.
    """
    return json.dumps(fraud_logic.confirm_fraud(phone, is_fraud))


@tool
def get_next_steps_tool(phone: str) -> str:
    """Get next steps and recommendations for the customer.
    
    Returns immediate actions taken, follow-up steps, and contact information.
    Use after customer confirms fraud to explain what happens next.
    """
    return json.dumps(fraud_logic.get_next_steps(phone))


# --- Resolution Tools ---

@tool
def request_new_card_tool(phone: str, delivery_method: str = "standard") -> str:
    """Request a new card to be sent to the customer.
    
    Args:
        phone: Customer phone number
        delivery_method: "standard" (5-7 days, free) or "expedited" (2-3 days, may have fee)
    
    Returns delivery timeline and tracking information.
    """
    return json.dumps(fraud_logic.request_new_card(phone, delivery_method))


@tool
def dispute_charges_tool(phone: str, transaction_ids_json: str) -> str:
    """File disputes for specific fraudulent transactions.
    
    Args:
        phone: Customer phone number
        transaction_ids_json: JSON array of transaction IDs to dispute, e.g. '["TXN-123", "TXN-456"]'
    
    Returns dispute confirmation and timeline for provisional credit.
    """
    try:
        transaction_ids = json.loads(transaction_ids_json)
        if not isinstance(transaction_ids, list):
            return json.dumps({"error": "transaction_ids must be a JSON array"})
    except Exception:
        return json.dumps({"error": "invalid_json_format"})
    
    return json.dumps(fraud_logic.dispute_charges(phone, transaction_ids))


@tool
def mark_customer_notified_tool(phone: str) -> str:
    """Mark that the customer has been successfully notified about the fraud.
    
    Call this at the end of a successful conversation to update the case status.
    """
    return json.dumps(fraud_logic.mark_customer_notified(phone))

