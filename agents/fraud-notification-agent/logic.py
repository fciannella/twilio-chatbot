import os
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


# In-memory stores and fixture cache
_FIXTURE_CACHE: Dict[str, Any] = {}
_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _fixtures_dir() -> Path:
    return Path(__file__).parent / "mock_data"


def _load_fixture(name: str) -> Any:
    if name in _FIXTURE_CACHE:
        return _FIXTURE_CACHE[name]
    p = _fixtures_dir() / name
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _FIXTURE_CACHE[name] = data
    return data


def _normalize_phone(phone: Optional[str]) -> Optional[str]:
    """Normalize phone number to E.164 format."""
    if not isinstance(phone, str) or not phone.strip():
        return None
    s = phone.strip()
    digits = ''.join(ch for ch in s if ch.isdigit() or ch == '+')
    if digits.startswith('+'):
        return digits
    return f"+{digits}"


def _get_customer(phone: str) -> Dict[str, Any]:
    """Get customer information by phone number."""
    ph = _normalize_phone(phone) or ""
    data = _load_fixture("customers.json")
    return dict((data.get("customers", {}) or {}).get(ph, {}))


def _get_fraud_case(phone: str) -> Dict[str, Any]:
    """Get fraud case information for a customer."""
    ph = _normalize_phone(phone) or ""
    data = _load_fixture("fraud_cases.json")
    return dict((data.get("fraud_cases", {}) or {}).get(ph, {}))


def _find_customer_by_name(provided_name: str) -> Optional[Dict[str, Any]]:
    """Find customer by name across all customers.
    
    Returns the customer dict with an added 'phone' key if found, None otherwise.
    """
    if not provided_name or not isinstance(provided_name, str):
        return None
    
    provided = provided_name.strip().lower()
    if not provided:
        return None
    
    data = _load_fixture("customers.json")
    customers = data.get("customers", {})
    
    provided_parts = provided.split()
    
    for phone, customer in customers.items():
        expected_name = customer.get("name", "").lower()
        name_parts = expected_name.split()
        
        # Check if any significant part matches (first or last name)
        match_found = False
        for provided_part in provided_parts:
            if len(provided_part) >= 2:  # Only check meaningful parts
                for name_part in name_parts:
                    if provided_part in name_part or name_part in provided_part:
                        match_found = True
                        break
        
        # Also check full name match
        if provided in expected_name or expected_name in provided:
            match_found = True
        
        if match_found:
            # Return customer with phone number included
            result = dict(customer)
            result['phone'] = phone
            return result
    
    return None


def _get_next_steps_template(fraud_type: str) -> Dict[str, Any]:
    """Get next steps template based on fraud type."""
    data = _load_fixture("next_steps.json")
    templates = data.get("next_steps_templates", {})
    return dict(templates.get(fraud_type, {}))


def _mask_phone(phone: str) -> str:
    """Mask phone number for privacy."""
    s = _normalize_phone(phone) or ""
    tail = s[-2:] if len(s) >= 2 else s
    return f"***-***-**{tail}"


# --- Identity Verification ---

def verify_name(session_id: str, phone: str, provided_name: str) -> Dict[str, Any]:
    """Verify the caller's identity by checking their name.
    
    Args:
        session_id: Current session ID
        phone: Customer phone number (may be invalid/fake)
        provided_name: Name provided by the caller
    
    Returns:
        {
            "verified": bool,
            "session_id": str,
            "phone": str,
            "customer_name": str (if verified),
            "error": str (optional)
        }
    """
    ph = _normalize_phone(phone) or ""
    cust = _get_customer(ph)
    
    # If phone lookup fails, try to find customer by name
    if not cust:
        cust = _find_customer_by_name(provided_name)
        if cust:
            # Customer was found by name, use their actual phone
            ph = cust.get('phone', '')
    
    if not cust:
        return {
            "verified": False,
            "error": "customer_not_found",
            "message": "We couldn't find your account information."
        }
    
    expected_name = cust.get("name", "").lower()
    provided = provided_name.strip().lower()
    
    # Simple name matching - check if provided name matches any part of full name
    # This handles cases like "Alex" matching "Alex Johnson"
    name_parts = expected_name.split()
    provided_parts = provided.split()
    
    # Check if any significant part matches (first or last name)
    match_found = False
    for provided_part in provided_parts:
        if len(provided_part) >= 2:  # Only check meaningful parts
            for name_part in name_parts:
                if provided_part in name_part or name_part in provided_part:
                    match_found = True
                    break
    
    # Also check full name match
    if provided in expected_name or expected_name in provided:
        match_found = True
    
    if match_found:
        sess = _SESSIONS.get(session_id) or {}
        sess["verified"] = True
        sess["phone"] = ph
        sess["verified_at"] = datetime.utcnow().isoformat() + "Z"
        sess["customer_name"] = cust.get("name")
        _SESSIONS[session_id] = sess
        
        return {
            "verified": True,
            "session_id": session_id,
            "phone": ph,
            "customer_name": cust.get("name")
        }
    else:
        return {
            "verified": False,
            "error": "name_mismatch",
            "message": "The name provided doesn't match our records. For security purposes, I'll need to verify your identity another way."
        }


# --- Fraud Information ---

def get_fraud_alert(phone: str) -> Dict[str, Any]:
    """Get fraud alert details for the customer.
    
    Returns comprehensive fraud case information including:
    - Case ID and severity
    - Fraud type and detection time
    - Transactions involved
    - Actions already taken
    """
    ph = _normalize_phone(phone) or ""
    cust = _get_customer(ph)
    if not cust:
        return {"error": "customer_not_found"}
    
    fraud_case = _get_fraud_case(ph)
    if not fraud_case:
        return {"error": "no_fraud_case_found"}
    
    return {
        "case_id": fraud_case.get("case_id"),
        "customer_name": cust.get("name"),
        "account_number": cust.get("account_number"),
        "detected_at": fraud_case.get("detected_at"),
        "fraud_type": fraud_case.get("fraud_type"),
        "severity": fraud_case.get("severity"),
        "card_last_four": fraud_case.get("card_last_four"),
        "card_status": fraud_case.get("card_status"),
        "total_amount": fraud_case.get("total_amount"),
        "transaction_count": len(fraud_case.get("transactions", [])),
        "actions_taken": fraud_case.get("actions_taken", []),
        "suspicious_activity": fraud_case.get("suspicious_activity", [])
    }


def get_suspicious_transactions(phone: str) -> Dict[str, Any]:
    """Get detailed list of suspicious transactions.
    
    Returns:
        {
            "transactions": [...],
            "total_amount": float,
            "currency": str
        }
    """
    ph = _normalize_phone(phone) or ""
    fraud_case = _get_fraud_case(ph)
    if not fraud_case:
        return {"error": "no_fraud_case_found"}
    
    transactions = fraud_case.get("transactions", [])
    total = fraud_case.get("total_amount", 0.0)
    
    # Get currency from first transaction or default to USD
    currency = "USD"
    if transactions:
        currency = transactions[0].get("currency", "USD")
    
    return {
        "transactions": transactions,
        "total_amount": total,
        "currency": currency,
        "transaction_count": len(transactions)
    }


def confirm_fraud(phone: str, is_fraud: bool) -> Dict[str, Any]:
    """Customer confirms or denies the fraud.
    
    Args:
        phone: Customer phone number
        is_fraud: True if customer confirms it's fraud, False if legitimate
    
    Returns:
        Status and next steps information
    """
    ph = _normalize_phone(phone) or ""
    fraud_case = _get_fraud_case(ph)
    if not fraud_case:
        return {"error": "no_fraud_case_found"}
    
    case_id = fraud_case.get("case_id")
    
    if is_fraud:
        # Customer confirms fraud
        return {
            "status": "confirmed_fraud",
            "case_id": case_id,
            "message": "Thank you for confirming. We have taken immediate action to secure your account.",
            "card_status": fraud_case.get("card_status"),
            "next_action": "get_next_steps"
        }
    else:
        # Customer says transactions are legitimate
        return {
            "status": "false_positive",
            "case_id": case_id,
            "message": "Thank you for letting us know these are your transactions. We will unblock your card and restore full access.",
            "card_status": "unblocking_in_progress",
            "estimated_time": "within 2 hours"
        }


def get_next_steps(phone: str) -> Dict[str, Any]:
    """Get next steps and recommendations for the customer.
    
    Returns detailed instructions on:
    - Immediate actions taken
    - Follow-up actions customer should take
    - Contact information for support
    """
    ph = _normalize_phone(phone) or ""
    fraud_case = _get_fraud_case(ph)
    if not fraud_case:
        return {"error": "no_fraud_case_found"}
    
    fraud_type = fraud_case.get("fraud_type", "")
    template = _get_next_steps_template(fraud_type)
    
    if not template:
        # Generic fallback
        return {
            "immediate": ["Your account has been secured"],
            "followup": ["Monitor your account for any suspicious activity"],
            "contact_info": {
                "fraud_hotline": "1-800-FRAUD-HELP",
                "email": "fraud@bank.example.com",
                "hours": "24/7"
            }
        }
    
    return {
        "case_id": fraud_case.get("case_id"),
        "fraud_type": fraud_type,
        "immediate_steps": template.get("immediate", []),
        "followup_steps": template.get("followup", []),
        "contact_info": template.get("contact_info", {})
    }


def request_new_card(phone: str, delivery_method: str = "standard") -> Dict[str, Any]:
    """Request a new card to be sent.
    
    Args:
        phone: Customer phone number
        delivery_method: "standard" (5-7 days) or "expedited" (2-3 days)
    
    Returns:
        Card delivery information
    """
    ph = _normalize_phone(phone) or ""
    cust = _get_customer(ph)
    if not cust:
        return {"error": "customer_not_found"}
    
    fraud_case = _get_fraud_case(ph)
    if not fraud_case:
        return {"error": "no_fraud_case_found"}
    
    delivery_days = "5-7" if delivery_method == "standard" else "2-3"
    
    return {
        "status": "card_ordered",
        "delivery_method": delivery_method,
        "estimated_delivery": f"{delivery_days} business days",
        "tracking_available": delivery_method == "expedited",
        "message": f"Your new {cust.get('card_type')} card will arrive in {delivery_days} business days."
    }


def dispute_charges(phone: str, transaction_ids: List[str]) -> Dict[str, Any]:
    """File disputes for specific transactions.
    
    Args:
        phone: Customer phone number
        transaction_ids: List of transaction IDs to dispute
    
    Returns:
        Dispute confirmation and timeline
    """
    ph = _normalize_phone(phone) or ""
    fraud_case = _get_fraud_case(ph)
    if not fraud_case:
        return {"error": "no_fraud_case_found"}
    
    all_transactions = fraud_case.get("transactions", [])
    disputed = [t for t in all_transactions if t.get("transaction_id") in transaction_ids]
    
    if not disputed:
        return {"error": "no_valid_transactions_found"}
    
    total_disputed = sum(t.get("amount", 0.0) for t in disputed)
    
    return {
        "status": "disputes_filed",
        "case_id": fraud_case.get("case_id"),
        "transactions_disputed": len(disputed),
        "total_amount": total_disputed,
        "currency": disputed[0].get("currency", "USD") if disputed else "USD",
        "provisional_credit": "3-5 business days",
        "investigation_timeline": "60-90 days",
        "message": f"We have filed disputes for {len(disputed)} transaction(s). You will receive provisional credit within 3-5 business days."
    }


def mark_customer_notified(phone: str) -> Dict[str, Any]:
    """Mark that the customer has been successfully notified.
    
    This is called at the end of a successful conversation.
    """
    ph = _normalize_phone(phone) or ""
    fraud_case = _get_fraud_case(ph)
    if not fraud_case:
        return {"error": "no_fraud_case_found"}
    
    # In a real system, this would update the database
    return {
        "status": "notification_complete",
        "case_id": fraud_case.get("case_id"),
        "notified_at": datetime.utcnow().isoformat() + "Z",
        "notification_method": "phone_call"
    }

