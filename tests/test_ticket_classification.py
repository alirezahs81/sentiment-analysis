"""Tests for the rule-based ticket classifier."""

from src.ticket_classification import TicketClassifier, SAMPLE_TICKETS, TICKET_CATEGORIES


def test_sample_tickets_consistent():
    assert len(SAMPLE_TICKETS) > 0
    assert set(SAMPLE_TICKETS["label"]) <= set(TICKET_CATEGORIES)


def test_billing_ticket():
    classifier = TicketClassifier()
    result = classifier.classify("Payment charged twice, need refund immediately!")
    assert result["category"] == "Billing & Payments"
    assert result["priority"] == "high"
    assert 0.0 <= result["confidence"] <= 1.0


def test_account_ticket():
    classifier = TicketClassifier()
    result = classifier.classify("Forgot password, can't access my account")
    assert result["category"] == "Account Access"


def test_technical_ticket():
    classifier = TicketClassifier()
    result = classifier.classify("App keeps crashing on startup, bug everywhere")
    assert result["category"] == "Technical Issues"


def test_unknown_text_falls_back_to_default():
    classifier = TicketClassifier()
    result = classifier.classify("xyzzy qwerty zxcvb")
    assert result["category"] == "General Inquiries"
    assert result["confidence"] == 0.0


def test_non_string_input_is_safe():
    classifier = TicketClassifier()
    result = classifier.classify(None)
    assert result["category"] == "General Inquiries"
