import pytest

from app.llm_clients import (
    LLMResult,
    call_model,
    call_mock,
)


def test_llm_result_structure():
    """LLMResult should store the core evaluation metadata."""
    result = LLMResult(
        output_text="Test response",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.00001,
    )

    assert result.output_text == "Test response"
    assert result.latency_ms == pytest.approx(100.0)
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.cost_usd == pytest.approx(0.00001)


def test_mock_model_returns_result():
    """The mock model should return a valid LLMResult."""
    result = call_mock(
        prompt="What is machine learning?"
    )

    assert isinstance(result, LLMResult)
    assert isinstance(result.output_text, str)
    assert result.output_text.strip() != ""
    assert result.latency_ms >= 0
    assert result.prompt_tokens >= 0
    assert result.completion_tokens >= 0
    assert result.cost_usd >= 0


def test_mock_model_custom_name():
    """The mock client should accept a custom model name."""
    result = call_mock(
        prompt="Explain Python.",
        model="test-model",
    )

    assert isinstance(result, LLMResult)
    assert result.output_text.strip() != ""


def test_call_model_routes_to_mock():
    """call_model should correctly route mock-model requests."""
    result = call_model(
        "mock-model",
        "What is artificial intelligence?",
    )

    assert isinstance(result, LLMResult)
    assert result.output_text.strip() != ""


def test_call_model_unknown_model_raises_error():
    """Unsupported model names should fail explicitly."""
    with pytest.raises((ValueError, KeyError)):
        call_model(
            "unsupported-model",
            "Test prompt",
        )