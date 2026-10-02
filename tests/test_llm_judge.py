import pytest

import app.llm_judge as judge


def test_judge_response_has_required_metrics():
    """The judge must return all five judge metrics."""
    result = judge.judge_response(
        question="What is machine learning?",
        context=(
            "Machine learning is a branch of artificial intelligence "
            "that enables systems to learn patterns from data."
        ),
        expected_answer=(
            "Machine learning enables systems to learn patterns from data."
        ),
        model_answer=(
            "Machine learning enables systems to learn patterns from data."
        ),
    )

    required_metrics = {
        "correctness",
        "relevance",
        "faithfulness",
        "coherence",
        "hallucination",
    }

    assert required_metrics.issubset(result.keys())


def test_judge_scores_are_normalized():
    """All judge metric scores must remain between 0 and 1."""
    result = judge.judge_response(
        question="What is Python?",
        context="Python is a programming language.",
        expected_answer="Python is a programming language.",
        model_answer="Python is a programming language.",
    )

    metric_names = {
        "correctness",
        "relevance",
        "faithfulness",
        "coherence",
        "hallucination",
    }

    for metric_name in metric_names:
        assert metric_name in result
        assert 0.0 <= float(result[metric_name]) <= 1.0


def test_judge_metadata_is_present():
    """Judge results should expose model, type, rubric and fallback metadata."""
    result = judge.judge_response(
        question="What is AI?",
        context="AI enables machines to perform tasks associated with intelligence.",
        expected_answer="AI enables machines to perform intelligent tasks.",
        model_answer="AI enables machines to perform intelligent tasks.",
    )

    assert "judge_type" in result
    assert "judge_model" in result
    assert "rubric_version" in result
    assert "fallback" in result

    assert isinstance(result["judge_type"], str)
    assert isinstance(result["judge_model"], str)
    assert isinstance(result["rubric_version"], str)
    assert isinstance(result["fallback"], bool)


def test_configured_rubric_version():
    """The project should use the explicitly versioned judge rubric."""
    assert judge.JUDGE_RUBRIC_VERSION == "v3"


def test_configured_judge_model():
    """The project should have a configured judge model."""
    assert isinstance(judge.JUDGE_MODEL, str)
    assert judge.JUDGE_MODEL.strip() != ""