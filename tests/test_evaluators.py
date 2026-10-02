import pytest

from app.evaluators import (
    STANDARD_METRICS,
    evaluate_response,
)


def scores_as_dict(results):
    """Convert evaluator output into {metric_name: score_value}."""
    return {
        item["metric_name"]: item["score_value"]
        for item in results
    }


def test_standard_metric_count():
    """The evaluation engine must expose exactly 7 standard metrics."""
    assert len(STANDARD_METRICS) == 7


def test_standard_metric_names():
    """Verify the expected standard evaluation metrics are present."""
    expected_metrics = {
        "exact_match",
        "token_f1",
        "tfidf_cosine_similarity",
        "correctness",
        "relevance",
        "faithfulness",
        "coherence",
    }

    assert set(STANDARD_METRICS) == expected_metrics


def test_evaluator_returns_all_standard_metrics():
    """A normal evaluation must produce all 7 standard metrics."""
    results = evaluate_response(
        question="What is machine learning?",
        answer="Machine learning enables systems to learn patterns from data.",
        expected="Machine learning enables systems to learn patterns from data.",
        context=(
            "Machine learning is a branch of artificial intelligence "
            "that enables systems to learn patterns from data."
        ),
    )

    scores = scores_as_dict(results)

    assert len(results) == 7
    assert set(scores.keys()) == set(STANDARD_METRICS)


def test_perfect_response_scores_high():
    """An answer matching the expected output should score perfectly."""
    results = evaluate_response(
        question="What is machine learning?",
        answer="Machine learning enables systems to learn patterns from data.",
        expected="Machine learning enables systems to learn patterns from data.",
        context=(
            "Machine learning is a branch of artificial intelligence "
            "that enables systems to learn patterns from data."
        ),
    )

    scores = scores_as_dict(results)

    assert scores["exact_match"] == 1.0
    assert scores["token_f1"] == pytest.approx(1.0)
    assert scores["tfidf_cosine_similarity"] == pytest.approx(1.0)
    assert scores["correctness"] == pytest.approx(1.0)
    assert scores["faithfulness"] == pytest.approx(1.0)
    assert scores["coherence"] == pytest.approx(1.0)


def test_relevant_response_is_detected():
    """A response related to the question should receive a relevance score."""
    results = evaluate_response(
        question="What is machine learning?",
        answer=(
            "Machine learning allows systems to learn patterns from data "
            "and make predictions."
        ),
        expected="Machine learning enables systems to learn patterns from data.",
        context=(
            "Machine learning is a branch of artificial intelligence "
            "that enables systems to learn patterns from data."
        ),
    )

    scores = scores_as_dict(results)

    assert scores["relevance"] > 0.0


def test_incorrect_response_does_not_score_perfectly():
    """A clearly incorrect answer must not receive a perfect correctness score."""
    results = evaluate_response(
        question="What is machine learning?",
        answer="Machine learning is a type of database.",
        expected="Machine learning enables systems to learn patterns from data.",
        context=(
            "Machine learning is a branch of artificial intelligence "
            "that enables systems to learn patterns from data."
        ),
    )

    scores = scores_as_dict(results)

    assert scores["correctness"] < 1.0
    assert scores["faithfulness"] < 1.0


def test_metric_scores_are_normalized():
    """Every standard metric must remain within the 0–1 range."""
    results = evaluate_response(
        question="What is Python?",
        answer="Python is a programming language.",
        expected="Python is a programming language.",
        context="Python is a high-level programming language.",
    )

    scores = scores_as_dict(results)

    for metric_name in STANDARD_METRICS:
        assert metric_name in scores
        assert 0.0 <= scores[metric_name] <= 1.0