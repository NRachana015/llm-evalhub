import pytest
from pydantic import ValidationError

from app.schemas import (
    DatasetCreate,
    PromptIn,
    ResponseOut,
    RunCreate,
    ScoreOut,
)


def test_prompt_requires_question():
    """PromptIn must require a question."""
    with pytest.raises(ValidationError):
        PromptIn()


def test_prompt_accepts_valid_data():
    """PromptIn should accept a valid prompt definition."""
    prompt = PromptIn(
        question="What is machine learning?",
        context="Machine learning learns patterns from data.",
        expected_output="Machine learning learns patterns from data.",
        category="AI Fundamentals",
        difficulty="Easy",
    )

    assert prompt.question == "What is machine learning?"
    assert prompt.category == "AI Fundamentals"
    assert prompt.difficulty == "Easy"


def test_dataset_requires_name_and_prompts():
    """DatasetCreate must require both name and prompts."""
    with pytest.raises(ValidationError):
        DatasetCreate()

    dataset = DatasetCreate(
        name="Test Dataset",
        prompts=[
            PromptIn(question="What is Python?")
        ],
    )

    assert dataset.name == "Test Dataset"
    assert len(dataset.prompts) == 1


def test_run_defaults_prompt_version_to_v1():
    """RunCreate should default to prompt version v1."""
    run = RunCreate(
        name="Test Run",
        dataset_id=1,
        models=["codestral-2508"],
    )

    assert run.prompt_version == "v1"


def test_run_accepts_experiment_configuration():
    """RunCreate should preserve experiment configuration."""
    run = RunCreate(
        name="Benchmark Test",
        dataset_id=13,
        models=["gemini-3.6-flash", "codestral-2508"],
        prompt_version="v2",
        temperature=0.0,
        max_tokens=256,
    )

    assert run.dataset_id == 13
    assert run.models == ["gemini-3.6-flash", "codestral-2508"]
    assert run.prompt_version == "v2"
    assert run.temperature == 0.0
    assert run.max_tokens == 256


def test_score_output():
    """ScoreOut should correctly represent an evaluation score."""
    score = ScoreOut(
        metric_name="correctness",
        score_value=0.95,
        explanation="The answer is correct.",
    )

    assert score.metric_name == "correctness"
    assert score.score_value == pytest.approx(0.95)
    assert score.explanation == "The answer is correct."


def test_response_output():
    """ResponseOut should correctly represent an evaluated response."""
    response = ResponseOut(
        id=1,
        model_name="codestral-2508",
        output_text="Machine learning learns patterns from data.",
        latency_ms=850.5,
        cost_usd=0.00004,
        scores=[
            ScoreOut(
                metric_name="correctness",
                score_value=1.0,
                explanation="Correct answer.",
            )
        ],
    )

    assert response.id == 1
    assert response.model_name == "codestral-2508"
    assert response.latency_ms == pytest.approx(850.5)
    assert response.cost_usd == pytest.approx(0.00004)
    assert len(response.scores) == 1