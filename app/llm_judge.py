"""
LLM-as-a-Judge evaluation module.

Uses Gemini as the real LLM judge.
Falls back to deterministic local scoring only when Gemini is unavailable.
"""

import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from google import genai


load_dotenv()


# ============================================================
# Configuration
# ============================================================

JUDGE_RUBRIC_VERSION = "v3"

JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    "gemini-3.6-flash"
).strip()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


# ============================================================
# Utility functions
# ============================================================

def clamp_score(score: float) -> float:
    """Keep a score between 0 and 1."""
    try:
        return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(text: str) -> str:
    """Normalize text for deterministic fallback scoring."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _words(text: str) -> set[str]:
    """Return normalized word tokens."""
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", _normalize_text(text)))


# ============================================================
# Local deterministic fallback
# ============================================================

def _local_judge(
    question: str,
    context: str,
    expected_answer: str,
    model_answer: str,
    reason: str,
) -> Dict[str, Any]:
    """
    Deterministic fallback used only when the real Gemini judge
    cannot be reached.
    """

    expected_words = _words(expected_answer)
    answer_words = _words(model_answer)
    question_words = _words(question)
    context_words = _words(context)

    # Remove very common words for relevance calculation.
    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "does",
        "do",
        "how",
        "why",
        "can",
        "be",
    }

    meaningful_question_words = {
        word
        for word in question_words
        if word not in stop_words
    }

    # Correctness
    if expected_words:
        correctness = len(
            expected_words.intersection(answer_words)
        ) / len(expected_words)
    else:
        correctness = 0.0

    # Relevance
    if meaningful_question_words:
        relevance = len(
            meaningful_question_words.intersection(answer_words)
        ) / len(meaningful_question_words)
    else:
        relevance = 0.0

    # Faithfulness
    if answer_words:
        faithfulness = len(
            answer_words.intersection(context_words)
        ) / len(answer_words)
    else:
        faithfulness = 0.0

    # Coherence
    answer_length = len(answer_words)

    if answer_length >= 3:
        coherence = 1.0
    elif answer_length > 0:
        coherence = 0.5
    else:
        coherence = 0.0

    hallucination = 1.0 - faithfulness

    return {
        "judge_type": "local_mock",
        "judge_model": "local_deterministic",
        "rubric_version": JUDGE_RUBRIC_VERSION,

        "correctness": clamp_score(correctness),
        "relevance": clamp_score(relevance),
        "faithfulness": clamp_score(faithfulness),
        "coherence": clamp_score(coherence),
        "hallucination": clamp_score(hallucination),

        "explanations": {
            "correctness": (
                "Estimated using deterministic word overlap with "
                "the expected answer. Real Gemini LLM judge was "
                "unavailable."
            ),
            "relevance": (
                "Estimated using deterministic meaningful-term "
                "overlap with the question. Real Gemini LLM judge "
                "was unavailable."
            ),
            "faithfulness": (
                "Estimated using deterministic word overlap with "
                "the supplied context. Real Gemini LLM judge was "
                "unavailable."
            ),
            "coherence": (
                "Estimated using a deterministic response-length "
                "check. Real Gemini LLM judge was unavailable."
            ),
            "hallucination": (
                "Estimated as the inverse of the local faithfulness "
                "score. Real Gemini LLM judge was unavailable."
            ),
        },

        "fallback": True,
        "fallback_reason": str(reason),
    }


# ============================================================
# Gemini LLM Judge
# ============================================================

def _gemini_llm_judge(
    question: str,
    context: str,
    expected_answer: str,
    model_answer: str,
) -> Dict[str, Any]:

    api_key = GOOGLE_API_KEY or GEMINI_API_KEY

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is not configured."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert LLM evaluation judge.

Evaluate the model's answer using the question, context,
and expected answer provided below.

QUESTION:
{question}

CONTEXT:
{context}

EXPECTED ANSWER:
{expected_answer}

MODEL ANSWER:
{model_answer}

Evaluate these five metrics:

1. correctness
2. relevance
3. faithfulness
4. coherence
5. hallucination

Scoring rules:

- correctness:
  How accurately does the model answer match the expected answer?

- relevance:
  How directly does the answer address the question?

- faithfulness:
  How well is the answer supported by the supplied context?

- coherence:
  How clear, logical, and understandable is the answer?

- hallucination:
  How much unsupported or fabricated information appears
  in the answer?

For correctness, relevance, faithfulness, and coherence:
0 means completely poor.
1 means excellent.

For hallucination:
0 means no hallucination.
1 means severe hallucination.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "correctness": 0.0,
    "relevance": 0.0,
    "faithfulness": 0.0,
    "coherence": 0.0,
    "hallucination": 0.0,
    "explanations": {{
        "correctness": "brief explanation",
        "relevance": "brief explanation",
        "faithfulness": "brief explanation",
        "coherence": "brief explanation",
        "hallucination": "brief explanation"
    }}
}}
"""

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    raw_text = getattr(response, "text", None)

    if not raw_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    raw_text = raw_text.strip()

    # Remove accidental markdown fences if Gemini adds them.
    raw_text = re.sub(
        r"^```json\s*",
        "",
        raw_text,
        flags=re.IGNORECASE,
    )

    raw_text = re.sub(
        r"\s*```$",
        "",
        raw_text,
    )

    data = json.loads(raw_text)

    required_metrics = [
        "correctness",
        "relevance",
        "faithfulness",
        "coherence",
        "hallucination",
    ]

    for metric in required_metrics:
        if metric not in data:
            raise ValueError(
                f"Gemini judge response missing metric: {metric}"
            )

        data[metric] = clamp_score(data[metric])

    explanations = data.get("explanations", {})

    for metric in required_metrics:
        explanations.setdefault(
            metric,
            "Gemini provided no explanation."
        )

    data["explanations"] = explanations

    return {
        "judge_type": "llm",
        "judge_model": JUDGE_MODEL,
        "rubric_version": JUDGE_RUBRIC_VERSION,

        "correctness": data["correctness"],
        "relevance": data["relevance"],
        "faithfulness": data["faithfulness"],
        "coherence": data["coherence"],
        "hallucination": data["hallucination"],

        "explanations": data["explanations"],

        "fallback": False,
        "fallback_reason": None,
    }


# ============================================================
# Public judge function
# ============================================================

def judge_response(
    question: str,
    context: str,
    expected_answer: str,
    model_answer: str,
) -> Dict[str, Any]:
    """
    Run the real Gemini LLM judge.

    If Gemini is unavailable, return deterministic fallback
    results instead of crashing the evaluation pipeline.
    """

    try:
        return _gemini_llm_judge(
            question=question,
            context=context,
            expected_answer=expected_answer,
            model_answer=model_answer,
        )

    except Exception as exc:

        return _local_judge(
            question=question,
            context=context,
            expected_answer=expected_answer,
            model_answer=model_answer,
            reason=f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LLM-as-a-Judge Self-Test")
    print("=" * 70)

    question = "What is machine learning?"

    context = (
        "Machine learning is a branch of artificial intelligence "
        "that enables systems to learn patterns from data and make "
        "predictions or decisions."
    )

    expected_answer = (
        "Machine learning enables systems to learn patterns from data."
    )

    model_answer = (
        "Machine learning enables computers to learn patterns "
        "from data and make predictions."
    )

    result = judge_response(
        question=question,
        context=context,
        expected_answer=expected_answer,
        model_answer=model_answer,
    )

    print(json.dumps(result, indent=2))

    print()
    print("-" * 70)
    print("Judge type   :", result["judge_type"])
    print("Judge model  :", result["judge_model"])
    print("Rubric       :", result["rubric_version"])
    print("Fallback     :", result["fallback"])

    if result["fallback"]:
        print("Reason       :", result["fallback_reason"])
    else:
        print("Status       : REAL GEMINI LLM JUDGE")

    print("-" * 70)