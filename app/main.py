from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from .database import SessionLocal
from .evaluators import (
    STANDARD_METRICS,
    evaluate_response,
)
from .llm_clients import call_model
from .llm_judge import judge_response
from .models import (
    Dataset,
    Prompt,
    Run,
    Response,
    Score,
    JudgeScore,
)
from .schemas import (
    DatasetCreate,
    RunCreate,
    ResponseOut,
)


# ============================================================
# APPLICATION
# ============================================================

APP_VERSION = "1.7.0"

app = FastAPI(
    title="LLM Evaluation Dashboard API",
    version=APP_VERSION,
)


# ============================================================
# CONSTANTS
# ============================================================

JUDGE_METRICS = [
    "correctness",
    "relevance",
    "faithfulness",
    "coherence",
    "hallucination",
]


# ============================================================
# DATABASE SESSION
# ============================================================

def get_db():
    """
    Database session generator.

    Kept available for future dependency injection
    and API expansion.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "LLM Evaluation Dashboard API",
        "version": APP_VERSION,
    }


# ============================================================
# PROMPT MANAGER
# ============================================================

def get_all_prompts():
    from .prompt_manager import list_prompts

    return list_prompts()


@app.get("/prompts")
def list_available_prompts():

    try:
        return get_all_prompts()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load prompts: {str(exc)}",
        )


@app.get("/prompts/{version}")
def get_prompt_version(version: str):

    from .prompt_manager import get_prompt

    try:
        return get_prompt(version)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt version '{version}' not found.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load prompt: {str(exc)}",
        )


# ============================================================
# DATASETS
# ============================================================

@app.get("/datasets")
def get_datasets():

    db: Session = SessionLocal()

    try:

        datasets = (
            db.query(Dataset)
            .order_by(Dataset.id)
            .all()
        )

        result = []

        for dataset in datasets:

            result.append(
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "created_at": dataset.created_at,
                    "prompt_count": len(dataset.prompts),
                }
            )

        return result

    finally:
        db.close()


@app.post("/datasets")
def create_dataset(payload: DatasetCreate):

    db: Session = SessionLocal()

    try:

        dataset = Dataset(
            name=payload.name,
        )

        db.add(dataset)
        db.flush()

        for item in payload.prompts:

            prompt = Prompt(
                dataset_id=dataset.id,
                question=item.question,
                context=item.context,
                expected_output=item.expected_output,
                category=item.category,
                difficulty=item.difficulty,
            )

            db.add(prompt)

        db.commit()
        db.refresh(dataset)

        return {
            "message": "Dataset created successfully.",
            "dataset_id": dataset.id,
            "name": dataset.name,
            "prompt_count": len(dataset.prompts),
        }

    except Exception:

        db.rollback()
        raise

    finally:
        db.close()


@app.get("/datasets/{dataset_id}/prompts")
def get_dataset_prompts(dataset_id: int):

    db: Session = SessionLocal()

    try:

        dataset = (
            db.query(Dataset)
            .filter(
                Dataset.id == dataset_id
            )
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found.",
            )

        prompts = (
            db.query(Prompt)
            .filter(
                Prompt.dataset_id == dataset_id
            )
            .order_by(Prompt.id)
            .all()
        )

        return [
            {
                "id": prompt.id,
                "dataset_id": prompt.dataset_id,
                "question": prompt.question,
                "context": prompt.context,
                "expected_output": prompt.expected_output,
                "category": prompt.category,
                "difficulty": prompt.difficulty,
            }
            for prompt in prompts
        ]

    finally:
        db.close()


# ============================================================
# HELPER — JUDGE STATUS
# ============================================================

def classify_judge_status(judge_types):

    normalized = {
        str(judge_type).strip().lower()
        for judge_type in judge_types
        if judge_type
    }

    if not normalized:
        return "unknown"

    if normalized == {"llm"}:
        return "llm"

    if normalized == {"local_mock"}:
        return "local_mock"

    if (
        "llm" in normalized
        and "local_mock" in normalized
    ):
        return "mixed"

    return "mixed"


# ============================================================
# CREATE EVALUATION RUN
# ============================================================

@app.post("/runs")
def create_run(payload: RunCreate):
    """
    Execute a complete evaluation experiment.

    Pipeline:

        Dataset
            ↓
        Prompt Version
            ↓
        LLM Model
            ↓
        Model Response
            ↓
        Standard Evaluation
            ↓
        LLM-as-a-Judge
            ↓
        Database

    Important resilience behavior:

    If one provider fails because of:
    - quota exhaustion
    - rate limit
    - network error
    - provider API error

    the other selected models continue running.

    The response contains failed_models so the caller
    knows exactly what happened.
    """

    db: Session = SessionLocal()

    try:

        # ====================================================
        # VALIDATE DATASET
        # ====================================================

        dataset = (
            db.query(Dataset)
            .filter(
                Dataset.id == payload.dataset_id
            )
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found.",
            )

        # ====================================================
        # LOAD PROMPT VERSION
        # ====================================================

        from .prompt_manager import get_prompt

        try:

            prompt_config = get_prompt(
                payload.prompt_version
            )

        except FileNotFoundError:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Prompt version "
                    f"'{payload.prompt_version}' not found."
                ),
            )

        prompt_template = prompt_config["template"]

        # ====================================================
        # VALIDATE MODELS
        # ====================================================

        if not payload.models:

            raise HTTPException(
                status_code=400,
                detail="At least one model must be provided.",
            )

        # Remove duplicate model names while preserving order.

        models = list(
            dict.fromkeys(
                payload.models
            )
        )

        # ====================================================
        # CREATE RUN
        # ====================================================

        run = Run(
            name=payload.name,
            dataset_id=payload.dataset_id,
            models_used=",".join(models),
            prompt_version=payload.prompt_version,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            created_at=datetime.utcnow(),
        )

        db.add(run)
        db.flush()

        # ====================================================
        # COUNTERS
        # ====================================================

        response_count = 0
        standard_score_count = 0
        judge_score_count = 0

        successful_models = set()
        failed_models = []

        judge_types_used = []
        rubric_versions_used = []

        # ====================================================
        # DATASET PROMPTS
        # ====================================================

        prompts = (
            db.query(Prompt)
            .filter(
                Prompt.dataset_id == dataset.id
            )
            .order_by(Prompt.id)
            .all()
        )

        if not prompts:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The selected dataset "
                    "contains no prompts."
                ),
            )

        # ====================================================
        # PROCESS EACH PROMPT
        # ====================================================

        for prompt in prompts:

            context_text = (
                prompt.context
                if prompt.context
                else "No context was provided."
            )

            # =================================================
            # BUILD FINAL MODEL PROMPT
            # =================================================

            final_prompt = (
                f"{prompt_template}\n\n"
                f"Context:\n"
                f"{context_text}\n\n"
                f"Question:\n"
                f"{prompt.question}"
            )

            # =================================================
            # RUN EACH SELECTED MODEL
            # =================================================

            for model_name in models:

                # =============================================
                # CALL MODEL
                # =============================================

                try:

                    try:

                        result = call_model(
                            model_name,
                            final_prompt,
                            temperature=payload.temperature,
                            max_tokens=payload.max_tokens,
                        )

                    except TypeError:

                        # Backward compatibility for the
                        # existing two-argument call_model API.

                        result = call_model(
                            model_name,
                            final_prompt,
                        )

                except Exception as exc:

                    # -----------------------------------------
                    # IMPORTANT:
                    # Do NOT abort the complete experiment.
                    # -----------------------------------------

                    failure_message = str(exc)

                    failed_models.append(
                        {
                            "model_name": model_name,
                            "question": prompt.question,
                            "error_type": type(
                                exc
                            ).__name__,
                            "error": failure_message,
                        }
                    )

                    continue

                # =============================================
                # VALIDATE MODEL RESULT
                # =============================================

                if result is None:

                    failed_models.append(
                        {
                            "model_name": model_name,
                            "question": prompt.question,
                            "error_type": "EmptyResult",
                            "error": (
                                "Model returned no result."
                            ),
                        }
                    )

                    continue

                output_text = (
                    result.output_text
                    if result.output_text is not None
                    else ""
                )

                # =============================================
                # SAVE MODEL RESPONSE
                # =============================================

                response = Response(
                    run_id=run.id,
                    model_name=model_name,
                    output_text=output_text,
                    latency_ms=float(
                        result.latency_ms or 0.0
                    ),
                    cost_usd=float(
                        result.cost_usd or 0.0
                    ),
                    prompt_tokens=(
                        result.prompt_tokens
                    ),
                    completion_tokens=(
                        result.completion_tokens
                    ),
                )

                db.add(response)
                db.flush()

                response_count += 1
                successful_models.add(model_name)

                # =============================================
                # STANDARD EVALUATION
                # =============================================

                try:

                    evaluation_results = evaluate_response(
                        question=prompt.question,
                        answer=output_text,
                        context=prompt.context or "",
                        expected=(
                            prompt.expected_output
                            or ""
                        ),
                        metrics=STANDARD_METRICS,
                        model_name=model_name,
                    )

                except TypeError:

                    # Compatibility with an evaluator that
                    # exposes the older function signature.

                    evaluation_results = evaluate_response(
                        question=prompt.question,
                        answer=output_text,
                        context=prompt.context or "",
                        expected=(
                            prompt.expected_output
                            or ""
                        ),
                        model_name=model_name,
                    )

                if not isinstance(
                    evaluation_results,
                    list,
                ):

                    raise RuntimeError(
                        (
                            "Evaluation engine returned "
                            "an invalid result structure."
                        )
                    )

                # =============================================
                # SAVE STANDARD SCORES
                # =============================================

                response_standard_count = 0

                for metric_result in evaluation_results:

                    if not isinstance(
                        metric_result,
                        dict,
                    ):
                        continue

                    metric_name = metric_result.get(
                        "metric_name"
                    )

                    score_value = metric_result.get(
                        "score_value"
                    )

                    if score_value is None:

                        # Some evaluator versions use "score".

                        score_value = metric_result.get(
                            "score"
                        )

                    explanation = metric_result.get(
                        "explanation"
                    )

                    if not metric_name:
                        continue

                    if score_value is None:
                        continue

                    try:

                        score_value = float(
                            score_value
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue

                    score_value = max(
                        0.0,
                        min(
                            1.0,
                            score_value,
                        ),
                    )

                    score = Score(
                        response_id=response.id,
                        metric_name=metric_name,
                        score_value=score_value,
                        explanation=explanation,
                    )

                    db.add(score)

                    standard_score_count += 1
                    response_standard_count += 1

                # =============================================
                # STANDARD METRIC VALIDATION
                # =============================================

                if response_standard_count != len(
                    STANDARD_METRICS
                ):

                    raise RuntimeError(
                        (
                            f"Standard evaluation produced "
                            f"{response_standard_count} saved "
                            f"metrics for response "
                            f"{response.id}, but "
                            f"{len(STANDARD_METRICS)} "
                            f"were expected."
                        )
                    )

                # =============================================
                # LLM-AS-A-JUDGE
                # =============================================

                judge_result = judge_response(
                    question=prompt.question,
                    context=prompt.context or "",
                    expected_answer=(
                        prompt.expected_output or ""
                    ),
                    model_answer=output_text,
                )

                if not isinstance(
                    judge_result,
                    dict,
                ):

                    raise RuntimeError(
                        "LLM judge returned an invalid result."
                    )

                # =============================================
                # JUDGE METADATA
                # =============================================

                judge_type = judge_result.get(
                    "judge_type",
                    "local_mock",
                )

                rubric_version = judge_result.get(
                    "rubric_version",
                    "v1",
                )

                judge_types_used.append(
                    judge_type
                )

                rubric_versions_used.append(
                    rubric_version
                )

                explanations = judge_result.get(
                    "explanations",
                    {},
                )

                if not isinstance(
                    explanations,
                    dict,
                ):
                    explanations = {}

                # =============================================
                # SAVE JUDGE SCORES
                # =============================================

                response_judge_count = 0

                for metric_name in JUDGE_METRICS:

                    score_value = judge_result.get(
                        metric_name
                    )

                    if score_value is None:
                        continue

                    try:

                        score_value = float(
                            score_value
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue

                    score_value = max(
                        0.0,
                        min(
                            1.0,
                            score_value,
                        ),
                    )

                    explanation = explanations.get(
                        metric_name
                    )

                    judge_score = JudgeScore(
                        response_id=response.id,
                        judge_type=judge_type,
                        rubric_version=rubric_version,
                        metric_name=metric_name,
                        score_value=score_value,
                        explanation=explanation,
                    )

                    db.add(judge_score)

                    judge_score_count += 1
                    response_judge_count += 1

                # =============================================
                # JUDGE VALIDATION
                # =============================================

                if response_judge_count != len(
                    JUDGE_METRICS
                ):

                    raise RuntimeError(
                        (
                            f"Judge evaluation produced "
                            f"{response_judge_count} saved "
                            f"metrics for response "
                            f"{response.id}, but "
                            f"{len(JUDGE_METRICS)} "
                            f"were expected."
                        )
                    )

        # ====================================================
        # IF NOTHING SUCCEEDED
        # ====================================================

        if response_count == 0:

            db.commit()

            return {
                "message": (
                    "Evaluation run completed with "
                    "no successful model responses."
                ),
                "status": "failed",
                "run_id": run.id,
                "name": run.name,
                "dataset_id": run.dataset_id,
                "models": models,
                "prompt_version": payload.prompt_version,
                "response_count": 0,
                "standard_score_count": 0,
                "judge_score_count": 0,
                "successful_models": [],
                "failed_models": failed_models,
                "judge_type": "unknown",
                "judge_rubric_version": None,
            }

        # ====================================================
        # FINAL COUNT VALIDATION
        # ====================================================

        expected_standard_score_count = (
            response_count
            * len(STANDARD_METRICS)
        )

        expected_judge_score_count = (
            response_count
            * len(JUDGE_METRICS)
        )

        if standard_score_count != (
            expected_standard_score_count
        ):

            raise RuntimeError(
                (
                    "Standard metric count mismatch. "
                    f"Expected "
                    f"{expected_standard_score_count}, "
                    f"but generated "
                    f"{standard_score_count}."
                )
            )

        if judge_score_count != (
            expected_judge_score_count
        ):

            raise RuntimeError(
                (
                    "Judge metric count mismatch. "
                    f"Expected "
                    f"{expected_judge_score_count}, "
                    f"but generated "
                    f"{judge_score_count}."
                )
            )

        # ====================================================
        # DETERMINE JUDGE STATUS
        # ====================================================

        judge_status = classify_judge_status(
            judge_types_used
        )

        unique_rubrics = sorted(
            {
                str(item)
                for item in rubric_versions_used
                if item
            }
        )

        if len(unique_rubrics) == 1:
            judge_rubric_version = unique_rubrics[0]
        elif unique_rubrics:
            judge_rubric_version = unique_rubrics
        else:
            judge_rubric_version = None

        llm_judge_count = sum(
            1
            for item in judge_types_used
            if str(item).lower() == "llm"
        )

        local_fallback_count = sum(
            1
            for item in judge_types_used
            if str(item).lower()
            == "local_mock"
        )

        # ====================================================
        # COMMIT EXPERIMENT
        # ====================================================

        db.commit()
        db.refresh(run)

        # ====================================================
        # FINAL STATUS
        # ====================================================

        if failed_models:

            run_status = "completed_with_errors"

        else:

            run_status = "completed"

        # ====================================================
        # RETURN EXPERIMENT SUMMARY
        # ====================================================

        return {
            "message": (
                "Evaluation run completed."
                if not failed_models
                else (
                    "Evaluation run completed with "
                    "one or more model failures."
                )
            ),
            "status": run_status,
            "run_id": run.id,
            "name": run.name,
            "dataset_id": run.dataset_id,
            "models": models,
            "prompt_version": payload.prompt_version,
            "prompt_name": prompt_config.get(
                "name"
            ),
            "temperature": payload.temperature,
            "max_tokens": payload.max_tokens,
            "response_count": response_count,
            "standard_metrics_per_response": len(
                STANDARD_METRICS
            ),
            "standard_score_count": (
                standard_score_count
            ),
            "judge_metrics_per_response": len(
                JUDGE_METRICS
            ),
            "judge_score_count": (
                judge_score_count
            ),
            "judge_type": judge_status,
            "judge_rubric_version": (
                judge_rubric_version
            ),
            "llm_judge_count": (
                llm_judge_count
            ),
            "local_fallback_count": (
                local_fallback_count
            ),
            "judge_total_count": (
                len(judge_types_used)
            ),
            "successful_models": sorted(
                successful_models
            ),
            "failed_models": failed_models,
        }

    except HTTPException:

        db.rollback()
        raise

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Evaluation run failed: {str(exc)}"
            ),
        )

    finally:

        db.close()


# ============================================================
# RUN DETAILS
# ============================================================

@app.get("/runs/{run_id}")
def get_run(run_id: int):

    db: Session = SessionLocal()

    try:

        run = (
            db.query(Run)
            .filter(
                Run.id == run_id
            )
            .first()
        )

        if run is None:

            raise HTTPException(
                status_code=404,
                detail="Run not found.",
            )

        return {
            "run_id": run.id,
            "name": run.name,
            "dataset_id": run.dataset_id,
            "models_used": run.models_used,
            "prompt_version": run.prompt_version,
            "temperature": run.temperature,
            "max_tokens": run.max_tokens,
            "created_at": run.created_at,
        }

    finally:

        db.close()


# ============================================================
# STANDARD RESULTS
# ============================================================

@app.get(
    "/runs/{run_id}/results",
    response_model=list[ResponseOut],
)
def get_run_results(run_id: int):

    db: Session = SessionLocal()

    try:

        run = (
            db.query(Run)
            .filter(
                Run.id == run_id
            )
            .first()
        )

        if run is None:

            raise HTTPException(
                status_code=404,
                detail="Run not found.",
            )

        responses = (
            db.query(Response)
            .filter(
                Response.run_id == run_id
            )
            .order_by(Response.id)
            .all()
        )

        results = []

        for response in responses:

            scores = (
                db.query(Score)
                .filter(
                    Score.response_id
                    == response.id
                )
                .order_by(Score.id)
                .all()
            )

            results.append(
                {
                    "id": response.id,
                    "model_name": response.model_name,
                    "output_text": response.output_text,
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.cost_usd,
                    "scores": [
                        {
                            "metric_name": (
                                score.metric_name
                            ),
                            "score_value": (
                                score.score_value
                            ),
                            "explanation": (
                                score.explanation
                            ),
                        }
                        for score in scores
                    ],
                }
            )

        return results

    finally:

        db.close()


# ============================================================
# LLM-AS-A-JUDGE RESULTS
# ============================================================

@app.get(
    "/runs/{run_id}/judge-results"
)
def get_judge_results(run_id: int):

    db: Session = SessionLocal()

    try:

        # ====================================================
        # VERIFY RUN
        # ====================================================

        run = (
            db.query(Run)
            .filter(
                Run.id == run_id
            )
            .first()
        )

        if run is None:

            raise HTTPException(
                status_code=404,
                detail="Run not found.",
            )

        # ====================================================
        # GET RESPONSES
        # ====================================================

        responses = (
            db.query(Response)
            .filter(
                Response.run_id == run_id
            )
            .order_by(Response.id)
            .all()
        )

        results = []

        # ====================================================
        # GET JUDGE SCORES
        # ====================================================

        for response in responses:

            judge_scores = (
                db.query(JudgeScore)
                .filter(
                    JudgeScore.response_id
                    == response.id
                )
                .order_by(
                    JudgeScore.id
                )
                .all()
            )

            results.append(
                {
                    "response_id": response.id,
                    "model_name": response.model_name,
                    "output_text": response.output_text,
                    "judge_scores": [
                        {
                            "judge_type": (
                                score.judge_type
                            ),
                            "rubric_version": (
                                score.rubric_version
                            ),
                            "metric_name": (
                                score.metric_name
                            ),
                            "score_value": (
                                score.score_value
                            ),
                            "explanation": (
                                score.explanation
                            ),
                            "created_at": (
                                score.created_at
                            ),
                        }
                        for score in judge_scores
                    ],
                }
            )

        # ====================================================
        # DETERMINE JUDGE METADATA
        # ====================================================

        all_judge_scores = (
            db.query(JudgeScore)
            .join(
                Response,
                JudgeScore.response_id
                == Response.id,
            )
            .filter(
                Response.run_id == run_id
            )
            .order_by(
                JudgeScore.id
            )
            .all()
        )

        if all_judge_scores:

            judge_types = sorted(
                {
                    score.judge_type
                    for score in all_judge_scores
                    if score.judge_type
                }
            )

            rubric_versions = sorted(
                {
                    score.rubric_version
                    for score in all_judge_scores
                    if score.rubric_version
                }
            )

            judge_type = classify_judge_status(
                judge_types
            )

            if len(rubric_versions) == 1:

                rubric_version = (
                    rubric_versions[0]
                )

            else:

                rubric_version = (
                    rubric_versions
                )

        else:

            judge_type = "unknown"
            rubric_version = None

        # ====================================================
        # RESPONSE-LEVEL COUNTS
        # ====================================================

        response_judge_types = []

        for response in responses:

            response_judge_scores = (
                db.query(JudgeScore)
                .filter(
                    JudgeScore.response_id
                    == response.id
                )
                .all()
            )

            types_for_response = {
                score.judge_type
                for score in response_judge_scores
                if score.judge_type
            }

            if "llm" in types_for_response:

                response_judge_types.append(
                    "llm"
                )

            elif "local_mock" in types_for_response:

                response_judge_types.append(
                    "local_mock"
                )

        llm_judge_count = sum(
            1
            for item in response_judge_types
            if item == "llm"
        )

        local_fallback_count = sum(
            1
            for item in response_judge_types
            if item == "local_mock"
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {
            "run_id": run_id,
            "judge_type": judge_type,
            "rubric_version": rubric_version,
            "llm_judge_count": llm_judge_count,
            "local_fallback_count": (
                local_fallback_count
            ),
            "judge_total_count": len(
                response_judge_types
            ),
            "results": results,
        }

    finally:

        db.close()


# ============================================================
# COMPARISON API
# ============================================================

@app.get("/compare")
def compare_runs(
    run_ids: list[int] = Query(
        ...,
        description=(
            "Run IDs to compare. "
            "Example: ?run_ids=23&run_ids=25"
        ),
    )
):
    """
    Compare multiple completed evaluation runs.

    The comparison is calculated per model and includes:

    - response count
    - average latency
    - total latency
    - average cost
    - total cost
    - standard evaluation metrics
    - LLM-as-a-Judge metrics
    - judge type
    - rubric version
    - real LLM judge count
    - local fallback count
    """

    db: Session = SessionLocal()

    try:

        # ====================================================
        # VALIDATE IDS
        # ====================================================

        unique_run_ids = list(
            dict.fromkeys(run_ids)
        )

        if len(unique_run_ids) < 2:

            raise HTTPException(
                status_code=400,
                detail=(
                    "At least two different run IDs "
                    "are required for comparison."
                ),
            )

        # ====================================================
        # LOAD RUNS
        # ====================================================

        runs = (
            db.query(Run)
            .filter(
                Run.id.in_(unique_run_ids)
            )
            .order_by(Run.id)
            .all()
        )

        found_ids = {
            run.id
            for run in runs
        }

        missing_ids = [
            run_id
            for run_id in unique_run_ids
            if run_id not in found_ids
        ]

        if missing_ids:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Run(s) not found: "
                    f"{missing_ids}"
                ),
            )

        comparison_models = set()
        comparison_runs = []

        # ====================================================
        # PROCESS EACH RUN
        # ====================================================

        for run in runs:

            responses = (
                db.query(Response)
                .filter(
                    Response.run_id == run.id
                )
                .order_by(Response.id)
                .all()
            )

            model_names = sorted(
                {
                    response.model_name
                    for response in responses
                }
            )

            comparison_models.update(
                model_names
            )

            model_results = []

            # =================================================
            # PROCESS EACH MODEL
            # =================================================

            for model_name in model_names:

                model_responses = [
                    response
                    for response in responses
                    if response.model_name
                    == model_name
                ]

                response_ids = [
                    response.id
                    for response
                    in model_responses
                ]

                # ---------------------------------------------
                # STANDARD METRICS
                # ---------------------------------------------

                standard_values = {
                    metric: []
                    for metric
                    in STANDARD_METRICS
                }

                if response_ids:

                    standard_scores = (
                        db.query(Score)
                        .filter(
                            Score.response_id.in_(
                                response_ids
                            )
                        )
                        .all()
                    )

                    for score in standard_scores:

                        if (
                            score.metric_name
                            in standard_values
                        ):

                            standard_values[
                                score.metric_name
                            ].append(
                                float(
                                    score.score_value
                                )
                            )

                standard_averages = {}

                for metric, values in (
                    standard_values.items()
                ):

                    if values:

                        standard_averages[
                            metric
                        ] = round(
                            sum(values)
                            / len(values),
                            4,
                        )

                    else:

                        standard_averages[
                            metric
                        ] = None

                # ---------------------------------------------
                # JUDGE METRICS
                # ---------------------------------------------

                judge_values = {
                    metric: []
                    for metric
                    in JUDGE_METRICS
                }

                judge_types = []
                rubric_versions = []

                if response_ids:

                    judge_scores = (
                        db.query(JudgeScore)
                        .filter(
                            JudgeScore.response_id.in_(
                                response_ids
                            )
                        )
                        .all()
                    )

                    for score in judge_scores:

                        if (
                            score.metric_name
                            in judge_values
                        ):

                            judge_values[
                                score.metric_name
                            ].append(
                                float(
                                    score.score_value
                                )
                            )

                        if score.judge_type:

                            judge_types.append(
                                score.judge_type
                            )

                        if score.rubric_version:

                            rubric_versions.append(
                                score.rubric_version
                            )

                judge_averages = {}

                for metric, values in (
                    judge_values.items()
                ):

                    if values:

                        judge_averages[
                            metric
                        ] = round(
                            sum(values)
                            / len(values),
                            4,
                        )

                    else:

                        judge_averages[
                            metric
                        ] = None

                # ---------------------------------------------
                # PERFORMANCE
                # ---------------------------------------------

                latencies = [
                    float(
                        response.latency_ms or 0.0
                    )
                    for response
                    in model_responses
                ]

                costs = [
                    float(
                        response.cost_usd or 0.0
                    )
                    for response
                    in model_responses
                ]

                total_latency = sum(
                    latencies
                )

                total_cost = sum(
                    costs
                )

                average_latency = (
                    total_latency
                    / len(latencies)
                    if latencies
                    else 0.0
                )

                average_cost = (
                    total_cost
                    / len(costs)
                    if costs
                    else 0.0
                )

                # ---------------------------------------------
                # JUDGE STATUS
                # ---------------------------------------------

                response_level_judge_types = []

                for response in model_responses:

                    response_judge_scores = (
                        db.query(JudgeScore)
                        .filter(
                            JudgeScore.response_id
                            == response.id
                        )
                        .all()
                    )

                    types_for_response = {
                        score.judge_type
                        for score
                        in response_judge_scores
                        if score.judge_type
                    }

                    if (
                        "llm"
                        in types_for_response
                    ):

                        response_level_judge_types.append(
                            "llm"
                        )

                    elif (
                        "local_mock"
                        in types_for_response
                    ):

                        response_level_judge_types.append(
                            "local_mock"
                        )

                llm_count = sum(
                    1
                    for item
                    in response_level_judge_types
                    if item == "llm"
                )

                local_count = sum(
                    1
                    for item
                    in response_level_judge_types
                    if item == "local_mock"
                )

                judge_status = classify_judge_status(
                    response_level_judge_types
                )

                unique_rubrics = sorted(
                    {
                        str(item)
                        for item
                        in rubric_versions
                        if item
                    }
                )

                if len(unique_rubrics) == 1:

                    rubric_version = (
                        unique_rubrics[0]
                    )

                elif unique_rubrics:

                    rubric_version = (
                        unique_rubrics
                    )

                else:

                    rubric_version = None

                # ---------------------------------------------
                # MODEL COMPARISON RESULT
                # ---------------------------------------------

                model_results.append(
                    {
                        "model_name": model_name,
                        "response_count": len(
                            model_responses
                        ),
                        "average_latency_ms": round(
                            average_latency,
                            4,
                        ),
                        "total_latency_ms": round(
                            total_latency,
                            4,
                        ),
                        "average_cost_usd": round(
                            average_cost,
                            8,
                        ),
                        "total_cost_usd": round(
                            total_cost,
                            8,
                        ),
                        "standard_metrics": (
                            standard_averages
                        ),
                        "judge_metrics": (
                            judge_averages
                        ),
                        "judge_type": judge_status,
                        "rubric_version": (
                            rubric_version
                        ),
                        "llm_judge_count": (
                            llm_count
                        ),
                        "local_fallback_count": (
                            local_count
                        ),
                        "judge_total_count": (
                            len(
                                response_level_judge_types
                            )
                        ),
                    }
                )

            # =================================================
            # RUN RESULT
            # =================================================

            comparison_runs.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "dataset_id": run.dataset_id,
                    "models_used": run.models_used,
                    "prompt_version": (
                        run.prompt_version
                    ),
                    "temperature": run.temperature,
                    "max_tokens": run.max_tokens,
                    "created_at": run.created_at,
                    "models": model_results,
                }
            )

        # ====================================================
        # RETURN COMPARISON
        # ====================================================

        return {
            "comparison_count": len(
                comparison_runs
            ),
            "run_ids": unique_run_ids,
            "models": sorted(
                comparison_models
            ),
            "runs": comparison_runs,
        }

    finally:

        db.close()