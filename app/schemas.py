from pydantic import BaseModel
from typing import Optional, List


# ============================================================
# PROMPT
# ============================================================

class PromptIn(BaseModel):
    question: str
    context: Optional[str] = None
    expected_output: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None


# ============================================================
# DATASET
# ============================================================

class DatasetCreate(BaseModel):
    name: str
    prompts: List[PromptIn]


# ============================================================
# RUN / EXPERIMENT
# ============================================================

class RunCreate(BaseModel):
    name: str
    dataset_id: int
    models: List[str]

    # Experiment tracking
    prompt_version: str = "v1"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ============================================================
# SCORE
# ============================================================

class ScoreOut(BaseModel):
    metric_name: str
    score_value: float
    explanation: Optional[str] = None

    class Config:
        from_attributes = True


# ============================================================
# RESPONSE
# ============================================================

class ResponseOut(BaseModel):
    id: int
    model_name: str
    output_text: str
    latency_ms: float
    cost_usd: float
    scores: List[ScoreOut] = []

    class Config:
        from_attributes = True