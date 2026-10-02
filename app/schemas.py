from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class PromptIn(BaseModel):
    question: str
    context: Optional[str] = None
    expected_output: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None


class DatasetCreate(BaseModel):
    name: str
    prompts: List[PromptIn]


class RunCreate(BaseModel):
    name: str
    dataset_id: int
    models: List[str]
    prompt_version: str = "v1"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ScoreOut(BaseModel):
    metric_name: str
    score_value: float
    explanation: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ResponseOut(BaseModel):
    id: int
    model_name: str
    output_text: str
    latency_ms: float
    cost_usd: float
    scores: List[ScoreOut] = []

    model_config = ConfigDict(from_attributes=True)