from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)

from sqlalchemy.orm import relationship

from .database import Base


# ============================================================
# DATASET
# ============================================================

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(
        Integer,
        primary_key=True,
    )

    name = Column(
        String(255),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    prompts = relationship(
        "Prompt",
        back_populates="dataset",
        cascade="all, delete",
    )


# ============================================================
# PROMPT
# ============================================================

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(
        Integer,
        primary_key=True,
    )

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id"),
    )

    question = Column(
        Text,
        nullable=False,
    )

    context = Column(
        Text,
        nullable=True,
    )

    expected_output = Column(
        Text,
        nullable=True,
    )

    category = Column(
        String(100),
        nullable=True,
    )

    difficulty = Column(
        String(50),
        nullable=True,
    )

    dataset = relationship(
        "Dataset",
        back_populates="prompts",
    )


# ============================================================
# RUN
# ============================================================

class Run(Base):
    __tablename__ = "runs"

    id = Column(
        Integer,
        primary_key=True,
    )

    name = Column(
        String(255),
        nullable=False,
    )

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id"),
    )

    models_used = Column(
        String(500),
    )

    prompt_version = Column(
        String(50),
        default="v1",
    )

    temperature = Column(
        Float,
        nullable=True,
    )

    max_tokens = Column(
        Integer,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    responses = relationship(
        "Response",
        back_populates="run",
        cascade="all, delete",
    )


# ============================================================
# RESPONSE
# ============================================================

class Response(Base):
    __tablename__ = "responses"

    id = Column(
        Integer,
        primary_key=True,
    )

    run_id = Column(
        Integer,
        ForeignKey("runs.id"),
    )

    model_name = Column(
        String(100),
    )

    output_text = Column(
        Text,
    )

    latency_ms = Column(
        Float,
    )

    prompt_tokens = Column(
        Integer,
        default=0,
    )

    completion_tokens = Column(
        Integer,
        default=0,
    )

    cost_usd = Column(
        Float,
        default=0.0,
    )

    status = Column(
        String(50),
        default="success",
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    run = relationship(
        "Run",
        back_populates="responses",
    )

    scores = relationship(
        "Score",
        back_populates="response",
        cascade="all, delete",
    )

    judge_scores = relationship(
        "JudgeScore",
        back_populates="response",
        cascade="all, delete",
    )


# ============================================================
# STANDARD SCORE
# ============================================================

class Score(Base):
    __tablename__ = "scores"

    id = Column(
        Integer,
        primary_key=True,
    )

    response_id = Column(
        Integer,
        ForeignKey("responses.id"),
    )

    metric_name = Column(
        String(100),
    )

    score_value = Column(
        Float,
    )

    explanation = Column(
        Text,
        nullable=True,
    )

    response = relationship(
        "Response",
        back_populates="scores",
    )


# ============================================================
# LLM-AS-A-JUDGE SCORE
# ============================================================

class JudgeScore(Base):
    __tablename__ = "judge_scores"

    id = Column(
        Integer,
        primary_key=True,
    )

    response_id = Column(
        Integer,
        ForeignKey("responses.id"),
    )

    judge_type = Column(
        String(50),
        nullable=True,
    )

    rubric_version = Column(
        String(50),
        nullable=True,
    )

    metric_name = Column(
        String(100),
        nullable=False,
    )

    score_value = Column(
        Float,
        nullable=False,
    )

    explanation = Column(
        Text,
        nullable=True,
    )

    response = relationship(
        "Response",
        back_populates="judge_scores",
    )