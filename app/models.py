from datetime import datetime

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


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)

    name = Column(
        String(255),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    prompts = relationship(
        "Prompt",
        back_populates="dataset",
        cascade="all, delete",
    )


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True)

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


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)

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
        default=datetime.utcnow,
    )

    responses = relationship(
        "Response",
        back_populates="run",
        cascade="all, delete",
    )


class Response(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True)

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

    cost_usd = Column(
        Float,
    )

    prompt_tokens = Column(
        Integer,
        nullable=True,
    )

    completion_tokens = Column(
        Integer,
        nullable=True,
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


class Score(Base):
    """
    One row per metric per model response.
    """

    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)

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


class JudgeScore(Base):
    """
    One row per LLM-as-a-Judge metric per model response.
    """

    __tablename__ = "judge_scores"

    id = Column(Integer, primary_key=True)

    response_id = Column(
        Integer,
        ForeignKey("responses.id"),
        nullable=False,
    )

    judge_type = Column(
        String(50),
        nullable=False,
    )

    rubric_version = Column(
        String(50),
        nullable=False,
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

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    response = relationship(
        "Response",
    )