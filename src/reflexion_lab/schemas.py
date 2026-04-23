from __future__ import annotations
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

FailureMode = Literal[
    "none",
    "entity_drift",
    "incomplete_multi_hop",
    "wrong_final_answer",
    "looping",
    "reflection_overfit",
]

class ContextChunk(BaseModel):
    title: str
    text: str

class QAExample(BaseModel):
    qid: str
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    gold_answer: str
    context: list[ContextChunk]

class JudgeResult(BaseModel):
    score: int = Field(..., ge=0, le=1, description="1 if the answer is correct, 0 otherwise")
    is_correct: bool = Field(..., description="Whether the predicted answer matches the gold answer")
    reason: str = Field(..., description="Short explanation of the judgement")
    failure_mode: FailureMode = Field("none", description="Classified failure mode when the answer is wrong")
    missing_hops: list[str] = Field(
        default_factory=list,
        description="Bridge entities / hops the answer failed to cover (empty if correct)",
    )

class ReflectionEntry(BaseModel):
    attempt_id: int = Field(..., description="Which attempt this reflection was produced from")
    failure_mode: FailureMode = Field(..., description="Failure mode this reflection is addressing")
    diagnosis: str = Field(..., description="What went wrong in the previous attempt")
    lesson: str = Field(..., description="General lesson learned to avoid repeating the mistake")
    strategy: str = Field(..., description="Concrete strategy the next attempt should try")
    keywords: list[str] = Field(
        default_factory=list,
        description="Entities or terms the next attempt should pay attention to",
    )

class AttemptTrace(BaseModel):
    attempt_id: int
    answer: str
    score: int
    reason: str
    reflection: Optional[ReflectionEntry] = None
    token_estimate: int = 0
    latency_ms: int = 0

class RunRecord(BaseModel):
    qid: str
    question: str
    gold_answer: str
    agent_type: Literal["react", "reflexion"]
    predicted_answer: str
    is_correct: bool
    attempts: int
    token_estimate: int
    latency_ms: int
    failure_mode: FailureMode
    reflections: list[ReflectionEntry] = Field(default_factory=list)
    traces: list[AttemptTrace] = Field(default_factory=list)

class ReportPayload(BaseModel):
    meta: dict
    summary: dict
    failure_modes: dict
    examples: list[dict]
    extensions: list[str]
    discussion: str

class ReflexionState(TypedDict):
    question: str
    context: list[str]
    trajectory: list[str]
    reflection_memory: list[str]
    attempt_count: int
    success: bool
    final_answer: str
