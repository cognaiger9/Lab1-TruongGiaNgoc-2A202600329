from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import (
    FAILURE_MODE_BY_QID,
    actor_answer,
    evaluator,
    get_usage_stats,
    reflector,
    reset_usage_stats,
)
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


def _format_reflection(entry: ReflectionEntry) -> str:
    keywords = ", ".join(entry.keywords) if entry.keywords else "n/a"
    return (
        f"Attempt {entry.attempt_id} [{entry.failure_mode}]: {entry.diagnosis} "
        f"Strategy: {entry.strategy} (focus on: {keywords})"
    )


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        last_judge_failure_mode = "wrong_final_answer"

        for attempt_id in range(1, self.max_attempts + 1):
            reset_usage_stats()
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = evaluator(example, answer)

            is_last_attempt = attempt_id == self.max_attempts
            reflection_entry: ReflectionEntry | None = None
            if self.agent_type == "reflexion" and judge.score == 0 and not is_last_attempt:
                reflection_entry = reflector(example, attempt_id, judge)
                reflections.append(reflection_entry)
                reflection_memory.append(_format_reflection(reflection_entry))

            stats = get_usage_stats()
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection_entry,
                token_estimate=stats["tokens"],
                latency_ms=stats["latency_ms"],
            )

            final_answer = answer
            final_score = judge.score
            last_judge_failure_mode = judge.failure_mode
            traces.append(trace)

            if judge.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = (
            "none"
            if final_score == 1
            else FAILURE_MODE_BY_QID.get(example.qid, last_judge_failure_mode)
        )
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
