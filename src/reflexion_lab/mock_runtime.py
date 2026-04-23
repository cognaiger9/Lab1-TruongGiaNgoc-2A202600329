from __future__ import annotations
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

MODEL = os.getenv("REFLEXION_MODEL", "gpt-4o-mini")

FAILURE_MODE_BY_QID: dict[str, str] = {}

_client: Optional[OpenAI] = None

import threading

_usage_local = threading.local()


def _get_usage() -> dict:
    if not hasattr(_usage_local, "stats"):
        _usage_local.stats = {"tokens": 0, "latency_ms": 0}
    return _usage_local.stats


def reset_usage_stats() -> None:
    _get_usage().update({"tokens": 0, "latency_ms": 0})


def get_usage_stats() -> dict:
    return dict(_get_usage())


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _format_context(example: QAExample) -> str:
    return "\n\n".join(f"[{c.title}]\n{c.text}" for c in example.context)


def _chat(system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
    start = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"} if json_mode else {"type": "text"},
    )
    stats = _get_usage()
    stats["latency_ms"] += int((time.perf_counter() - start) * 1000)
    if resp.usage is not None:
        stats["tokens"] += resp.usage.total_tokens
    return resp.choices[0].message.content or ""


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> str:
    memory_block = ""
    if agent_type == "reflexion" and reflection_memory:
        bullets = "\n".join(f"- {r}" for r in reflection_memory)
        memory_block = (
            "\n\nPrevious reflections (use to avoid repeating mistakes):\n" + bullets
        )

    user = (
        f"Question:\n{example.question}\n\n"
        f"Context:\n{_format_context(example)}"
        f"{memory_block}\n\n"
        f"Attempt #{attempt_id}. Return ONLY the final answer as a short entity or phrase "
        "(no explanation, no punctuation beyond the answer itself)."
    )
    return _chat(ACTOR_SYSTEM, user).strip()


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(
            score=1,
            is_correct=True,
            reason="Normalized predicted answer matches the gold answer.",
            failure_mode="none",
            missing_hops=[],
        )

    user = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}\n\n"
        f"Context:\n{_format_context(example)}\n\n"
        "Return a JSON object with EXACTLY these fields:\n"
        '  "score": 0 or 1 (1 only if predicted answer is semantically equivalent to the gold answer),\n'
        '  "is_correct": boolean,\n'
        '  "reason": short string explaining the judgement,\n'
        '  "failure_mode": one of "none", "entity_drift", "incomplete_multi_hop", '
        '"wrong_final_answer", "looping", "reflection_overfit",\n'
        '  "missing_hops": array of strings naming bridge entities the predicted answer failed to reach.'
    )
    raw = _chat(EVALUATOR_SYSTEM, user, json_mode=True)
    try:
        return JudgeResult.model_validate_json(raw)
    except Exception:
        data = json.loads(raw)
        data.setdefault("is_correct", bool(data.get("score", 0)))
        data.setdefault("failure_mode", "wrong_final_answer")
        data.setdefault("missing_hops", [])
        return JudgeResult.model_validate(data)


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    missing = ", ".join(judge.missing_hops) or "none"
    user = (
        f"Question: {example.question}\n"
        f"Context:\n{_format_context(example)}\n\n"
        f"Judge verdict: {judge.reason}\n"
        f"Failure mode: {judge.failure_mode}\n"
        f"Missing hops: {missing}\n\n"
        f"This was attempt #{attempt_id}. Produce a JSON object with EXACTLY these fields:\n"
        '  "attempt_id": integer,\n'
        '  "failure_mode": string (same set as the judge),\n'
        '  "diagnosis": string (what went wrong in the previous attempt),\n'
        '  "lesson": string (general rule to avoid repeating the mistake),\n'
        '  "strategy": string (concrete plan for the next attempt),\n'
        '  "keywords": array of strings (entities the next attempt should attend to).'
    )
    raw = _chat(REFLECTOR_SYSTEM, user, json_mode=True)
    data = json.loads(raw)
    data.setdefault("attempt_id", attempt_id)
    data.setdefault("failure_mode", judge.failure_mode)
    data.setdefault("keywords", [])
    return ReflectionEntry.model_validate(data)
