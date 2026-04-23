from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Literal
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    mode: str = "real"  # "mock" hoặc "real"

    def _get_runtime(self):
        """Chọn runtime dựa vào mode."""
        if self.mode == "mock":
            from . import mock_runtime as rt
        else:
            from . import llm_runtime as rt
        return rt

    def run(self, example: QAExample) -> RunRecord:
        rt = self._get_runtime()
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            # --- Actor ---
            if self.mode == "mock":
                answer = rt.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
                actor_tokens, actor_latency = 320, 160  # mock estimates
            else:
                answer, actor_tokens, actor_latency = rt.actor_answer(
                    example, attempt_id, self.agent_type, reflection_memory
                )

            # --- Evaluator ---
            if self.mode == "mock":
                judge = rt.evaluator(example, answer)
                eval_tokens, eval_latency = 65, 40  # mock estimates
            else:
                judge, eval_tokens, eval_latency = rt.evaluator(example, answer)

            token_estimate = actor_tokens + eval_tokens
            latency_ms = actor_latency + eval_latency

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_estimate,
                latency_ms=int(latency_ms),
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            # --- Reflexion logic (CORE TODO đã hoàn thiện) ---
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                if self.mode == "mock":
                    ref_entry = rt.reflector(example, attempt_id, judge)
                    ref_tokens, ref_latency = 120, 90
                else:
                    ref_entry, ref_tokens, ref_latency = rt.reflector(
                        example, attempt_id, judge
                    )
                reflection_memory.append(ref_entry.next_strategy)
                reflections.append(ref_entry)
                trace.reflection = ref_entry
                trace.token_estimate += ref_tokens
                trace.latency_ms += int(ref_latency)

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)

        # Failure mode detection
        if final_score == 1:
            failure_mode = "none"
        elif self.mode == "mock":
            failure_mode = rt.FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        else:
            failure_mode = _detect_failure_mode(example, final_answer, reflections)

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


def _detect_failure_mode(example, answer, reflections) -> str:
    """Phát hiện failure mode dựa trên nội dung reflection."""
    if not reflections:
        return "wrong_final_answer"
    reasons = " ".join(r.failure_reason.lower() for r in reflections)
    if "hop" in reasons or "incomplete" in reasons:
        return "incomplete_multi_hop"
    if "drift" in reasons or "wrong entity" in reasons:
        return "entity_drift"
    if "loop" in reasons or "same answer" in reasons:
        return "looping"
    if "overfit" in reasons:
        return "reflection_overfit"
    return "wrong_final_answer"


class ReActAgent(BaseAgent):
    def __init__(self, mode: str = "real") -> None:
        super().__init__(agent_type="react", max_attempts=1, mode=mode)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, mode: str = "real") -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, mode=mode)

    def run(self, example: QAExample) -> RunRecord:
        # Adaptive: điều chỉnh max_attempts theo difficulty
        difficulty_map = {"easy": 2, "medium": 3, "hard": 4}
        original = self.max_attempts
        self.max_attempts = difficulty_map.get(example.difficulty, 3)
        result = super().run(example)
        self.max_attempts = original  
        return result