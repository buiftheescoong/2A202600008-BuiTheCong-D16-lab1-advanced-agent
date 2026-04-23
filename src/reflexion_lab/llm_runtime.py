from __future__ import annotations
import json
import os
import re
import time
import random

from openai import OpenAI, RateLimitError, APIStatusError
from dotenv import load_dotenv

from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM, COMPRESSOR_SYSTEM
from .utils import normalize_answer

load_dotenv()

# ---- Cấu hình model ----
# Actor dùng model YẾU để chứng minh tác dụng của Reflexion
ACTOR_MODEL = os.getenv("ACTOR_MODEL", "gpt-3.5-turbo")
# Evaluator & Reflector dùng model TỐT HƠN để JSON output chuẩn
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Token tracking toàn cục ----
_total_tokens = 0

# ---- FAILURE_MODE_BY_QID (không dùng mock, để tương thích với agents.py) ----
FAILURE_MODE_BY_QID: dict[str, str] = {}

# ---- Rate limit config ----
# Delay giữa các lần gọi API (giây) — tránh hit RPM limit
INTER_CALL_DELAY = float(os.getenv("INTER_CALL_DELAY", "0.5"))
# Số lần retry khi bị rate limit
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
# Base delay cho exponential backoff (giây)
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "2.0"))


def get_total_tokens() -> int:
    return _total_tokens


def reset_tokens() -> None:
    global _total_tokens
    _total_tokens = 0


def _call_openai(system: str, user: str, model: str) -> tuple[str, int, float]:
    """Gọi OpenAI API với retry + exponential backoff khi bị rate limit.

    Trả về (response_text, total_tokens, latency_ms).
    Token thực tế lấy từ response.usage.prompt_tokens + completion_tokens.
    """
    global _total_tokens

    # Delay nhỏ giữa mỗi lần gọi để không bị RPM throttle
    time.sleep(INTER_CALL_DELAY)

    for attempt in range(MAX_RETRIES):
        start = time.time()
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=512,
            )
            latency_ms = (time.time() - start) * 1000
            # Đọc token thực tế từ usage object
            usage = response.usage
            tokens = usage.prompt_tokens + usage.completion_tokens
            _total_tokens += tokens
            content = response.choices[0].message.content or ""
            return content, tokens, latency_ms

        except RateLimitError as e:
            # Bị rate limit → exponential backoff + jitter
            wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
            print(f"[yellow]Rate limit hit (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait:.1f}s...[/yellow]")
            time.sleep(wait)

        except APIStatusError as e:
            # Lỗi server (5xx) → retry
            if e.status_code >= 500:
                wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
                print(f"[yellow]API error {e.status_code} (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait:.1f}s...[/yellow]")
                time.sleep(wait)
            else:
                raise  # 4xx (bad request, auth) → không retry

    raise RuntimeError(f"OpenAI API failed after {MAX_RETRIES} retries")


def _parse_json(text: str) -> dict:
    """Parse JSON từ LLM response, xử lý trường hợp có markdown code block."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    # Tìm JSON object trong text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, float]:
    """Gọi Actor LLM (model yếu) để trả lời câu hỏi.

    Trả về (answer, tokens, latency_ms).
    """
    context_text = "\n\n".join(
        f"### {c.title}\n{c.text}" for c in example.context
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {example.question}"
    if reflection_memory:
        memory_text = "\n".join(f"- {m}" for m in reflection_memory)
        user_prompt += (
            f"\n\nReflection from previous attempts (avoid these mistakes):\n{memory_text}"
        )

    content, tokens, latency = _call_openai(ACTOR_SYSTEM, user_prompt, model=ACTOR_MODEL)

    # Parse "Final Answer: ..." từ response
    answer = content.strip()
    match = re.search(r"Final Answer:\s*(.+)", content, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
    return answer, tokens, latency


def evaluator(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, int, float]:
    """Gọi Evaluator LLM (model tốt) để chấm điểm.

    Trả về (JudgeResult, tokens, latency_ms).
    """
    user_prompt = (
        f"Question: {example.question}\n"
        f"Gold Answer: {example.gold_answer}\n"
        f"Predicted Answer: {answer}"
    )
    content, tokens, latency = _call_openai(EVALUATOR_SYSTEM, user_prompt, model=EVAL_MODEL)
    try:
        data = _parse_json(content)
        return (
            JudgeResult(
                score=int(data.get("score", 0)),
                reason=data.get("reason", ""),
                missing_evidence=data.get("missing_evidence", []),
                spurious_claims=data.get("spurious_claims", []),
            ),
            tokens,
            latency,
        )
    except Exception:
        # Fallback: so sánh trực tiếp nếu JSON parse thất bại
        is_match = normalize_answer(example.gold_answer) == normalize_answer(answer)
        return (
            JudgeResult(
                score=1 if is_match else 0,
                reason="Exact match fallback" if is_match else "No match (JSON parse failed)",
            ),
            tokens,
            latency,
        )


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, float]:
    """Gọi Reflector LLM (model tốt) để phân tích lỗi và đề xuất chiến thuật.

    Trả về (ReflectionEntry, tokens, latency_ms).
    """
    user_prompt = (
        f"Question: {example.question}\n"
        f"Wrong Answer Given at attempt {attempt_id}\n"
        f"Evaluation: score={judge.score}, reason={judge.reason}\n"
        f"Missing Evidence: {judge.missing_evidence}\n"
        f"Spurious Claims: {judge.spurious_claims}"
    )
    content, tokens, latency = _call_openai(REFLECTOR_SYSTEM, user_prompt, model=EVAL_MODEL)
    try:
        data = _parse_json(content)
        return (
            ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=data.get("failure_reason", judge.reason),
                lesson=data.get("lesson", ""),
                next_strategy=data.get("next_strategy", ""),
            ),
            tokens,
            latency,
        )
    except Exception:
        return (
            ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Could not parse reflection",
                next_strategy="Try again more carefully",
            ),
            tokens,
            latency,
        )


def compress_memory(reflection_memory: list[str]) -> tuple[list[str], int, float]:
    """Bonus memory_compression: nén reflection memory khi > 3 entries.

    Trả về (compressed_memory, tokens, latency_ms).
    """
    if len(reflection_memory) <= 3:
        return reflection_memory, 0, 0.0
    text = "\n".join(f"- {m}" for m in reflection_memory)
    summary, tokens, latency = _call_openai(COMPRESSOR_SYSTEM, text, model=EVAL_MODEL)
    return [summary.strip()], tokens, latency
