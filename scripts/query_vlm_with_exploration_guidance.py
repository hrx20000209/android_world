#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query the local AndroidWorld VLM port directly with the same prompt template.

Usage:
  python scripts/query_vlm_with_exploration_guidance.py \
    --instruction "Add a new expense of 20 dollars in OpenTracks." \
    --history-file /path/to/history.txt \
    --image /path/to/screen.png \
    --add-info "exploration candidates: ... (from your internal agent hints)"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "http://localhost:8081/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _as_list(value: str | None) -> list[str]:
    """Parse a CSV-like string list for history."""
    if not value:
        return []
    stripped = value.strip()
    if not stripped:
        return []
    return [x.strip() for x in stripped.split("\n") if x.strip()]


def _normalize_history(values: list[str] | None, history_file: str | None) -> list[str]:
    items: list[str] = []
    if values:
        items.extend(values)
    if history_file:
        path = Path(history_file)
        if not path.exists():
            raise FileNotFoundError(f"history_file not found: {path}")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                items.append(line)
    return items


def build_agent_user_prompt(
    instruction: str,
    history: list[str],
    add_info: str = "",
    add_thought: bool = True,
    include_guidance: bool = True,
) -> str:
    """Build the user prompt following the current AndroidWorld explore-agent style.

    This mirrors mobile_agent_utils.generate_user_prompt_single_image and then
    appends an exploration recovery request.
    """
    user_prompt = f"The user query: {instruction}"

    if add_thought:
        if history:
            user_prompt += f"\nTask progress (You have done the following operation on the current device): {history}.\n"
        if add_info:
            user_prompt += f"\nThe following tips can help you complete user tasks: {add_info}."
        add_reflection = os.environ.get("ADD_REFLECTION", "")
        if add_reflection:
            user_prompt += (
                "\nBefore answering, you must:\n"
                "1. Analyze if the previous action was appropriate.\n"
                "2. Verify if its effects match expectations.\n"
                "3. When the action was executed incorrectly, attempt correction.\n"
                "4. Explain reasoning step-by-step before the next action.\n"
            )
            user_prompt += "\nFill the content in <thinking></thinking> tags following this structure:\n"
            user_prompt += "<thinking>\n"
            user_prompt += "[Action Analysis]\n(1) Correctness assessment: ...\n(2) Outcome alignment: ...\n"
            user_prompt += "(3) Observation of the current screenshot: ...\n"
            user_prompt += "(4) Next Step Planning: ...\n"
            user_prompt += "(5) Action: ...\n</thinking>\n"
            user_prompt += "Finally provide the <tool_call></tool_call> XML tags."
        else:
            user_prompt += (
                "\nBefore answering, explain your reasoning step-by-step in <thinking></thinking> tags, "
                "and insert them before the <tool_call></tool_call> XML tags."
            )
            user_prompt += "\nAfter answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags."
    if not add_thought:
        user_prompt += ""

    if include_guidance:
        user_prompt += "\n\nAdditionally, you must output an explicit 'Exploration path guidance' section:\n"
        user_prompt += (
            "Put it inside <exploration_path_guidance></exploration_path_guidance> tags.\n"
            "It should briefly explain recovery plan across alternatives when current state is stale or fail-prone.\n"
            "Use this exact structure:\n"
            "<exploration_path_guidance>\n"
            "1) Why this screen may have deviated from expected state\n"
            "2) 2~3 candidate recovery branches to try\n"
            "3) Which branch is highest priority and why\n"
            "</exploration_path_guidance>"
        )

    user_prompt += (
        "\n\nAlso always provide a <tool_call></tool_call> XML tag for the next action with "
        "one JSON object and keys: name / arguments."
    )
    return user_prompt


def _pil_to_data_url(image_path: str, quality: int = 90) -> str:
    import io

    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_message_payload(
    system_prompt: str,
    user_prompt: str,
    image_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for image_path in image_paths or []:
        data_url = _pil_to_data_url(image_path)
        # Keep openai-compatible shape; common llama.cpp-compatible backends accept this form.
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def _post_llm(
    api_url: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    top_k: int = -1,
    history_n: int = 3,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    repeat_penalty: float | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    extra_params: dict[str, Any] | None = None,
    timeout: float = 3000.0,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "history_n": history_n,
        "stream": False,
    }
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if repeat_penalty is not None:
        payload["repeat_penalty"] = repeat_penalty
    if seed is not None:
        payload["seed"] = seed
    if stop:
        payload["stop"] = stop
        if extra_params:
            payload.update(extra_params)
    response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout)
    response.raise_for_status()
    raw = response.json()
    content = ""
    if isinstance(raw, dict):
        try:
            content = raw["choices"][0]["message"]["content"]
        except Exception:
            if "error" in raw:
                raise RuntimeError(f"LLM backend returned error: {raw['error']}")
            raise
    return str(content), raw


def _parse_extra_params(items: list[str] | None) -> dict[str, Any]:
    if not items:
        return {}
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra-llm-param 需要 key=value 格式: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"invalid extra param key: {item}")
        if value.lower() in {"true", "false"}:
            parsed_value: Any = value.lower() == "true"
        else:
            try:
                if "." in value:
                    parsed_value = float(value)
                else:
                    parsed_value = int(value)
            except ValueError:
                parsed_value = value
        out[key] = parsed_value
    return out


def _extract_region(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _extract_tool_call(text: str) -> str:
    return _extract_region(text, "tool_call")


def _extract_conclusion(text: str) -> str:
    return _extract_region(text, "conclusion")


def _extract_guidance(text: str) -> str:
    return _extract_region(text, "exploration_path_guidance")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Direct VLM query with exploration guidance.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="LLM server endpoint.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="system prompt string.")
    parser.add_argument("--instruction", required=True, help="Current user task/query.")
    parser.add_argument(
        "--history",
        default="",
        help="History actions, one per line.",
    )
    parser.add_argument(
        "--history-file",
        default="",
        help="Optional file with history actions (one per line).",
    )
    parser.add_argument(
        "--add-info",
        default="",
        help="Optional hint text injected as add_info, equivalent to explorer agent extra tips.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Screenshot image path; can be repeated for multiple images.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--history-n", type=int, default=3)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop", action="append", default=None, help="stop token，可重复指定。")
    parser.add_argument(
        "--extra-llm-param",
        action="append",
        default=None,
        help="透传额外参数，格式 key=value，可重复指定，覆盖未知字段。"
    )
    parser.add_argument("--timeout", type=float, default=3000.0)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw request/response JSON for copy-paste reproduction.",
    )
    parser.add_argument("--no-guidance", action="store_true", help="Skip exploration guidance block.")
    args = parser.parse_args()

    history = _normalize_history(_as_list(args.history), args.history_file)
    user_prompt = build_agent_user_prompt(
        instruction=args.instruction,
        history=history,
        add_info=args.add_info,
        add_thought=True,
        include_guidance=not args.no_guidance,
    )
    messages = _build_message_payload(
        system_prompt=args.system_prompt,
        user_prompt=user_prompt,
        image_paths=args.image,
    )

    if args.raw:
        print("========== REQUEST ==========")
        request_payload = {
            "api_url": args.api_url,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "history_n": args.history_n,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "repeat_penalty": args.repeat_penalty,
            "seed": args.seed,
            "stop": args.stop,
            "extra_llm_param": _parse_extra_params(args.extra_llm_param),
            "timeout": args.timeout,
        }
        print(json.dumps(request_payload, ensure_ascii=False, indent=2))

    content, raw = _post_llm(
        api_url=args.api_url,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        history_n=args.history_n,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
        stop=args.stop,
        extra_params=_parse_extra_params(args.extra_llm_param),
        timeout=args.timeout,
    )

    guidance = _extract_guidance(content)
    tool_call = _extract_tool_call(content)
    conclusion = _extract_conclusion(content)

    print("========== VLM RESPONSE ==========")
    print(content)
    print("\n========== EXTRACTION ==========")
    print(f"[tool_call]\n{tool_call or '(none)'}")
    print(f"\n[conclusion]\n{conclusion or '(none)'}")
    print(f"\n[exploration_path_guidance]\n{guidance or '(none)'}")

    if args.raw:
        print("\n========== RAW RESPONSE ==========")
        print(json.dumps(raw, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
