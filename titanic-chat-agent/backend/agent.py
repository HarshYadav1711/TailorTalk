"""LangChain agent orchestration for Titanic dataset questions."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_ollama import ChatOllama

from backend.data_loader import load_titanic_df
from backend.tools import average, calculate_percentage, count_by, histogram, summary_stats

SYSTEM_PROMPT = """
You are a Titanic dataset analysis assistant.
Rules you must follow:
1) Always use the provided tools for any calculation, counting, percentages, averages, grouping, or plots.
2) Never do math in your head and never guess values.
3) If a request cannot be answered with available tools, respond exactly: "I cannot compute that".
4) Keep responses concise and factual.
5) If a tool returns structured data, summarize it clearly in plain text.
"""


_PLANNER_LLM = ChatOllama(
    model="mistral",
    temperature=0,
    format="json",
    num_predict=100,
    client_kwargs={"timeout": 6},
)

_TOOL_REGISTRY = {
    "calculate_percentage": calculate_percentage,
    "average": average,
    "count_by": count_by,
    "summary_stats": summary_stats,
    "histogram": histogram,
}


def _available_columns() -> list[str]:
    return list(load_titanic_df().columns)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _fallback_plan(question: str) -> dict[str, Any]:
    q = question.lower()
    columns = _available_columns()
    matched_column = next((col for col in columns if col in q), None)

    if "histogram" in q and matched_column:
        return {"tool": "histogram", "args": {"column": matched_column}}
    if "summary" in q and matched_column:
        return {"tool": "summary_stats", "args": {"column": matched_column}}
    if ("average" in q or "mean" in q) and matched_column:
        return {"tool": "average", "args": {"column": matched_column}}
    if "fare" in q and "histogram" not in q and "summary" not in q:
        return {"tool": "average", "args": {"column": "fare"}}
    if "age" in q and "histogram" not in q and "summary" not in q:
        return {"tool": "average", "args": {"column": "age"}}
    if "count" in q and ("by" in q or "each" in q) and matched_column:
        return {"tool": "count_by", "args": {"column": matched_column}}
    if "percentage" in q or "percent" in q:
        if "surviv" in q:
            target = "1" if any(w in q for w in ("survived", "survive")) else "0"
            return {"tool": "calculate_percentage", "args": {"column": "survived", "value": target}}
        if "female" in q or "male" in q:
            target = "female" if "female" in q else "male"
            return {"tool": "calculate_percentage", "args": {"column": "sex", "value": target}}
    return {"tool": "none", "args": {}}


def _plan_tool_call(question: str) -> dict[str, Any]:
    planner_prompt = f"""
{SYSTEM_PROMPT.strip()}
You must return ONLY valid JSON with this schema:
{{
  "tool": "calculate_percentage|average|histogram|count_by|summary_stats|none",
  "args": {{}}
}}
Available columns: {", ".join(_available_columns())}
Tool argument rules:
- calculate_percentage: args={{"column":"<column>","value":"<value>"}}
- average: args={{"column":"<numeric_column>"}}
- histogram: args={{"column":"<numeric_column>","bins":20}} (bins optional)
- count_by: args={{"column":"<column>"}}
- summary_stats: args={{"column":"<numeric_column>"}}
- if unsupported or ambiguous, return {{"tool":"none","args":{{}}}}
Question: {question}
"""
    try:
        raw = _PLANNER_LLM.invoke(planner_prompt)
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
    except Exception:
        return _fallback_plan(question)

    parsed = _extract_json_object(content)
    if not parsed:
        return _fallback_plan(question)
    if parsed.get("tool") not in _TOOL_REGISTRY and parsed.get("tool") != "none":
        return _fallback_plan(question)
    if not isinstance(parsed.get("args", {}), dict):
        return _fallback_plan(question)
    planned = {"tool": parsed.get("tool", "none"), "args": parsed.get("args", {})}

    q = question.lower().strip()
    if "what was the fare" in q or "average fare" in q:
        return {"tool": "average", "args": {"column": "fare"}}
    if "what was the age" in q or "average age" in q:
        return {"tool": "average", "args": {"column": "age"}}
    if planned["tool"] == "count_by" and not any(k in q for k in ("count", "how many", "by", "group")):
        return _fallback_plan(question)
    return planned


def _format_fallback(tool_result: dict[str, Any]) -> str:
    tool = tool_result.get("tool")
    if tool == "average":
        return f"Average {tool_result['column']}: {tool_result['average']}."
    if tool == "calculate_percentage":
        return (
            f"{tool_result['percentage']}% of rows have {tool_result['column']} = "
            f"{tool_result['value']} ({tool_result['matched_count']}/{tool_result['total_rows']})."
        )
    if tool == "count_by":
        parts = [f"{k}: {v}" for k, v in tool_result["counts"].items()]
        return f"Counts by {tool_result['column']}: " + ", ".join(parts[:8])
    if tool == "summary_stats":
        return (
            f"Summary for {tool_result['column']}: mean {tool_result['mean']}, "
            f"median {tool_result['median']}, min {tool_result['min']}, max {tool_result['max']}."
        )
    if tool == "histogram":
        return f"Generated histogram for {tool_result['column']}."
    return "I cannot compute that"


def answer_question(question: str) -> dict[str, Any]:
    """Run agent against a user question and return response metadata."""
    plan = _plan_tool_call(question)
    tool_name = str(plan.get("tool", "none"))
    args = dict(plan.get("args", {}))
    visualization_base64: str | None = None
    if tool_name == "none":
        return {
            "response": "I cannot compute that",
            "tool_used": None,
            "visualization_base64": None,
        }

    if tool_name not in _TOOL_REGISTRY:
        return {
            "response": "I cannot compute that",
            "tool_used": None,
            "visualization_base64": None,
        }

    try:
        tool_result = _TOOL_REGISTRY[tool_name](**args)
    except Exception:
        return {
            "response": "I cannot compute that",
            "tool_used": None,
            "visualization_base64": None,
        }

    if tool_name == "histogram":
        visualization_base64 = str(tool_result.get("image_base64", ""))

    text = _format_fallback(tool_result)

    return {
        "response": text,
        "tool_used": tool_name,
        "visualization_base64": visualization_base64,
    }

