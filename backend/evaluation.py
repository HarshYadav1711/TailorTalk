"""Simple evaluation script for Titanic chat agent behavior."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from backend.agent import answer_question
except ImportError:
    from agent import answer_question


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_tool: str
    expected_keywords: tuple[str, ...]


TEST_CASES: list[EvalCase] = [
    EvalCase("What is the average age?", "average", ("average", "age")),
    EvalCase("Show me a histogram of fare.", "histogram", ("histogram", "fare")),
    EvalCase("Count passengers by sex.", "count_by", ("sex",)),
    EvalCase("What percentage of passengers survived?", "calculate_percentage", ("survived", "%")),
    EvalCase("Give summary stats for fare.", "summary_stats", ("mean", "median", "fare")),
    EvalCase("How many passengers are in each class?", "count_by", ("class", "pclass")),
    EvalCase("What is the average fare?", "average", ("average", "fare")),
    EvalCase("Create a histogram for age with 15 bins.", "histogram", ("histogram", "age")),
    EvalCase("What percentage of passengers are female?", "calculate_percentage", ("female", "%")),
    EvalCase("Give summary statistics for age.", "summary_stats", ("age", "median")),
]


def _contains_all_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def run_evaluation() -> None:
    """Run all fixed test queries and print an accuracy report."""
    total = len(TEST_CASES)
    passed = 0

    print("Titanic Chat Agent Evaluation")
    print("=" * 40)

    for index, case in enumerate(TEST_CASES, start=1):
        result = answer_question(case.query)
        response_text = result["response"]
        tool_used = result["tool_used"] or ""

        tool_ok = tool_used == case.expected_tool
        text_ok = _contains_all_keywords(response_text, case.expected_keywords)
        case_ok = tool_ok and text_ok
        passed += int(case_ok)

        status = "PASS" if case_ok else "FAIL"
        print(f"[{status}] Case {index}: {case.query}")
        print(f"  expected tool: {case.expected_tool} | actual tool: {tool_used or 'none'}")
        print(f"  response: {response_text}")

    accuracy = (passed / total) * 100
    print("=" * 40)
    print(f"Passed: {passed}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    run_evaluation()

