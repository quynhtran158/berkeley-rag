"""
evaluate.py — Evaluation framework for the Unofficial Berkeley CS Guide.

Runs 5 test questions against the RAG system and produces a structured
evaluation report comparing system responses to expected correct answers.
"""

from query import ask

EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the difference between taking CS61B with Hilfinger vs. Hug?",
        "expected_answer": (
            "Hug's CS61B is more scaffolded with clearer project specs and slightly higher exam averages "
            "(60-70%). Hilfinger's version is harder with more cryptic specs, lower exam averages (~50-60%), "
            "and less guidance. Most students recommend Hug unless they want maximum rigor and already have "
            "programming experience."
        ),
    },
    {
        "id": 2,
        "question": "What are the main topics I should study for the CS186 database exam?",
        "expected_answer": (
            "SQL (complex queries, joins, subqueries), relational algebra, B+ trees (insertions, deletions, "
            "splits by hand), buffer management policies (LRU, MRU, CLOCK), two-phase locking and deadlock, "
            "and the ARIES recovery protocol. ARIES appears on every CS186 exam."
        ),
    },
    {
        "id": 3,
        "question": "Which dining hall has the shortest wait times at UC Berkeley?",
        "expected_answer": (
            "Clark Kerr and Foothill have the shortest wait times because they are underused. Crossroads has "
            "the longest waits, especially 12pm-1pm during the lunch rush (10-15 minutes even off-peak). "
            "Cafe 3 is dinner-only so avoids the lunch rush entirely."
        ),
    },
    {
        "id": 4,
        "question": "When should I start applying for Amazon software engineering internships for the following summer?",
        "expected_answer": (
            "Amazon opens applications in September for the following summer. Online assessments (OAs) go out "
            "October-November, and offers are extended by December. You should apply immediately when "
            "applications open in September."
        ),
    },
    {
        "id": 5,
        "question": "How long is the typical wait for a first appointment at CAPS (campus mental health)?",
        "expected_answer": (
            "Typically 2-4 weeks for the first appointment. For same-day slots, call at 8am when they're "
            "released. For mental health emergencies, call the Tang Center crisis line at 510-642-9494 "
            "rather than waiting for a CAPS appointment."
        ),
    },
]

ACCURACY_SCALE = ["accurate", "partially accurate", "inaccurate"]


def evaluate_response(question_id: int, question: str, expected: str,
                       actual: str, sources: list[str]) -> dict:
    """
    Manually evaluate one question/response pair.
    For automated running, we print results for human review.
    """
    return {
        "id": question_id,
        "question": question,
        "expected_answer": expected,
        "system_response": actual,
        "sources_retrieved": sources,
    }


def run_evaluation() -> list[dict]:
    """Run all 5 evaluation questions and collect results."""
    results = []
    print("=" * 70)
    print("EVALUATION REPORT — Unofficial Berkeley CS Guide")
    print("=" * 70)

    for item in EVALUATION_QUESTIONS:
        print(f"\n[Question {item['id']}] {item['question']}")
        print("-" * 60)

        result = ask(item["question"])
        record = evaluate_response(
            question_id=item["id"],
            question=item["question"],
            expected=item["expected_answer"],
            actual=result["answer"],
            sources=result["sources"],
        )

        print(f"Expected Answer:\n  {item['expected_answer']}\n")
        print(f"System Response:\n  {result['answer']}\n")
        print(f"Sources Retrieved: {result['sources']}")
        print(f"\nTop Retrieved Chunk (distance: {result['retrieved_chunks'][0]['distance']:.4f}):")
        print(f"  [{result['retrieved_chunks'][0]['source']}] "
              f"{result['retrieved_chunks'][0]['text'][:200]}...")

        results.append(record)
        print("\n" + "=" * 70)

    return results


def print_summary(results: list[dict]) -> None:
    """Print a structured markdown-ready summary for inclusion in README."""
    print("\n\n" + "=" * 70)
    print("EVALUATION SUMMARY (Markdown Format)")
    print("=" * 70)
    for r in results:
        print(f"\n### Question {r['id']}: {r['question']}")
        print(f"**Expected:** {r['expected_answer'][:200]}...")
        print(f"**System Response:** {r['system_response'][:300]}...")
        print(f"**Sources:** {', '.join(r['sources_retrieved'])}")
        print(f"**Accuracy Judgment:** [Fill in: accurate / partially accurate / inaccurate]")


if __name__ == "__main__":
    results = run_evaluation()
    print_summary(results)
