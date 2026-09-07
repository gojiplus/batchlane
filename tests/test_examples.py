"""Exercise the documented planning API without making paid requests."""

import batchlane as bl


def test_plan_example():
    rows = [
        bl.BatchLine(
            "row-1",
            "groq/llama-3.3-70b-versatile",
            [{"role": "user", "content": "Classify: the product was great"}],
        ),
        bl.BatchLine(
            "row-2",
            "groq/llama-3.3-70b-versatile",
            [{"role": "user", "content": "Classify: it broke in a week"}],
        ),
    ]
    plan = bl.plan(rows)
    assert plan.n_chunks == 1
    assert [line.custom_id for line in plan.chunks[0]] == ["row-1", "row-2"]
    assert bl.capabilities_for("groq").window.allowed == ("24h", "7d")
