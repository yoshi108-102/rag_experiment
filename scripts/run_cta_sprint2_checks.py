"""Run Sprint 2 checks: regression + artifact export."""

from __future__ import annotations

from pathlib import Path

from src.cta import (
    default_regression_scenarios,
    export_session_artifacts,
    run_regression_scenario,
)
from src.core.env import initialize_environment
from src.cta.engine import CTAInterviewEngine
from src.cta.metrics import summarize_turn_latency


def main() -> None:
    initialize_environment()
    scenarios = default_regression_scenarios()
    all_passed = True

    for scenario in scenarios:
        engine = CTAInterviewEngine(template_seed=11)
        result = run_regression_scenario(engine, scenario)
        print(f"[Scenario] {result.scenario_name}: {'PASS' if result.passed else 'FAIL'}")
        print(f"  expected types: {result.expected_question_types}")
        print(f"  actual types:   {result.actual_question_types}")
        print(f"  expected status:{result.expected_statuses}")
        print(f"  actual status:  {result.actual_statuses}")
        all_passed = all_passed and result.passed

        session_ids = engine.store.list_session_ids()
        if not session_ids:
            continue
        session_id = session_ids[0]
        summary = summarize_turn_latency(engine.store.list_turns(session_id))
        print(
            f"  latency: count={summary.count}, p50={summary.p50_ms}ms, "
            f"p95={summary.p95_ms}ms, within_target={summary.within_target}"
        )
        artifacts = export_session_artifacts(
            engine.store,
            session_id,
            Path("logs") / "cta_sprint2",
        )
        print(f"  artifacts: {artifacts}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
