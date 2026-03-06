from __future__ import annotations

from src.cta.engine import CTAInterviewEngine
from src.cta.regression import default_regression_scenarios, run_regression_scenario


def test_default_regression_scenarios_pass() -> None:
    for scenario in default_regression_scenarios():
        engine = CTAInterviewEngine(template_seed=11)
        result = run_regression_scenario(engine, scenario)
        assert result.passed is True


def test_regression_is_deterministic_with_same_seed() -> None:
    scenario = default_regression_scenarios()[0]
    engine_a = CTAInterviewEngine(template_seed=31)
    engine_b = CTAInterviewEngine(template_seed=31)

    result_a = run_regression_scenario(engine_a, scenario)
    result_b = run_regression_scenario(engine_b, scenario)

    assert result_a.actual_question_types == result_b.actual_question_types
    assert result_a.actual_statuses == result_b.actual_statuses

