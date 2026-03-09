"""Deterministic regression scenarios for CTA flows."""

from __future__ import annotations

from dataclasses import dataclass

from src.cta.engine import CTAInterviewEngine
from src.cta.models import CTAResponse, SubjectPlan


@dataclass(frozen=True)
class RegressionStep:
    user_input: str
    expected_question_type: str
    expected_status: str


@dataclass(frozen=True)
class RegressionScenario:
    name: str
    subjects: list[SubjectPlan]
    steps: list[RegressionStep]


@dataclass(frozen=True)
class RegressionResult:
    scenario_name: str
    passed: bool
    expected_question_types: list[str]
    actual_question_types: list[str]
    expected_statuses: list[str]
    actual_statuses: list[str]


def default_regression_scenarios() -> list[RegressionScenario]:
    return [
        RegressionScenario(
            name="topic-transition-and-finish",
            subjects=[SubjectPlan(name="業務判断", topics=["状況把握", "判断基準"])],
            steps=[
                RegressionStep(
                    user_input="状況を確認して情報を整理しました。",
                    expected_question_type="CDM1",
                    expected_status="ACTIVE",
                ),
                RegressionStep(
                    user_input="いいえ、特にありません。",
                    expected_question_type="STD7",
                    expected_status="ACTIVE",
                ),
                RegressionStep(
                    user_input="終了します。",
                    expected_question_type="STD11",
                    expected_status="FINISHED",
                ),
            ],
        ),
        RegressionScenario(
            name="rich-answer-then-topic-limit-finish",
            subjects=[SubjectPlan(name="振り返り", topics=["対応手順"])],
            steps=[
                RegressionStep(
                    user_input="状況を確認しました。判断基準も整理しました。",
                    expected_question_type="CDM2",
                    expected_status="ACTIVE",
                ),
                RegressionStep(
                    user_input="その判断で問題ないでしょうか？",
                    expected_question_type="STD11",
                    expected_status="FINISHED",
                ),
            ],
        ),
    ]


def run_regression_scenario(
    engine: CTAInterviewEngine,
    scenario: RegressionScenario,
) -> RegressionResult:
    start = engine.start_session(subjects=scenario.subjects, generation_mode="TEMPLATE_RANDOM")
    actual_question_types: list[str] = []
    actual_statuses: list[str] = []

    for step in scenario.steps:
        response: CTAResponse = engine.handle_user_input(start.session_id, step.user_input)
        actual_question_types.append(response.question_type)
        actual_statuses.append(response.status)

    expected_question_types = [step.expected_question_type for step in scenario.steps]
    expected_statuses = [step.expected_status for step in scenario.steps]

    passed = actual_question_types == expected_question_types and actual_statuses == expected_statuses
    return RegressionResult(
        scenario_name=scenario.name,
        passed=passed,
        expected_question_types=expected_question_types,
        actual_question_types=actual_question_types,
        expected_statuses=expected_statuses,
        actual_statuses=actual_statuses,
    )
