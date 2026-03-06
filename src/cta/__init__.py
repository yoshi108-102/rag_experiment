"""CTA Sprint 1 package."""

from src.core.env import initialize_environment
from src.cta.engine import CTAInterviewEngine
from src.cta.export import export_session_artifacts
from src.cta.metrics import summarize_turn_latency
from src.cta.models import SubjectPlan
from src.cta.regression import default_regression_scenarios, run_regression_scenario


def create_default_cta_engine() -> CTAInterviewEngine:
    """Build a default CTA engine after loading `.env` variables."""
    initialize_environment()
    return CTAInterviewEngine()


__all__ = [
    "CTAInterviewEngine",
    "SubjectPlan",
    "create_default_cta_engine",
    "export_session_artifacts",
    "summarize_turn_latency",
    "default_regression_scenarios",
    "run_regression_scenario",
]
