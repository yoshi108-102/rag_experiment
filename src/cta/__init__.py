"""CTA Sprint 1 package."""

from src.core.env import initialize_environment
from src.cta.engine import CTAInterviewEngine
from src.cta.models import SubjectPlan


def create_default_cta_engine() -> CTAInterviewEngine:
    """Build a default CTA engine after loading `.env` variables."""
    initialize_environment()
    return CTAInterviewEngine()


__all__ = ["CTAInterviewEngine", "SubjectPlan", "create_default_cta_engine"]
