from ai_market_research.models import ValidateOpportunity


def validate_research(data: dict) -> ValidateOpportunity:
    return ValidateOpportunity.model_validate(data)
