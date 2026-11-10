from pydantic import ValidationError

from ai_market_research.exceptions import StructuredOutputError
from ai_market_research.models import (
    ResearchOutcome,
    research_outcome_adapter,
)


class StructuredOutputParser:
    def parse(self, response: str) -> ResearchOutcome:
        try:
            return research_outcome_adapter.validate_json(response)
        except ValidationError as exc:
            error_code = exc.errors()[0]["type"]

            raise StructuredOutputError(
                "La respuesta no cumple el esquema esperado",
                error_code=error_code,
            ) from exc
