from collections.abc import Awaitable, Callable

from ai_market_research.error_policy import is_retryable_error
from ai_market_research.exceptions import StructuredOutputError
from ai_market_research.models import ResearchOutcome
from ai_market_research.parser import StructuredOutputParser


class StructuredOutputRetry:
    def __init__(self, max_attempts: int = 3):
        if max_attempts < 1:
            raise ValueError("max_attempts debe ser >= 1")

        self.max_attempts = max_attempts
        self.parser = StructuredOutputParser()

    async def run(
        self,
        generate: Callable[[], Awaitable[str]],
    ) -> ResearchOutcome:
        for attempt in range(1, self.max_attempts + 1):
            response = await generate()

            try:
                return self.parser.parse(response)

            except StructuredOutputError as exc:
                if not is_retryable_error(exc.error_code):
                    raise

                if attempt == self.max_attempts:
                    raise

        raise RuntimeError("Estado inesperado")
