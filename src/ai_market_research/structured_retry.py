from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ai_market_research.error_policy import is_retryable_error
from ai_market_research.exceptions import StructuredOutputError
from ai_market_research.models import ResearchOutcome
from ai_market_research.parser import StructuredOutputParser
from ai_market_research.parser_protocol import OutputParser


@dataclass(frozen=True)
class StructuredRetryResult:
    outcome: ResearchOutcome
    attempts: int
    last_error_code: str | None = None
    error_codes: tuple[str, ...] = ()


class StructuredOutputRetry:
    def __init__(
        self,
        max_attempts: int = 3,
        parser: OutputParser | None = None,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts debe ser >= 1")

        self.max_attempts = max_attempts
        self.parser = parser if parser is not None else StructuredOutputParser()
        self.last_attempts = 0

    async def run_with_metrics(
        self,
        generate: Callable[[], Awaitable[str]],
    ) -> StructuredRetryResult:
        last_error_code: str | None = None
        error_codes: list[str] = []

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await generate()
                outcome = self.parser.parse(response)

                return StructuredRetryResult(
                    outcome=outcome,
                    attempts=attempt,
                    last_error_code=last_error_code,
                    error_codes=tuple(error_codes),
                )

            except StructuredOutputError as exc:
                last_error_code = exc.error_code
                error_codes.append(exc.error_code)

                if not is_retryable_error(exc.error_code):
                    raise

                if attempt == self.max_attempts:
                    raise

            except TimeoutError:
                last_error_code = "timeout"
                error_codes.append("timeout")

                if attempt == self.max_attempts:
                    raise

        raise RuntimeError("Estado inesperado")

    async def run(
        self,
        generate: Callable[[], Awaitable[str]],
    ) -> ResearchOutcome:
        self.last_attempts = 0

        async def tracked_generate() -> str:
            self.last_attempts += 1
            return await generate()

        result = await self.run_with_metrics(tracked_generate)

        return result.outcome
