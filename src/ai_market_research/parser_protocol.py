from typing import Protocol

from ai_market_research.models import ResearchOutcome


class OutputParser(Protocol):
    def parse(self, response: str) -> ResearchOutcome: ...
