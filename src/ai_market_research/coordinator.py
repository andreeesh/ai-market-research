import asyncio

from ai_market_research.config import RetryConfig
from ai_market_research.models import ResearchResult
from ai_market_research.providers import ResearchProvider
from ai_market_research.research import research_with_retry


class ResearchCoordinator:
    def __init__(self, max_concurrency: int = 2):
        if max_concurrency < 1:
            raise ValueError("max_concurrency should be >= 1")

        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        market: str,
        provider: ResearchProvider,
    ) -> str:
        async with self.semaphore:
            return await research_with_retry(
                market,
                config=RetryConfig(timeout=3),
                provider=provider,
            )

    async def run_safe(
        self,
        market: str,
        provider: ResearchProvider,
    ) -> ResearchResult:
        try:
            content = await self.run(market, provider)

            return ResearchResult(
                market=market,
                success=True,
                content=content,
            )

        except (ValueError, TimeoutError) as exc:
            return ResearchResult(
                market=market,
                success=False,
                error=str(exc),
            )
