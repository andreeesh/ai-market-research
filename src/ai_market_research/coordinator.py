import asyncio

from ai_market_research.config import RetryConfig
from ai_market_research.models import ResearchResult
from ai_market_research.providers import ResearchProvider
from ai_market_research.research import research_with_retry
from ai_market_research.structured_retry import (
    StructuredOutputRetry,
    StructuredRetryResult,
)


class ResearchCoordinator:
    def __init__(self, max_concurrency: int = 2):
        if max_concurrency < 1:
            raise ValueError("max_concurrency should be >= 1")

        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        market: str,
        provider: ResearchProvider,
        config: RetryConfig | None = None,
    ) -> str:
        if config is None:
            config = RetryConfig(timeout=3)

        async with self.semaphore:
            return await research_with_retry(
                market,
                config=config,
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

    async def run_structured(
        self,
        market: str,
        provider: ResearchProvider,
        config: RetryConfig | None = None,
    ) -> StructuredRetryResult:
        if config is None:
            config = RetryConfig()

        async with self.semaphore:
            retry = StructuredOutputRetry(max_attempts=config.max_attempts)

            async def generate() -> str:
                return await asyncio.wait_for(
                    provider.research(market),
                    timeout=config.timeout,
                )

            return await retry.run_with_metrics(generate)
