import asyncio
import time

from ai_market_research.config import RetryConfig
from ai_market_research.coordinator import ResearchCoordinator
from ai_market_research.providers import (
    MockResearchProvider,
    StaticResearchProvider,
)
from ai_market_research.research import research_with_retry


async def run_research():
    # markets = ["Agricultura", "Software", "Salud"]
    semaphore = asyncio.Semaphore(2)
    coordinator = ResearchCoordinator(max_concurrency=2)

    async def run_provider(provider):
        async with semaphore:
            return await research_with_retry(
                "Agricultura",
                config=RetryConfig(timeout=3),
                provider=provider,
            )

    providers = [
        MockResearchProvider(),
        StaticResearchProvider(),
    ]

    results = await asyncio.gather(
        *(coordinator.run("Agricultura", provider) for provider in providers)
    )

    # results = await asyncio.gather(
    #     *(
    #         research_with_retry(
    #             "Agricultura",
    #             config=RetryConfig(timeout=3),
    #             provider=provider,
    #         )
    #         for provider in providers
    #     ),
    #     return_exceptions=True,
    # )

    for result in results:
        print(result)

    # for market, result in zip(markets, results):
    #     if isinstance(result, TimeoutError):
    #         print(f"TIMEOUT [{market}]: tiempo máximo excedido")
    #     elif isinstance(result, BaseException):
    #         print(f"ERROR: [{market}]: {result}")
    #     else:
    #         print(f"OK [{market}]: {result}")


def main() -> None:
    start = time.perf_counter()
    asyncio.run(run_research())
    elapsed = time.perf_counter() - start
    print(f"Tiempo total: {elapsed:.2f} segundos")


if __name__ == "__main__":
    main()
