import asyncio
import time

from openai import AsyncOpenAI, RateLimitError

from ai_market_research.config import RetryConfig
from ai_market_research.coordinator import ResearchCoordinator
from ai_market_research.openai_provider import OpenAIResearchProvider
from ai_market_research.providers import (
    MockResearchProvider,
    StaticResearchProvider,
)


async def run_research():
    coordinator = ResearchCoordinator(max_concurrency=2)
    client = AsyncOpenAI(max_retries=0)
    openai_provider = OpenAIResearchProvider(client)

    providers = [
        MockResearchProvider(),
        StaticResearchProvider(),
        openai_provider,
    ]

    results = await asyncio.gather(
        *(
            coordinator.run(
                "Agricultura",
                provider,
                config=RetryConfig(
                    timeout=60,
                    max_attempts=1,
                    jitter=0,
                ),
            )
            for provider in providers
        )
    )

    for result in results:
        print(result)


def main() -> None:
    start = time.perf_counter()

    try:
        asyncio.run(run_research())

    except RateLimitError as exc:
        error_code = exc.code

        if error_code is None and isinstance(exc.body, dict):
            error_data = exc.body.get("error", {})
            if isinstance(error_data, dict):
                error_code = error_data.get("code")

        if error_code == "insufficient_quota":
            print("OpenAI: créditos agotados. Revisá tu facturación.")
        else:
            print("OpenAI: límite de solicitudes alcanzado.")

    elapsed = time.perf_counter() - start
    print(f"Tiempo total: {elapsed:.2f} segundos")


if __name__ == "__main__":
    main()
