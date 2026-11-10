import asyncio
import random

from ai_market_research.config import RetryConfig
from ai_market_research.providers import MockResearchProvider, ResearchProvider


async def research_with_retry(
    market: str,
    config: RetryConfig | None = None,
    provider: ResearchProvider | None = None,
) -> str:
    if config is None:
        config = RetryConfig()
    if provider is None:
        provider = MockResearchProvider()

    for attempt in range(1, config.max_attempts + 1):
        print(f"Intento {attempt}/{config.max_attempts}: {market}")

        try:
            return await asyncio.wait_for(
                provider.research(market),
                timeout=config.timeout,
            )
        except TimeoutError:
            if attempt < config.max_attempts:
                base_delay = 2 ** (attempt - 1)
                delay = base_delay + random.uniform(0, config.jitter)
                print(f"Esperando {delay}s antes de reintentar")
                await asyncio.sleep(delay)

    raise TimeoutError(
        f"Investigacion de {market} fallida despues de {config.max_attempts} intentos"
    )
