import asyncio

import pytest

from ai_market_research.config import RetryConfig
from ai_market_research.coordinator import ResearchCoordinator
from ai_market_research.models import ResearchSuccess


@pytest.mark.parametrize(
    "max_concurrency",
    [0, -1, -10],
)
def test_coordinator_rejects_invalid_concurrency(max_concurrency):
    with pytest.raises(ValueError):
        ResearchCoordinator(max_concurrency=max_concurrency)


@pytest.mark.asyncio
async def test_coordinator_limits_concurrency():
    coordinator = ResearchCoordinator(max_concurrency=2)

    active = 0
    max_active = 0

    class FakeProvider:
        async def research(self, market: str) -> str:
            nonlocal active, max_active

            active += 1
            max_active = max(max_active, active)

            try:
                await asyncio.sleep(0.01)
                return f"Investigacion: {market}"
            finally:
                active -= 1

    provider = FakeProvider()

    results = await asyncio.gather(
        *(coordinator.run(f"Mercado {i}", provider) for i in range(5))
    )

    assert max_active == 2
    assert active == 0
    assert len(results) == 5


@pytest.mark.asyncio
async def test_coordinator_handles_partial_failures():
    coordinator = ResearchCoordinator(max_concurrency=3)

    class FakeProvider:
        async def research(self, market: str) -> str:
            if market == "Software":
                raise ValueError("Mercado no disponible")

            return f"Investigacion: {market}"

    provider = FakeProvider()

    markets = ["Agricultura", "Software", "Energia"]

    results = await asyncio.gather(
        *(coordinator.run(market, provider) for market in markets),
        return_exceptions=True,
    )

    assert results[0] == "Investigacion: Agricultura"
    assert isinstance(results[1], ValueError)
    assert results[2] == "Investigacion: Energia"


@pytest.mark.asyncio
async def test_coordinator_run_safe_handles_failure():
    coordinator = ResearchCoordinator()

    class FailingProvider:
        async def research(self, market: str) -> str:
            raise ValueError("Proveedor no disponible")

    result = await coordinator.run_safe(
        "Software",
        FailingProvider(),
    )

    assert result.market == "Software"
    assert result.success is False
    assert result.content is None
    assert result.error == "Proveedor no disponible"


@pytest.mark.asyncio
async def test_coordinator_run_safe_success():
    coordinator = ResearchCoordinator()

    class SuccessfulProvider:
        async def research(self, market: str) -> str:
            return f"Reporte de {market}"

    result = await coordinator.run_safe(
        "Agricultura",
        SuccessfulProvider(),
    )

    assert result.market == "Agricultura"
    assert result.success is True
    assert result.content == "Reporte de Agricultura"
    assert result.error is None


@pytest.mark.asyncio
async def test_coordinator_multiple_safe_results():
    coordinator = ResearchCoordinator(max_concurrency=2)

    class FakeProvider:
        async def research(self, market: str) -> str:
            if market == "Software":
                raise ValueError("Mercado no disponible")

            return f"Reporte de {market}"

    provider = FakeProvider()
    markets = ["Agricultura", "Software", "Energia"]

    results = await asyncio.gather(
        *(coordinator.run_safe(market, provider) for market in markets)
    )

    assert len(results) == 3

    assert results[0].success is True
    assert results[0].content == "Reporte de Agricultura"

    assert results[1].success is False
    assert results[1].error == "Mercado no disponible"

    assert results[2].success is True
    assert results[2].content == "Reporte de Energia"


@pytest.mark.asyncio
async def test_coordinator_run_structured_success():
    coordinator = ResearchCoordinator(max_concurrency=2)

    class FakeProvider:
        async def research(self, market: str) -> str:
            return f"""
            {{
                "market": "{market}",
                "success": true,
                "content": "Oportunidad detectada"
            }}
            """

    result = await coordinator.run_structured(
        "Agricultura",
        FakeProvider(),
    )

    assert isinstance(result.outcome, ResearchSuccess)
    assert result.outcome.market == "Agricultura"
    assert result.attempts == 1
    assert result.error_codes == ()


@pytest.mark.asyncio
async def test_coordinator_run_structured_recovers():
    coordinator = ResearchCoordinator(max_concurrency=2)

    class FakeProvider:
        def __init__(self):
            self.attempts = 0

        async def research(self, market: str) -> str:
            self.attempts += 1

            if self.attempts == 1:
                return '{"success": true'

            return f"""
            {{
                "market": "{market}",
                "success": true,
                "content": "Oportunidad detectada"
            }}
            """

    result = await coordinator.run_structured(
        "Agricultura",
        FakeProvider(),
    )

    assert isinstance(result.outcome, ResearchSuccess)
    assert result.outcome.market == "Agricultura"
    assert result.attempts == 2
    assert result.last_error_code == "json_invalid"
    assert result.error_codes == ("json_invalid",)


@pytest.mark.asyncio
async def test_coordinator_structured_timeout():
    coordinator = ResearchCoordinator()

    class SlowProvider:
        async def research(self, market: str) -> str:
            await asyncio.sleep(1)
            return "{}"

    config = RetryConfig(
        max_attempts=1,
        timeout=0.01,
        jitter=0,
    )

    with pytest.raises(TimeoutError):
        await coordinator.run_structured(
            "Agricultura",
            SlowProvider(),
            config=config,
        )


@pytest.mark.asyncio
async def test_coordinator_structured_recovers_after_timeout():
    coordinator = ResearchCoordinator()

    class SlowProvider:
        def __init__(self):
            self.attempts = 0

        async def research(self, market: str) -> str:
            self.attempts += 1

            if self.attempts == 1:
                await asyncio.sleep(0.1)

            return f"""
            {{
                "market": "{market}",
                "success": true,
                "content": "Investigación completada"
            }}
            """

    config = RetryConfig(
        max_attempts=2,
        timeout=0.01,
        jitter=0,
    )

    provider = SlowProvider()

    result = await coordinator.run_structured(
        "Agricultura",
        provider,
        config=config,
    )

    assert isinstance(result.outcome, ResearchSuccess)
    assert result.attempts == 2
    assert provider.attempts == 2
    assert result.error_codes == ("timeout",)


@pytest.mark.asyncio
async def test_coordinator_structured_recovers_from_mixed_errors():
    coordinator = ResearchCoordinator()

    class SlowProvider:
        def __init__(self):
            self.attempts = 0

        async def research(self, market: str) -> str:
            self.attempts += 1

            if self.attempts == 1:
                await asyncio.sleep(0.1)

            if self.attempts == 2:
                return '{"success": true'

            return f"""
            {{
                "market": "{market}",
                "success": true,
                "content": "Investigación completada"
            }}
            """

    config = RetryConfig(
        max_attempts=3,
        timeout=0.01,
        jitter=0,
    )

    provider = SlowProvider()

    result = await coordinator.run_structured(
        "Agricultura",
        provider,
        config=config,
    )

    assert isinstance(result.outcome, ResearchSuccess)
    assert result.attempts == 3
    assert provider.attempts == 3
    assert result.last_error_code == "json_invalid"
    assert result.error_codes == ("timeout", "json_invalid")
