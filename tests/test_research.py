import asyncio

import httpx2 as httpx
import pytest
from openai import RateLimitError

import ai_market_research.research as research_module
from ai_market_research.config import RetryConfig
from ai_market_research.research import research_with_retry


@pytest.mark.asyncio
async def test_research_success():
    attempts = 0

    async def fake_success_counted(market: str) -> str:
        nonlocal attempts
        attempts += 1
        return f"Investigacion completada: {market}"

    class FakeSuccessProvider:
        async def research(self, market: str) -> str:
            return await fake_success_counted(market)

    provider = FakeSuccessProvider()

    result = await research_with_retry(
        "Agricultura",
        config=RetryConfig(max_attempts=3, timeout=3),
        provider=provider,
    )

    assert result == "Investigacion completada: Agricultura"
    assert attempts == 1


@pytest.fixture
def no_sleep(monkeypatch):
    delays = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(research_module.asyncio, "sleep", fake_sleep)

    return delays


async def fake_research_market(market: str) -> str:
    return f"Investigacion completada: {market}"


async def fake_timeout(market: str) -> str:
    raise TimeoutError("API no disponible")


@pytest.mark.asyncio
async def test_research_retries_exhausted(monkeypatch, no_sleep):
    attempts = 0

    async def fake_timeout_counted(market: str) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("API no disponible")

    class FakeTimeoutProvider:
        async def research(self, market: str) -> str:
            return await fake_timeout_counted(market)

    provider = FakeTimeoutProvider()

    monkeypatch.setattr(
        research_module.random,
        "uniform",
        lambda a, b: 0.25,
    )

    with pytest.raises(TimeoutError):
        await research_with_retry(
            "Agricultura",
            config=RetryConfig(max_attempts=3, timeout=1),
            provider=provider,
        )

    assert attempts == 3
    assert no_sleep == [1.25, 2.25]


@pytest.mark.asyncio
async def test_research_recovers_after_timeout(monkeypatch, no_sleep):
    attempts = 0

    async def fake_unstable_api(market: str) -> str:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise TimeoutError("Error temporal")

        return f"Investigacion completada: {market}"

    class FakeUnstableProvider:
        async def research(self, market: str) -> str:
            return await fake_unstable_api(market)

    provider = FakeUnstableProvider()

    monkeypatch.setattr(
        research_module.random,
        "uniform",
        lambda a, b: 0.25,
    )

    result = await research_with_retry(
        "Agricultura",
        config=RetryConfig(max_attempts=3, timeout=1),
        provider=provider,
    )

    assert result == "Investigacion completada: Agricultura"
    assert attempts == 2
    assert no_sleep == [1.25]


@pytest.mark.asyncio
async def test_permanent_error_no_retry():
    attempts = 0

    async def fake_invalid_request(market: str) -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("Parámetros inválidos")

    class FakeInvalidProvider:
        async def research(self, market: str) -> str:
            return await fake_invalid_request(market)

    provider = FakeInvalidProvider()

    with pytest.raises(ValueError):
        await research_with_retry(
            "Agricultura",
            config=RetryConfig(max_attempts=3, timeout=1),
            provider=provider,
        )

    assert attempts == 1


@pytest.mark.asyncio
async def test_research_without_jitter(no_sleep):
    async def fake_research(market: str) -> str:
        raise TimeoutError("API timeout")

    class FakeTimeoutProvider:
        async def research(self, market: str) -> str:
            return await fake_research(market)

    provider = FakeTimeoutProvider()

    with pytest.raises(TimeoutError):
        await research_module.research_with_retry(
            "Agricultura",
            config=RetryConfig(max_attempts=3, jitter=0),
            provider=provider,
        )

    assert no_sleep == [1, 2]


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    semaphore = asyncio.Semaphore(2)

    active = 0
    max_active = 0

    async def fake_task():
        nonlocal active, max_active

        async with semaphore:
            active += 1
            max_active = max(max_active, active)

            await asyncio.sleep(0.01)

            active -= 1

    await asyncio.gather(*(fake_task() for _ in range(5)))

    assert max_active == 2
    assert active == 0


@pytest.mark.asyncio
async def test_insufficient_quota_no_retry():
    attempts = 0

    class FakeProvider:
        async def research(self, market: str) -> str:
            nonlocal attempts
            attempts += 1

            raise RateLimitError(
                "You have no credits remaining",
                response=httpx.Response(
                    429,
                    request=httpx.Request(
                        "POST",
                        "https://api.openai.com/v1/responses",
                    ),
                ),
                body={"error": {"code": "insufficient_quota"}},
            )

    with pytest.raises(RateLimitError):
        await research_with_retry(
            "Agricultura",
            config=RetryConfig(max_attempts=3, timeout=1),
            provider=FakeProvider(),
        )

    assert attempts == 1
