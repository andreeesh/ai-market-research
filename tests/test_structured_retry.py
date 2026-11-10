import pytest

from ai_market_research.exceptions import StructuredOutputError
from ai_market_research.models import ResearchSuccess
from ai_market_research.structured_retry import StructuredOutputRetry


@pytest.mark.asyncio
async def test_retry_recovers_after_invalid_json():
    attempts = 0

    async def fake_generate() -> str:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return '{"success": true'

        return """
        {
            "market": "Agricultura",
            "success": true,
            "content": "Oportunidad detectada"
        }
        """

    retry = StructuredOutputRetry(max_attempts=3)

    result = await retry.run(fake_generate)

    assert isinstance(result, ResearchSuccess)
    assert result.content == "Oportunidad detectada"
    assert attempts == 2


@pytest.mark.asyncio
async def test_retry_exhausts_attempts():
    attempts = 0

    async def fake_generate() -> str:
        nonlocal attempts
        attempts += 1

        return '{"success": true'

    retry = StructuredOutputRetry(max_attempts=3)

    with pytest.raises(StructuredOutputError) as exc_info:
        await retry.run(fake_generate)

    assert attempts == 3
    assert exc_info.value.error_code == "json_invalid"


@pytest.mark.asyncio
async def test_retry_stops_on_non_retryable_error(monkeypatch):
    attempts = 0

    async def fake_generate() -> str:
        nonlocal attempts
        attempts += 1
        return '{"success": true'

    monkeypatch.setattr(
        "ai_market_research.structured_retry.is_retryable_error",
        lambda error_code: False,
    )

    retry = StructuredOutputRetry(max_attempts=3)

    with pytest.raises(StructuredOutputError):
        await retry.run(fake_generate)

    assert attempts == 1
