import asyncio

import pytest

from ai_market_research.exceptions import StructuredOutputError
from ai_market_research.models import ResearchOutcome, ResearchSuccess
from ai_market_research.parser import StructuredOutputParser
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


@pytest.mark.asyncio
async def test_retry_success_on_first_attempt():
    attempts = 0

    async def fake_generate() -> str:
        nonlocal attempts
        attempts += 1

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
    assert attempts == 1

@pytest.mark.asyncio
async def test_retry_metrics_empty_on_first_success():
    attempts = 0

    async def fake_generate() -> str:
        nonlocal attempts
        attempts += 1

        return """
        {
            "market": "Agricultura",
            "success": true,
            "content": "Oportunidad detectada"
        }
        """

    retry = StructuredOutputRetry(max_attempts=3)

    result = await retry.run_with_metrics(fake_generate)

    assert isinstance(result.outcome, ResearchSuccess)
    assert result.outcome.content == "Oportunidad detectada"
    assert result.attempts == 1
    assert result.last_error_code is None
    assert result.error_codes == ()
    assert attempts == 1


@pytest.mark.parametrize("max_attempts", [0, -1, -5])
def test_retry_rejects_invalid_max_attempts(max_attempts):
    with pytest.raises(ValueError, match="max_attempts debe ser"):
        StructuredOutputRetry(max_attempts=max_attempts)


@pytest.mark.asyncio
async def test_retry_with_single_attempt():
    attempts = 0

    async def fake_generate() -> str:
        nonlocal attempts
        attempts += 1
        return '{"success": true'

    retry = StructuredOutputRetry(max_attempts=1)

    with pytest.raises(StructuredOutputError):
        await retry.run(fake_generate)

    assert attempts == 1


def test_retry_accepts_custom_parser():
    parser = StructuredOutputParser()

    retry = StructuredOutputRetry(
        max_attempts=3,
        parser=parser,
    )

    assert retry.parser is parser


@pytest.mark.asyncio
async def test_retry_uses_injected_parser():
    calls = []

    expected = ResearchSuccess(
        market="Agricultura",
        content="Resultado simulado",
    )

    class FakeParser:
        def parse(self, response: str) -> ResearchOutcome:
            calls.append(response)
            return expected

    async def fake_generate() -> str:
        return "respuesta del proveedor"

    retry = StructuredOutputRetry(
        max_attempts=3,
        parser=FakeParser(),
    )

    result = await retry.run(fake_generate)

    assert result is expected
    assert calls == ["respuesta del proveedor"]


@pytest.mark.asyncio
async def test_retry_tracks_attempt_count():
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

    await retry.run(fake_generate)

    assert retry.last_attempts == 2


@pytest.mark.asyncio
async def test_retry_resets_attempt_count():
    attempts = 0

    valid_response = """
    {
        "market": "Agricultura",
        "success": true,
        "content": "Oportunidad detectada"
    }
    """

    async def generate_with_retry() -> str:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return '{"success": true'

        return valid_response

    async def generate_success() -> str:
        return valid_response

    retry = StructuredOutputRetry(max_attempts=3)

    await retry.run(generate_with_retry)
    assert retry.last_attempts == 2

    await retry.run(generate_success)
    assert retry.last_attempts == 1


@pytest.mark.asyncio
async def test_retry_returns_metrics():
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

    result = await retry.run_with_metrics(fake_generate)

    assert isinstance(result.outcome, ResearchSuccess)
    assert result.outcome.content == "Oportunidad detectada"
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_retry_tracks_attempts_on_failure():
    attempts = 0

    async def fake_generate() -> str:
        nonlocal attempts
        attempts += 1
        return '{"success": true'

    retry = StructuredOutputRetry(max_attempts=3)

    with pytest.raises(StructuredOutputError):
        await retry.run(fake_generate)

    assert attempts == 3
    assert retry.last_attempts == 3


@pytest.mark.asyncio
async def test_retry_metrics_are_independent():
    retry = StructuredOutputRetry(max_attempts=3)

    agriculture_attempts = 0

    async def generate_agriculture() -> str:
        nonlocal agriculture_attempts
        agriculture_attempts += 1

        await asyncio.sleep(0)

        if agriculture_attempts == 1:
            return '{"success": true'

        return """
        {
            "market": "Agricultura",
            "success": true,
            "content": "Resultado agrícola"
        }
        """

    async def generate_software() -> str:
        await asyncio.sleep(0)

        return """
        {
            "market": "Software",
            "success": true,
            "content": "Resultado tecnológico"
        }
        """

    agriculture, software = await asyncio.gather(
        retry.run_with_metrics(generate_agriculture),
        retry.run_with_metrics(generate_software),
    )

    assert agriculture.attempts == 2
    assert software.attempts == 1

    assert agriculture.outcome.market == "Agricultura"
    assert software.outcome.market == "Software"


@pytest.mark.asyncio
async def test_retry_tracks_last_validation_error():
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

    result = await retry.run_with_metrics(fake_generate)

    assert result.attempts == 2
    assert result.last_error_code == "json_invalid"
    assert result.error_codes == ("json_invalid",)

@pytest.mark.asyncio
async def test_retry_tracks_multiple_validation_errors():
    attempts = 0

    async def fake_generate() -> str:
        nonlocal attempts
        attempts += 1

        if attempts <= 2:
            return '{"success": true'

        return """
        {
            "market": "Agricultura",
            "success": true,
            "content": "Oportunidad detectada"
        }
        """

    retry = StructuredOutputRetry(max_attempts=3)

    result = await retry.run_with_metrics(fake_generate)

    assert result.attempts == 3
    assert result.last_error_code == "json_invalid"
    assert result.error_codes == ("json_invalid", "json_invalid")
