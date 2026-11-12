import inspect
from unittest.mock import AsyncMock

import pytest
from openai import AuthenticationError, RateLimitError

from ai_market_research.openai_provider import OpenAIResearchProvider


def test_openai_provider_has_async_research():
    client = AsyncMock()
    provider = OpenAIResearchProvider(client)
    assert inspect.iscoroutinefunction(provider.research)


@pytest.mark.asyncio
async def test_openai_provider_returns_research():
    client = AsyncMock()
    client.responses.create.return_value.output_text = "Investigación de Agricultura"
    provider = OpenAIResearchProvider(client)
    result = await provider.research("Agricultura")
    assert result == "Investigación de Agricultura"


@pytest.mark.asyncio
async def test_openai_provider_calls_api():
    client = AsyncMock()
    client.responses.create.return_value.output_text = "Análisis del mercado agrícola"
    provider = OpenAIResearchProvider(client)
    result = await provider.research("Agricultura")
    client.responses.create.assert_awaited_once()
    assert result == "Análisis del mercado agrícola"


@pytest.mark.asyncio
async def test_openai_provider_propagates_authentication_error():
    client = AsyncMock()
    error = AuthenticationError(
        "Invalid API key",
        response=__import__("httpx").Response(
            401,
            request=__import__("httpx").Request(
                "POST",
                "https://api.openai.com/v1/responses",
            ),
        ),
        body={"error": {"code": "invalid_api_key"}},
    )

    client.responses.create.side_effect = error
    provider = OpenAIResearchProvider(client)

    with pytest.raises(AuthenticationError):
        await provider.research("Agricultura")


@pytest.mark.asyncio
async def test_openai_provider_propagates_insufficient_quota():
    client = AsyncMock()

    error = RateLimitError(
        "You have no credits remaining",
        response=__import__("httpx").Response(
            429,
            request=__import__("httpx").Request(
                "POST",
                "https://api.openai.com/v1/responses",
            ),
        ),
        body={"error": {"code": "insufficient_quota"}},
    )

    client.responses.create.side_effect = error
    provider = OpenAIResearchProvider(client)

    with pytest.raises(RateLimitError):
        await provider.research("Agricultura")

    client.responses.create.assert_awaited_once()
