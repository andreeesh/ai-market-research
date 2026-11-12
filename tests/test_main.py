from unittest.mock import patch

from openai import RateLimitError

from ai_market_research.main import main


def test_main_handles_insufficient_quota(capsys):
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

    def fake_asyncio_run(coro):
        coro.close()
        raise error

    with patch(
        "ai_market_research.main.asyncio.run",
        side_effect=fake_asyncio_run,
    ):
        main()

    output = capsys.readouterr().out

    assert "créditos agotados" in output
    assert "Tiempo total:" in output
