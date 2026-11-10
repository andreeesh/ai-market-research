import pytest

from ai_market_research.error_policy import is_retryable_error


@pytest.mark.parametrize(
    "error_code",
    [
        "json_invalid",
        "missing",
        "union_tag_invalid",
        "union_tag_not_found",
    ],
)
def test_retryable_validation_errors(error_code):
    assert is_retryable_error(error_code) is True


@pytest.mark.parametrize(
    "error_code",
    [
        "permission_denied",
        "authentication_failed",
        "rate_limit_exceeded",
        "unknown_error",
    ],
)
def test_non_retryable_errors(error_code):
    assert is_retryable_error(error_code) is False
