from dataclasses import FrozenInstanceError, replace

import pytest

from ai_market_research.config import RetryConfig


@pytest.mark.parametrize("max_attempts", [0, -1, -5])
def test_invalid_max_attempts(max_attempts):
    with pytest.raises(ValueError):
        RetryConfig(max_attempts=max_attempts)


@pytest.mark.parametrize("timeout", [0, -0.1, -5])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        RetryConfig(timeout=timeout)


@pytest.mark.parametrize("jitter", [-0.1, -1, -5])
def test_invalid_jitter(jitter):
    with pytest.raises(ValueError):
        RetryConfig(jitter=jitter)


def test_valid_boundary_values():
    config = RetryConfig(max_attempts=1, timeout=0.001, jitter=0)

    assert config.max_attempts == 1
    assert config.timeout == 0.001
    assert config.jitter == 0


def test_config_is_inmutable():
    config = RetryConfig()

    with pytest.raises(FrozenInstanceError):
        config.jitter = 1  # pyright: ignore[reportAttributeAccessIssue]


def test_config_replace():
    original = RetryConfig()
    modified = replace(original, max_attempts=5)

    assert original.max_attempts == 3
    assert modified.max_attempts == 5
    assert original is not modified
