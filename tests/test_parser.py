import pytest
from pydantic import ValidationError

from ai_market_research.error_policy import is_retryable_error
from ai_market_research.exceptions import StructuredOutputError
from ai_market_research.models import ResearchFailure, ResearchSuccess
from ai_market_research.parser import StructuredOutputParser


def test_parser_success():
    parser = StructuredOutputParser()

    response = """
    {
        "market": "Agricultura",
        "success": true,
        "content": "Oportunidad detectada"
    }
    """

    result = parser.parse(response)

    assert isinstance(result, ResearchSuccess)
    assert result.market == "Agricultura"
    assert result.content == "Oportunidad detectada"


def test_parser_failure():
    parser = StructuredOutputParser()

    response = """
    {
        "market": "Software",
        "success": false,
        "error": "Proveedor no disponible"
    }
    """

    result = parser.parse(response)

    assert isinstance(result, ResearchFailure)
    assert result.market == "Software"
    assert result.success is False
    assert result.error == "Proveedor no disponible"


def test_parser_rejects_malformed_json():
    parser = StructuredOutputParser()

    response = """
    {
        "market": "Agricultura",
        "success": true,
    """

    with pytest.raises(StructuredOutputError) as exc_info:
        parser.parse(response)

    original_error = exc_info.value.__cause__

    assert isinstance(original_error, ValidationError)

    errors = original_error.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "json_invalid"


def test_parser_rejects_missing_required_field():
    parser = StructuredOutputParser()

    response = """
    {
        "market": "Agricultura",
        "success": true
    }
    """

    with pytest.raises(StructuredOutputError) as exc_info:
        parser.parse(response)

    original_error = exc_info.value.__cause__

    assert isinstance(original_error, ValidationError)

    errors = original_error.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "missing"


def test_parser_raises_domain_error():
    parser = StructuredOutputParser()

    response = '{"market": "Agricultura", "success": true}'

    with pytest.raises(StructuredOutputError) as exc_info:
        parser.parse(response)

    assert isinstance(
        exc_info.value.__cause__,
        ValidationError,
    )


def test_parser_exposes_error_code():
    parser = StructuredOutputParser()

    response = '{"market": "Agricultura", "success": true}'

    with pytest.raises(StructuredOutputError) as exc_info:
        parser.parse(response)

    assert exc_info.value.error_code == "missing"


def test_parser_error_is_retryable():
    parser = StructuredOutputParser()

    response = '{"market": "Agricultura", "success": true}'

    with pytest.raises(StructuredOutputError) as exc_info:
        parser.parse(response)

    error = exc_info.value

    assert error.error_code == "missing"
    assert is_retryable_error(error.error_code) is True
