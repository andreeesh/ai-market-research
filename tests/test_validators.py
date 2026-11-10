import pytest
from pydantic import ValidationError

from ai_market_research.validators import validate_research


def test_negative_competitors():
    data = {
        "title": "Software agrícola",
        "estimated_competitors": -5,
        "problem": "Falta de herramientas",
        "source_url": "https://example.com",
    }

    with pytest.raises(ValidationError) as exc_info:
        validate_research(data)

        errors = exc_info.value.errors()

        assert errors[0]["loc"] == ("estimated_competitors",)
        assert errors[0]["type"] == "greater_than_equal"


def test_valid_opportunity():
    data = {
        "title": "Software de gestión agrícola",
        "estimated_competitors": 5,
        "problem": "Los productores necesitan mejores herramientas",
        "source_url": "https://example.com",
    }

    result = validate_research(data)

    assert result.title == "Software de gestión agrícola"


def test_competitors_wrong_type():
    data = {
        "title": "Software agrícola",
        "estimated_competitors": "5",
        "problem": "Falta de herramientas",
        "source_url": "https://example.com",
    }

    with pytest.raises(ValidationError) as exc_info:
        validate_research(data)

    errors = exc_info.value.errors()

    assert errors[0]["loc"] == ("estimated_competitors",)
    assert errors[0]["type"] == "int_type"


@pytest.mark.parametrize("competitors", [-1, -5, -100])
def test_negative_competitors_parametrized(competitors):
    data = {
        "title": "Software agrícola",
        "estimated_competitors": competitors,
        "problem": "Falta de herramientas",
        "source_url": "https://example.com",
    }

    with pytest.raises(ValidationError):
        validate_research(data)

def test_zero_competitors():
    data = {
        "title": "Nuevo mercado",
        "estimated_competitors": 0,
        "problem": "Necesidad sin resolver",
        "source_url": "https://example.com",
    }

    result = validate_research(data)

    assert result.estimated_competitors == 0