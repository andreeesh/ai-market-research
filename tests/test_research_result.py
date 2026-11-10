import pytest
from pydantic import ValidationError

from ai_market_research.models import (
    ResearchFailure,
    ResearchResult,
    ResearchSuccess,
    research_outcome_adapter,
)


def test_research_result_serialization():
    result = ResearchResult(
        market="Agricultura",
        success=True,
        content="Investigacion completada",
    )

    data = result.model_dump()

    assert data["market"] == "Agricultura"
    assert data["success"] is True
    assert data["content"] == "Investigacion completada"
    assert data["error"] is None


@pytest.mark.parametrize(
    "success, content, error",
    [
        (True, "Reporte válido", "Error inesperado"),
        (False, "Reporte inválido", "Investigación fallida"),
    ],
)
def test_research_result_rejects_inconsistent_data(success, content, error):
    with pytest.raises(ValueError):
        ResearchResult(
            market="Agricultura",
            success=success,
            content=content,
            error=error,
        )


@pytest.mark.parametrize(
    "success, content, error",
    [
        (True, None, None),
        (False, None, None),
    ],
)
def test_research_result_rejects_missing_data(success, content, error):
    with pytest.raises(ValueError):
        ResearchResult(
            market="Agricultura",
            success=success,
            content=content,
            error=error,
        )


def test_research_success_rejects_false():
    with pytest.raises(ValueError):
        ResearchSuccess(
            market="Agricultura",
            success=False,
            content="Reporte completado",
        )


def test_research_failure_rejects_true():
    with pytest.raises(ValueError):
        ResearchFailure(
            market="Software",
            success=True,
            error="Proveedor no disponible",
        )


def test_research_outcome_discriminator():
    success_data = {
        "market": "Agricultura",
        "success": True,
        "content": "Reporte completado",
    }

    failure_data = {
        "market": "Software",
        "success": False,
        "error": "Proveedor no disponible",
    }

    success = research_outcome_adapter.validate_python(success_data)
    failure = research_outcome_adapter.validate_python(failure_data)

    assert isinstance(success, ResearchSuccess)
    assert isinstance(failure, ResearchFailure)


@pytest.mark.parametrize(
    "success_value",
    [
        None,
        "unknown",
        123,
    ],
)
def test_research_outcome_rejects_invalid_discriminator(success_value):
    data = {
        "market": "Agricultura",
        "success": success_value,
        "content": "Reporte completado",
    }

    with pytest.raises(ValidationError):
        research_outcome_adapter.validate_python(data)


def test_research_outcome_missing_discriminator():
    data = {
        "market": "Agricultura",
        "content": "Reporte completado",
    }

    with pytest.raises(ValidationError) as exc_info:
        research_outcome_adapter.validate_python(data)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "union_tag_not_found"


def test_research_outcome_invalid_discriminator_error():
    data = {
        "market": "Agricultura",
        "success": "unknown",
        "content": "Reporte completado",
    }

    with pytest.raises(ValidationError) as exc_info:
        research_outcome_adapter.validate_python(data)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "union_tag_invalid"


def test_research_outcome_validate_json():
    json_response = """
    {
        "market": "Agricultura",
        "success": true,
        "content": "El mercado presenta oportunidades"
    }
    """

    result = research_outcome_adapter.validate_json(json_response)

    assert isinstance(result, ResearchSuccess)
    assert result.market == "Agricultura"
    assert result.success is True
    assert result.content == "El mercado presenta oportunidades"

def test_research_outcome_rejects_malformed_json():
    json_response = """
    {
        "market": "Agricultura",
        "success": true,
    """

    with pytest.raises(ValidationError) as exc_info:
        research_outcome_adapter.validate_json(json_response)

    errors = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["type"] == "json_invalid"