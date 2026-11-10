from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, HttpUrl, TypeAdapter, model_validator


@dataclass
class Opportunity:
    title: str
    description: str
    market: str
    estimated_competitors: int


class ValidateOpportunity(BaseModel):
    title: str
    estimated_competitors: int = Field(strict=True, ge=0)
    problem: str
    source_url: HttpUrl


class ResearchResult(BaseModel):
    market: str
    success: bool
    content: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.success:
            if self.error is not None:
                raise ValueError("Un resultado exitoso no puede contener errores")

            if self.content is None:
                raise ValueError("Un resultado exitoso debe contener contenido")

        else:
            if self.content is not None:
                raise ValueError("Un resultado fallido no puede contener contenido")

            if self.error is None:
                raise ValueError("Un resultado fallido debe contener un error")

        return self


class ResearchSuccess(BaseModel):
    market: str
    success: Literal[True] = True
    content: str


class ResearchFailure(BaseModel):
    market: str
    success: Literal[False] = False
    error: str


ResearchOutcome = Annotated[
    ResearchSuccess | ResearchFailure,
    Field(discriminator="success"),
]

research_outcome_adapter = TypeAdapter(ResearchOutcome)

opportunity = Opportunity(
    title="Software para veterinarias",
    description="Gestion de turnos y pacientes",
    market="Argentina",
    estimated_competitors=12,
)
