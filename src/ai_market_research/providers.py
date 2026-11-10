import asyncio
from typing import Protocol


class ResearchProvider(Protocol):
    async def research(self, market: str) -> str: ...


class MockResearchProvider:
    async def research(self, market: str) -> str:
        await asyncio.sleep(1)
        return f"Investigacion completada: {market}"


class StaticResearchProvider:
    async def research(self, market: str) -> str:
        await asyncio.sleep(1)
        return f"Reporte estatico del mercado {market}"
