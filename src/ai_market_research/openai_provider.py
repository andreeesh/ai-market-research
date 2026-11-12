from openai import AsyncOpenAI


class OpenAIResearchProvider:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def research(self, market: str) -> str:
        response = await self.client.responses.create(
            model="gpt-5-mini",
            input=f"Investigá el mercado de {market}.",
        )

        return response.output_text
