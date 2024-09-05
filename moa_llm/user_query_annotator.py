import os
from typing import Optional

from .neuron import LLMNeuron


class UserQueryAnnotator(LLMNeuron):
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        base_url: str = os.getenv("BASE_URL", "https://api.together.xyz/v1"),
        moa_id: Optional[str] = None,
        neuron_type: str = "annotator",
    ):
        default_system_prompt = """
        You are a User Query Annotator. Your task is to process and optimize the user's query, breaking it down into clear, specific steps and add context if necessary.

        Only return the rephrased prompt and nothing else.
        """
        super().__init__(
            model,
            system_prompt or default_system_prompt,
            temperature,
            weight=1.0,
            base_url=base_url,
            moa_id=moa_id,
            neuron_type=neuron_type,
        )

    async def annotate(self, user_query: str) -> str:
        result = await self.process(user_query)
        return result["content"]
