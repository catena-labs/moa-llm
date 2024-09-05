import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env.local file
load_dotenv(".env.local")
load_dotenv(".env")

BASE_URL = os.getenv("BASE_URL", "https://api.together.xyz/v1")

logger = logging.getLogger(__name__)


class Neuron(ABC):
    """
    Abstract base class for a neuron in the mixture of agents model.
    """

    def __init__(self, system_prompt: str, temperature: float, weight: float = 1.0):
        """
        Initialize a Neuron.

        Args:
            system_prompt (str): The system prompt for the neuron.
            temperature (float): The temperature parameter for response generation.
            weight (float, optional): The weight of this neuron in the layer. Defaults to 1.0.
        """
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.weight = weight

    @abstractmethod
    async def process(self, input_data: Any, prev_response: List[str] | None = None) -> Any:
        """
        Process the input data.

        Args:
            input_data (Any): The input data to process.
            prev_response (List[str] | None, optional): Previous responses. Defaults to None.

        Returns:
            Any: The processed output.
        """
        pass


class LLMNeuron(Neuron):
    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        weight: float = 1.0,
        max_tokens: int = 2048,
        base_url: str = BASE_URL,
        moa_id: Optional[str] = None,
        num_previous_responses: int = int(os.getenv("NUM_PREVIOUS_RESPONSES", "1")),
        neuron_type: str = "",
    ):
        super().__init__(system_prompt, temperature, weight)
        self.model = model
        self.max_tokens = max_tokens
        api_key = os.getenv("PROVIDER_API_KEY")
        if not api_key:
            raise ValueError("PROVIDER_API_KEY not found in .env.local file")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.stream = os.getenv("STREAM", "false").lower() == "true"
        self.moa_id = moa_id
        self.num_previous_responses = num_previous_responses
        self.neuron_type = neuron_type

    def log_result(self, messages, result, response_time):
        if os.environ.get("LOG_RESULTS", "true").lower() == "true":
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            date_str = datetime.now().strftime("%Y_%m_%d")
            moa_id = self.moa_id or "unknown"
            output_file = output_dir / f"{date_str}_moa_{moa_id}.jsonl"

            log_entry = {
                "input": messages,
                "output": result,
                "generator": self.model,
                "moa_id": moa_id,
                "temperature": self.temperature,
                "response_time": response_time,
                "neuron_type": self.neuron_type,
            }

            with output_file.open("a") as f:
                json.dump(log_entry, f)
                f.write("\n")

    async def process(
        self, input_data: Union[str, List[Dict[str, str]]], prev_response: List[str] | None = None
    ) -> Dict[str, Any]:
        """
        Process the input data through the LLMNeuron.

        Args:
            input_data (Union[str, List[Dict[str, str]]]): The input data to be processed. It can be a string or a list of dictionaries with 'role' and 'content' keys.
            prev_response (List[str] | None, optional): The previous responses to be included in the system prompt. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the response content, response time, and weight.
        """
        start_time = time.time()

        messages = (
            input_data
            if isinstance(input_data, list)
            else [{"role": "user", "content": input_data}]
        )

        if prev_response:
            last_n_responses = prev_response[-self.num_previous_responses :]
            system_content = self.collect_responses(
                self.system_prompt if self.system_prompt else "", last_n_responses
            )
            messages.insert(0, {"role": "system", "content": system_content})

        for sleep_time in [1, 2, 4, 8, 16]:
            try:
                if self.stream:
                    response_content = await self.process_stream(messages)
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    response_content = response.choices[0].message.content

                end_time = time.time()
                response_time = end_time - start_time

                logger.debug(f"Model: {self.model}")
                logger.debug(f"Prompt: {messages}")
                logger.debug(f"Response: {response_content}")
                logger.info(f"Response Time: {response_time:.2f} seconds")

                self.log_result(messages, response_content, response_time)

                return {"content": response_content, "time": response_time, "weight": self.weight}

            except Exception as e:
                logger.error(f"Error requesting {self.model}; prompt: {input_data}")
                logger.error(f"Error: {e}. Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

        # If all retries fail, return the default response
        end_time = time.time()
        response_time = end_time - start_time
        logger.error(f"Failed to get response from model {self.model} after multiple retries")
        return {
            "content": "Model did not generate a response. Ignore this.",
            "time": response_time,
            "weight": self.weight,
        }

    async def process_stream(self, messages: List[Dict[str, str]]) -> str:
        chat_completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        response_content = ""
        async for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content

        return response_content

    @staticmethod
    def collect_responses(prompt: str, prev_responses: List[str]) -> str:
        """
        Collect and format previous responses.

        This method currently uses a simple concatenation approach to aggregate responses.
        In the future, we could explore more sophisticated aggregation methods.

        Args:
            prompt (str): The original prompt.
            prev_responses (List[str]): List of previous responses.

        Returns:
            str: Formatted string containing the original prompt and previous responses.
        """
        responses = "\n".join(
            [
                f"## Model {i+1}\n\nResponse:\n\n{str(response)}"
                for i, response in enumerate(prev_responses)
            ]
        )
        return f"{prompt}\n\n# Previous responses:\n{responses}"
