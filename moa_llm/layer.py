import asyncio
from typing import Any, Dict, List, Optional, Sequence

from .neuron import Neuron


class Layer:
    """
    Represents a layer of neurons in the mixture of agents model.

    This class manages a sequence of neurons and handles their concurrent processing.
    """

    def __init__(self, neurons: Sequence[Neuron], max_workers: int = 4):
        """
        Initialize a Layer instance.

        Args:
            neurons (Sequence[Neuron]): A sequence of Neuron objects that make up this layer.
            max_workers (int, optional): The maximum number of concurrent workers. Defaults to 4.
        """
        self.neurons = neurons
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process(
        self, input_data: str, prev_response: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process the input data through all neurons in the layer concurrently.

        Args:
            input_data (str): The input data to be processed.
            prev_response (Optional[List[Dict[str, Any]]], optional): The response from the previous layer, if any. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the results from each neuron.
        """
        prev_response_content = [r["content"] for r in prev_response] if prev_response else None

        async def process_neuron(neuron):
            async with self.semaphore:
                return await neuron.process(input_data, prev_response_content)

        tasks = [process_neuron(neuron) for neuron in self.neurons]
        results = await asyncio.gather(*tasks)

        return results

    @property
    def max_workers(self) -> int:
        """
        Get the maximum number of concurrent workers.

        Returns:
            int: The maximum number of workers.
        """
        return self._max_workers

    @max_workers.setter
    def max_workers(self, value: int):
        """
        Set the maximum number of concurrent workers and update the semaphore.

        Args:
            value (int): The new maximum number of workers.
        """
        self._max_workers = value
        self.semaphore = asyncio.Semaphore(value)
