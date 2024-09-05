"""
This Python file defines the MixtureOfAgents class, which implements a multi-layer approach for processing queries using various AI models. The class manages multiple proposal layers, an aggregation layer, and an optional query annotator to process input and produce a final output. Key features include:

1. Initialization of proposal layers, aggregation layer, and optional annotator
2. Configuration of processing parameters like max workers and result passing strategy
3. Methods for processing single queries or message arrays
4. Asynchronous processing of layers and aggregation of results
5. Timing and logging of processing steps
6. Handling of both single-turn queries and multi-turn conversations

"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from .aggregation_layer import AggregationLayer
from .layer import Layer
from .user_query_annotator import UserQueryAnnotator

logger = logging.getLogger(__name__)


class MixtureOfAgents:
    """
    A class that implements a mixture of agents approach for processing queries.

    This class manages multiple layers of neural networks and an aggregation layer
    to process input queries and produce a final output.
    """

    def __init__(
        self,
        proposal_layers: List[Layer],
        aggregator_layer: AggregationLayer,
        annotator: Optional[UserQueryAnnotator] = None,
        use_annotator: bool = False,
        max_workers: int = 4,
        pass_corresponding_results: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Initialize the MixtureOfAgents.

        Args:
            proposal_layers (List[Layer]): List of proposal layers for processing.
            aggregator_layer (AggregationLayer): The final aggregation layer.
            annotator (Optional[UserQueryAnnotator]): An optional query annotator.
            use_annotator (bool): Flag to determine if the annotator should be used.
            max_workers (int): Maximum number of concurrent workers.
            pass_corresponding_results (bool): Flag to pass corresponding results between layers, instead of fully connected layers.
            messages (Optional[List[Dict[str, str]]]): Optional list of messages to process.
        """
        self.proposal_layers = proposal_layers
        self.aggregator_layer = aggregator_layer
        self.annotator = annotator if isinstance(annotator, UserQueryAnnotator) else None
        self.use_annotator = use_annotator
        self.max_workers = max_workers
        for layer in self.proposal_layers:
            layer.max_workers = max_workers
        self.aggregator_layer.max_workers = max_workers
        self.moa_id = str(uuid.uuid4())
        self.pass_corresponding_results = pass_corresponding_results
        self.messages = messages

    async def process(self, input_data: Union[str, List[Dict[str, str]]]) -> Dict[str, Any]:
        """
        Process the input data through the mixture of agents.

        Args:
            input_data (Union[str, List[Dict[str, str]]]): The input query or messages to process.

        Returns:
            Dict[str, Any]: A dictionary containing the processed result, including
            content, response times, total completion time, and annotated query (if used).
        """
        start_time = time.time()
        results: List[Dict[str, Any]] = []
        response_times = {}

        # Use input_data directly if it's a list of messages, otherwise create a single message
        messages = (
            input_data
            if isinstance(input_data, list)
            else [{"role": "user", "content": input_data}]
        )

        # Layer 0: Process with the first proposal layer
        if len(self.proposal_layers) > 0:
            layer_results = await self.proposal_layers[0].process(messages)
            results.extend(layer_results)
            for i, result in enumerate(layer_results):
                response_times[f"Layer 1 - Neuron {i+1}"] = result["time"]

        # Layers 1 to N-1: Process with remaining proposal layers
        for layer_num in range(1, len(self.proposal_layers)):
            layer_results = []
            layer = self.proposal_layers[layer_num]

            if self.pass_corresponding_results:
                tasks = [
                    process_neuron(i, neuron, messages, results)
                    for i, neuron in enumerate(layer.neurons)
                ]
                layer_results = await asyncio.gather(*tasks)
                for i, neuron_output in enumerate(layer_results):
                    response_times[f"Layer {layer_num+1} - Neuron {i+1}"] = neuron_output["time"]
            else:
                layer_output = await layer.process(messages, prev_response=results)
                layer_results.extend(layer_output)
                for j, result in enumerate(layer_output):
                    response_times[f"Layer {layer_num+1} - Neuron {j+1}"] = result["time"]

            results.extend(layer_results)

        # Layer N: Final aggregation
        final_result = await self.aggregator_layer.process(messages, results)
        response_times["Aggregation Layer"] = final_result[0]["time"]

        end_time = time.time()
        total_completion_time = end_time - start_time

        logger.info(f"Total completion time: {total_completion_time:.2f} seconds")

        return {
            "content": final_result[0]["content"],
            "response_times": response_times,
            "total_completion_time": total_completion_time,
            "annotated_query": messages if self.use_annotator and self.annotator else None,
        }


async def process_neuron(i, neuron, messages, results):
    prev_result = [results[i]] if i < len(results) else []
    return await neuron.process(messages, prev_response=prev_result)
