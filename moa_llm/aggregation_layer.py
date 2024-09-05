import random
import string
from typing import Any, Dict, List, Optional

from .layer import Layer
from .neuron import Neuron


class AggregationLayer(Layer):
    """
    A layer that aggregates responses from previous layers and processes them through a single neuron.

    This layer can optionally shuffle the responses, apply dropout, and use a custom aggregation prompt template.

    Attributes:
        neuron (Neuron): The neuron used for processing the aggregated input.
        aggregation_prompt_template (str | None): Template for aggregating responses.
        shuffle (bool): Whether to shuffle the responses before processing.
        dropout_rate (float): The rate at which to randomly drop responses.
        use_weights (bool): Whether to use weights for the responses (not implemented in this method).
    """

    def __init__(
        self,
        neuron: Neuron,
        aggregation_prompt_template: str | None = None,
        shuffle: bool = False,
        dropout_rate: float = 0.0,
        use_weights: bool = False,
    ):
        """
        Initialize the AggregationLayer.

        Args:
            neuron (Neuron): The neuron to use for processing.
            aggregation_prompt_template (str | None, optional): Template for aggregating responses. Defaults to None.
            shuffle (bool, optional): Whether to shuffle responses. Defaults to False.
            dropout_rate (float, optional): Rate for randomly dropping responses. Defaults to 0.0.
            use_weights (bool, optional): Whether to use weights for responses. Defaults to False.
        """
        super().__init__([neuron])
        self.neuron = neuron
        self.aggregation_prompt_template = aggregation_prompt_template
        self.shuffle = shuffle
        self.dropout_rate = dropout_rate
        self.use_weights = use_weights

    async def process(
        self, input_data: Any, prev_response: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process the input data and previous responses through the aggregation layer.

        Args:
            input_data (Any): The input data to process.
            prev_response (Optional[List[Dict[str, Any]]], optional): Previous responses to aggregate. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list containing the result of processing.
        """
        if prev_response is None:
            prev_response = []

        if self.shuffle:
            random.shuffle(prev_response)

        # Apply dropout
        prev_response = [r for r in prev_response if random.random() > self.dropout_rate]

        if self.aggregation_prompt_template is not None:
            aggregated_input = self.collect_responses(
                self.aggregation_prompt_template, prev_response, input_data
            )

            self.neuron.system_prompt = aggregated_input

        result = await self.neuron.process(input_data)

        return [result]

    @staticmethod
    def collect_responses(
        aggregation_prompt_template: str, results: List[Dict[str, Any]], user_query: str
    ) -> str:
        """
        Collect and format responses from multiple models for aggregation.

        Args:
            aggregation_prompt_template (str): The template for formatting the aggregated prompt.
            results (List[Dict[str, Any]]): List of results from different models.
            user_query (str): The original user query.

        Returns:
            str: Formatted string containing aggregated responses and weights.
        """
        # Combine responses from all models
        responses = "\n".join(
            [f"Model {i+1}:\n\n{str(result['content'])}" for i, result in enumerate(results)]
        )

        # Collect weights if available
        weight_str = ""
        if any("weight" in result for result in results):
            weight_str = "\n".join(
                [f"Model {i+1}: {result.get('weight', 1.0)}" for i, result in enumerate(results)]
            )

        # Prepare arguments for formatting
        format_args = {
            "user_query": user_query,
            "responses": responses,
            "response_weights": weight_str,
        }

        # Ensure all required keys are present in the template
        required_keys = [
            key[1]
            for key in string.Formatter().parse(aggregation_prompt_template)
            if key[1] is not None
        ]
        for key in required_keys:
            if key not in format_args:
                format_args[key] = f"[{key} not provided]"

        # Format and return the final aggregated prompt
        return aggregation_prompt_template.format(**format_args)
