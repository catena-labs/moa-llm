"""
config_loader.py

This module is responsible for loading and parsing configuration files for the Mixture of Agents (MoA) system.
It provides functionality to:

1. Load configurations from YAML files or strings
2. Parse and validate the configuration structure
3. Create MixtureOfAgents instances based on the loaded configuration
4. Set up layers, neurons, and other components of the MoA system

The main functions in this module are:
- load_config: Loads and parses a configuration from various input types
- create_moa_from_config: Creates a MixtureOfAgents instance from a parsed configuration

This module plays a crucial role in setting up the MoA system by interpreting user-defined
configurations and instantiating the necessary components accordingly.
"""

import os
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

from .aggregation_layer import AggregationLayer
from .layer import Layer
from .mixture_of_agents import MixtureOfAgents
from .neuron import LLMNeuron
from .user_query_annotator import UserQueryAnnotator

load_dotenv(".env.local")
load_dotenv(".env")

BASE_URL = os.getenv("BASE_URL", "https://api.together.xyz/v1")


def load_config(
    config_input: Union[str, Dict[str, Any]], is_file_path: bool = False
) -> Dict[str, Any]:
    """
    Load and parse a configuration from various input types.

    This function can handle configurations provided as YAML strings, file paths to YAML files,
    or pre-loaded dictionaries.

    Args:
        config_input (Union[str, Dict[str, Any]]): The configuration input. Can be a YAML string,
            a file path to a YAML file, or a pre-loaded dictionary.
        is_file_path (bool, optional): If True, treats a string input as a file path.
            Defaults to False.

    Returns:
        Dict[str, Any]: The parsed configuration as a dictionary.

    Raises:
        ValueError: If the input is invalid or cannot be parsed.
        FileNotFoundError: If is_file_path is True and the specified file doesn't exist.
    """
    if isinstance(config_input, str):
        if is_file_path:
            try:
                with open(config_input, "r") as file:
                    return yaml.safe_load(file)
            except FileNotFoundError:
                raise ValueError(f"Config file not found: {config_input}")
        else:
            try:
                return yaml.safe_load(config_input)
            except yaml.YAMLError:
                raise ValueError("Invalid YAML string provided")
    elif isinstance(config_input, dict):
        return config_input
    else:
        raise ValueError("config_input must be either a YAML string, a file path, or a dictionary")


def create_moa_from_config(
    config_input: Union[str, Dict[str, Any]],
    is_file_path: bool = False,
    max_workers: int = 4,
    messages: Optional[List[Dict[str, str]]] = None,
) -> MixtureOfAgents:
    """
    Create a MixtureOfAgents instance from a configuration.

    Args:
        config_input (Union[str, Dict[str, Any]]): Configuration input as YAML string, file path, or dictionary.
        is_file_path (bool, optional): If True, treats config_input as a file path. Defaults to False.
        max_workers (int, optional): Maximum number of concurrent workers. Defaults to 4.
        messages (Optional[List[Dict[str, str]]], optional): Initial messages for the MoA. Defaults to None.

    Returns:
        MixtureOfAgents: An initialized MixtureOfAgents instance.
    """
    # Load and parse the configuration
    config = load_config(config_input, is_file_path)
    base_url = config.get("base_url", "https://api.together.xyz/v1")

    # Create UserQueryAnnotator if specified in the config
    annotator = None
    if config.get("use_annotator", False):
        annotator_config = config.get("annotator", {})
        annotator = UserQueryAnnotator(
            model=annotator_config.get("model"),
            temperature=annotator_config.get("temperature", 0.7),
            system_prompt=annotator_config.get("system_prompt"),
            base_url=annotator_config.get("base_url", BASE_URL),
            moa_id=None,  # Will be set later
            neuron_type="annotator",
        )

    # Create proposal layers
    proposal_layers = []
    for layer_index, layer_config in enumerate(config.get("proposal_layers", [])):
        neurons = [
            LLMNeuron(
                model=neuron["model"],
                system_prompt=neuron.get("system_prompt", ""),
                temperature=neuron.get("temperature", 0.7),
                weight=neuron.get("weight", 1.0),
                max_tokens=neuron.get("max_tokens", 2048),
                base_url=base_url,
                moa_id=None,
                neuron_type=f"proposer_layer_{layer_index + 1}",
            )
            for neuron in layer_config["neurons"]
        ]
        proposal_layers.append(Layer(neurons))

    # Create aggregation layer
    agg_config = config["aggregation_layer"]
    agg_neuron = LLMNeuron(
        model=agg_config["model"],
        temperature=agg_config.get("temperature", 0.7),
        max_tokens=agg_config.get("max_tokens", 2048),
        base_url=base_url,
        moa_id=None,
        neuron_type="aggregator",
    )
    aggregation_layer = AggregationLayer(
        agg_neuron,
        agg_config.get("system_prompt", None),
        shuffle=agg_config.get("shuffle", False),
        dropout_rate=agg_config.get("dropout_rate", 0.0),
        use_weights=agg_config.get("use_weights", False),
    )

    # Create MixtureOfAgents instance
    moa = MixtureOfAgents(
        proposal_layers=proposal_layers,
        aggregator_layer=aggregation_layer,
        annotator=annotator,
        use_annotator=config.get("use_annotator", False),
        max_workers=max_workers,
        pass_corresponding_results=config.get("pass_corresponding_results", False),
        messages=messages,
    )

    # Set moa_id for all neurons
    for layer in proposal_layers:
        for neuron in layer.neurons:
            neuron.moa_id = moa.moa_id
    aggregation_layer.neuron.moa_id = moa.moa_id
    if annotator:
        annotator.moa_id = moa.moa_id

    return moa
