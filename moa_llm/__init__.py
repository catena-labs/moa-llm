import logging

from .aggregation_layer import AggregationLayer
from .config_loader import create_moa_from_config
from .layer import Layer
from .mixture_of_agents import MixtureOfAgents
from .neuron import LLMNeuron, Neuron
from .user_query_annotator import UserQueryAnnotator
from .version import VERSION, VERSION_SHORT

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
__all__ = [
    "Neuron",
    "LLMNeuron",
    "Layer",
    "AggregationLayer",
    "MixtureOfAgents",
    "UserQueryAnnotator",
]
