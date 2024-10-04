import streamlit as st
import asyncio
import yaml

from typing import List, Dict
from moa_llm import MixtureOfAgents, LLMNeuron, AggregationLayer, Layer
from dotenv import load_dotenv

# Load environment variables from .env.local file
load_dotenv(".env.local")

DEFAULT_AGGREGATION_LAYER_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
  
Responses from models:
{responses}
"""

def convert_to_yaml(config):
    proposal_layers = []
    for layer in config["proposal_layers"]:
        neurons = []
        for neuron in layer["neurons"]:
            neuron_config = {
                "model": neuron["model_id"],
                "temperature": neuron["temperature"],
            }
            if neuron["weight"] != 1.0:
                neuron_config["weight"] = neuron["weight"]
            if neuron["system_prompt"]:
                neuron_config["system_prompt"] = neuron["system_prompt"]
            neurons.append(neuron_config)
        proposal_layers.append({"neurons": neurons})

    aggregation_layer = {
        "model": config["aggregation_layer"]["model_id"],
        "system_prompt": config["aggregation_layer"]["system_prompt"],
        "temperature": config["aggregation_layer"]["temperature"],
        "max_tokens": config["aggregation_layer"].get("max_tokens", 8000),
    }

    yaml_config = {
        "use_annotator": config["use_annotator"],
        "proposal_layers": proposal_layers,
        "aggregation_layer": aggregation_layer,
        "use_weights": config["use_weights"],
    }

    return yaml.dump(yaml_config)

def create_moa_config(config: Dict) -> MixtureOfAgents:
    """
    Create a Mixture of Agents (MOA) configuration based on the given configuration.

    Args:
        config (Dict): A dictionary containing the MOA configuration.

    Returns:
        MixtureOfAgents: A configured MOA instance.
    """
    proposal_layers = []
    for layer_config in config.get("proposal_layers", []):
        layer_neurons = []
        for neuron_config in layer_config.get("neurons", []):
            layer_neurons.append(
                LLMNeuron(
                    model=neuron_config["model_id"],
                    temperature=neuron_config["temperature"],
                    weight=neuron_config.get("weight", 1.0),
                    max_tokens=2048,
                    neuron_type=f"proposer_layer_{len(proposal_layers)+1}_neuron_{len(layer_neurons)+1}",
                    system_prompt=neuron_config.get("system_prompt", ""),
                )
            )
        proposal_layers.append(Layer(layer_neurons))

    agg_neuron = LLMNeuron(
        model=config["aggregation_layer"]["model_id"],
        temperature=config["aggregation_layer"]["temperature"],
        max_tokens=2048,
        neuron_type="aggregator",
        system_prompt=config["aggregation_layer"]["system_prompt"],
    )
    aggregation_layer = AggregationLayer(
        agg_neuron,
        shuffle=False,
        dropout_rate=0.0,
        use_weights=config.get("use_weights", False),
    )

    annotator = None
    if config.get("use_annotator", False):
        annotator = LLMNeuron(
            model=config["annotator"]["model_id"],
            temperature=config["annotator"]["temperature"],
            max_tokens=2048,
            neuron_type="annotator",
            system_prompt=config["annotator"]["system_prompt"],
        )

    return MixtureOfAgents(
        proposal_layers=proposal_layers,
        aggregator_layer=aggregation_layer,
        annotator=annotator,
        use_annotator=config.get("use_annotator", False),
        max_workers=4,
        pass_corresponding_results=config.get("pass_corresponding_results", False),
        messages=None,
    )

async def chat_page():
    """
    Chat page for interacting with the MOA.
    """
    st.title("MOA Chat")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    user_input = st.chat_input("Enter your message")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if "moa" in st.session_state:
            moa = st.session_state["moa"]
            # Process user input using MOA
            result = await moa.process([{"role": "user", "content": user_input}])
            st.session_state["messages"].append({"role": "assistant", "content": result["content"]})
            with st.chat_message("assistant"):
                st.markdown(result["content"])

async def app():
    """
    Main Streamlit application function.
    Handles the UI layout, user interactions, and MOA processing.
    """
    st.set_page_config(page_title="MOA Chat", layout="wide")

    # Sidebar navigation
    pages = {
        "Configuration": config_page,
        "Chat": chat_page,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    if selection == "Chat":
        await chat_page()
    else:
        pages[selection]()

def config_page():
    """
    Configuration page for setting up the MOA.
    """
    st.title("MOA Configuration")

    if "config" in st.session_state:
        print("config in session state")
        config = st.session_state["config"]
    else:
        print("config not in session state")
        config = {
            "annotator": None,
            "proposal_layers": [],
            "aggregation_layer": {
                "model_id": "",
                "temperature": 0.7,
                "system_prompt": DEFAULT_AGGREGATION_LAYER_SYSTEM_PROMPT,
            },
            "use_annotator": False,
            "use_weights": False,
            "pass_corresponding_results": False,
        }

    use_annotator = st.sidebar.checkbox("Use Annotator", value=config["use_annotator"], key="use_annotator")
    if use_annotator:
        st.subheader("Annotator")
        annotator_model_id = st.text_input("Annotator Model ID", value=config["annotator"]["model_id"] if config["annotator"] else "", key="annotator_model_id")
        annotator_temperature = st.slider("Annotator Temperature", 0.0, 1.0, value=config["annotator"]["temperature"] if config["annotator"] else 0.7, key="annotator_temperature")
        annotator_system_prompt = st.text_area("Annotator System Prompt", value=config["annotator"]["system_prompt"] if config["annotator"] else "", key="annotator_system_prompt")

    # Proposal layers configuration
    st.subheader("Proposal Layers")
    num_proposal_layers = st.sidebar.number_input("Number of Proposal Layers", min_value=1, max_value=3, value=len(config["proposal_layers"]), step=1, key="num_proposal_layers")
    proposal_layers = []
    for layer_idx in range(num_proposal_layers):
        with st.expander(f"Proposal Layer {layer_idx+1}"):
            existing_layer_config = config["proposal_layers"][layer_idx] if layer_idx < len(config["proposal_layers"]) else {"neurons": []}
            num_neurons = st.sidebar.number_input(f"Number of Neurons in Layer {layer_idx+1}", min_value=1, max_value=5, value=len(existing_layer_config["neurons"]) if existing_layer_config["neurons"] else 1, step=1, key=f"num_neurons_layer_{layer_idx+1}")
            layer_neurons = []
            cols = st.columns(num_neurons)
            for neuron_idx, col in enumerate(cols):
                existing_neuron_config = existing_layer_config["neurons"][neuron_idx] if neuron_idx < len(existing_layer_config["neurons"]) else {}
                with col:
                    model_id = st.text_input(f"Model ID for Neuron {neuron_idx+1}", value=existing_neuron_config.get("model_id", ""), key=f"neuron_{layer_idx+1}_{neuron_idx+1}_model_id")
                    temperature = st.slider(f"Temperature for Neuron {neuron_idx+1}", 0.0, 1.0, value=existing_neuron_config.get("temperature", 0.7), key=f"neuron_{layer_idx+1}_{neuron_idx+1}_temperature")
                    weight = st.number_input(f"Weight for Neuron {neuron_idx+1}", min_value=0.0, value=existing_neuron_config.get("weight", 1.0), step=0.1, key=f"neuron_{layer_idx+1}_{neuron_idx+1}_weight")
                    system_prompt = st.text_area(f"System Prompt for Neuron {neuron_idx+1}", value=existing_neuron_config.get("system_prompt", ""), key=f"neuron_{layer_idx+1}_{neuron_idx+1}_system_prompt")
                    layer_neurons.append({"model_id": model_id, "temperature": temperature, "weight": weight, "system_prompt": system_prompt})
            proposal_layers.append({"neurons": layer_neurons})

    # Aggregation layer configuration
    st.subheader("Aggregation Layer")
    agg_model_id = st.text_input("Aggregator Model ID", value=config["aggregation_layer"]["model_id"], key="agg_model_id")
    agg_temperature = st.slider("Aggregator Temperature", 0.0, 1.0, value=config["aggregation_layer"]["temperature"], key="agg_temperature")
    agg_system_prompt = st.text_area("Aggregator System Prompt", value=config["aggregation_layer"]["system_prompt"], key="agg_system_prompt")

    # Other configurations
    use_weights = st.checkbox("Use Weights", value=config["use_weights"], key="use_weights")
    pass_corresponding_results = st.checkbox("Pass Corresponding Results", value=config["pass_corresponding_results"], key="pass_corresponding_results")
    show_json = st.checkbox("Show JSON", key="show_json")

    if st.button("Save Configuration"):
        config = {
            "annotator": {
                "model_id": annotator_model_id,
                "temperature": annotator_temperature,
                "system_prompt": annotator_system_prompt,
            } if use_annotator else None,
            "proposal_layers": proposal_layers,
            "aggregation_layer": {
                "model_id": agg_model_id,
                "temperature": agg_temperature,
                "system_prompt": agg_system_prompt,
            },
            "use_annotator": use_annotator,
            "use_weights": use_weights,
            "pass_corresponding_results": pass_corresponding_results,
        }
        st.session_state["config"] = config
        st.session_state["moa"] = create_moa_config(config)

        if show_json:
            with st.expander("Configuration JSON"):
                st.subheader("Configuration JSON")
                st.json(config)
                yaml_config = convert_to_yaml(config)
                st.code(yaml_config, language="yaml")            


if __name__ == "__main__":
    asyncio.run(app())