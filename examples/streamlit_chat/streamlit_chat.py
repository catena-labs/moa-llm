import streamlit as st
import asyncio
from typing import List, Dict
from moa_llm import MixtureOfAgents, LLMNeuron, AggregationLayer, Layer
from dotenv import load_dotenv

# Load environment variables from .env.local file
load_dotenv(".env.local")

def create_moa_config(proposers: List[Dict[str, float]], aggregator: Dict[str, str]) -> MixtureOfAgents:
    """
    Create a Mixture of Agents (MOA) configuration based on the given proposers and aggregator settings.

    Args:
        proposers (List[Dict[str, float]]): A list of dictionaries containing proposer configurations.
        aggregator (Dict[str, str]): A dictionary containing aggregator configurations.

    Returns:
        MixtureOfAgents: A configured MOA instance.
    """
    proposal_layers = [
        Layer([
            LLMNeuron(
                model=proposer["model_id"],
                temperature=proposer["temperature"],
                weight=1.0,
                max_tokens=2048,
                neuron_type=f"proposer_layer_{i+1}",
            )
        ]) for i, proposer in enumerate(proposers)
    ]

    agg_neuron = LLMNeuron(
        model=aggregator["model_id"],
        temperature=aggregator["temperature"],
        max_tokens=2048,
        neuron_type="aggregator",
    )
    aggregation_layer = AggregationLayer(
        agg_neuron,
        shuffle=False,
        dropout_rate=0.0,
        use_weights=False,
    )
    aggregation_layer.system_prompt = aggregator["system_prompt"]

    return MixtureOfAgents(
        proposal_layers=proposal_layers,
        aggregator_layer=aggregation_layer,
        annotator=None,
        use_annotator=False,
        max_workers=4,
        pass_corresponding_results=False,
        messages=None,
    )

async def app():
    """
    Main Streamlit application function.
    Handles the UI layout, user interactions, and MOA processing.
    """
    st.title("MOA Chat")

    # Configuration section
    st.header("Configuration")
    proposers = []
    for i in range(4):
        with st.expander(f"Proposer {i+1}"):
            model_id = st.text_input(f"Model ID for Proposer {i+1}", key=f"proposer_{i+1}_model_id")
            temperature = st.slider(f"Temperature for Proposer {i+1}", 0.0, 1.0, 0.7, key=f"proposer_{i+1}_temperature")
            if model_id:
                proposers.append({"model_id": model_id, "temperature": temperature})

    with st.expander("Aggregator"):
        agg_model_id = st.text_input("Model ID for Aggregator", key="agg_model_id")
        agg_temperature = st.slider("Temperature for Aggregator", 0.0, 1.0, 0.7, key="agg_temperature")
        agg_system_prompt = st.text_area("System Prompt for Aggregator", key="agg_system_prompt")

    if st.button("Save Configuration"):
        aggregator = {
            "model_id": agg_model_id,
            "temperature": agg_temperature,
            "system_prompt": agg_system_prompt,
        }
        st.session_state["moa"] = create_moa_config(proposers, aggregator)

    # Chat section
    st.header("Chat")
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

if __name__ == "__main__":
    asyncio.run(app())