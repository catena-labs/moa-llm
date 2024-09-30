"""
This script implements a simple chat interface using Streamlit and communicates with a MoA (Mixture of Agents) model.
It allows users to interact with the AI model in a conversational manner.
"""

import requests
import streamlit as st

# Set the title of the Streamlit app
st.title("MoA Chat")

# Initialize the chat history in the session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages in the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input and process it
if prompt := st.chat_input("Ask a question"):
    # Add user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the payload for the API request
    payload = {
        "model": "moa-model",
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ],
        "stream": False,
    }

    # Send a POST request to the MoA model API
    response = requests.post("http://0.0.0.0:8000/chat/completions", json=payload)
    data = response.json()

    # Extract the assistant's response from the API response
    assistant_response = data["choices"][0]["message"]["content"]
    
    # Add assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    # Display the assistant's response in the chat interface
    with st.chat_message("assistant"):
        st.markdown(assistant_response)