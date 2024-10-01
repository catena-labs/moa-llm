# Streamlit Chat Apps

This directory contains two Streamlit applications for interacting with Mixture of Agents (MoA) models:

## streamlit_chat_local_server.py

This application is designed to connect to a server (see `server.py`) and interact with a pre-configured MoA model. It provides a simple chat interface where users can send messages and receive responses from the MoA model.

To run this application, you need to have the `server.py` (see `examples/openai_compatible_server/README.md`) running and accessible. Then, execute the following command:

```bash
streamlit run streamlit_chat_local_server.py
```

## streamlit_chat.py

This application allows users to experiment with MoA configurations directly within the Streamlit interface. Users can specify the proposer models, their temperatures, and the aggregator model with its temperature and system prompt.

### Screenshots

![Configuration Screenshot](assets/config_screenshot.png?raw=true)

![Chat Screenshot](assets/chat_screenshot.png?raw=true)

The application dynamically creates the MoA configuration based on the user's input and enables interactive chat with the configured MoA model.

To run this application, execute the following command:

```bash
streamlit run streamlit_chat.py
```

Both applications provide a user-friendly chat interface for interacting with MoA models, either by connecting to a pre-configured server or by dynamically configuring the MoA within the Streamlit app.