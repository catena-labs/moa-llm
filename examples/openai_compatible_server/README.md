# OpenAI-Compatible Server with MoA

This directory contains an example of how to use the MoA (Mixture of Annotators) framework to create a local server that provides an OpenAI-compatible API for chat completions. The server is built using FastAPI and can be configured to use different models and settings for the proposal and aggregation layers of the MoA.

## Prerequisites

Before running the server, make sure you have the following dependencies installed:

- Python 3.7 or higher
- FastAPI
- Pydantic
- MoA-LLM

You can install the required Python packages using pip:

```bash
pip install fastapi pydantic moa-llm
```

## Configuration

The server configuration is defined in the `moa_config.yaml` file. This file specifies the models and settings for the proposal and aggregation layers of the MoA. You can modify this file to use different models or adjust the temperature and other settings.

## Running the Server

To run the server, execute the following command:

```bash
python server.py [path/to/moa_config.yaml]
```

Replace `[path/to/moa_config.yaml]` with the path to your MoA configuration file.

If you don't provide a path to the configuration file, the server will use the default `moa_config.yaml` file in the same directory.

The server will start running on `http://0.0.0.0:8000`. You can send POST requests to the `/chat/completions` endpoint with a JSON payload containing the chat messages and other parameters, as specified by the OpenAI API.

Here's an example of a valid request payload:

```json
{
  "model": "moa-model",
  "messages": [
    {"role": "user", "content": "Hello, can you help me with a coding problem?"},
  ],
  "max_tokens": 512,
}
```

The server will respond with a JSON payload containing the generated chat completion, following the OpenAI API format.

## Streaming Responses

If you set the `stream` parameter to `true` in the request payload, the server will stream the generated response token by token, allowing you to display the response as it's being generated.

## Customization

You can customize the server by modifying the `moa_config.yaml` file or the `server.py` script. For example, you can change the models used in the proposal and aggregation layers, adjust the temperature and other settings, or modify the server's behavior.

## License

This example is part of the MoA-LLM project and is licensed under the [MIT License](https://opensource.org/licenses/MIT).