# moa-llm: Mixture of Agents for LLMs

## Overview

moa-llm is a Python library that orchestrates Large Language Models (LLMs) in a neural network-inspired structure. This innovative approach enables sophisticated, multi-step processing of queries by leveraging various LLMs as "neurons" within the network. moa-llm is designed to harness the unique strengths of different models, combining their outputs to produce more comprehensive, accurate, and nuanced results than any single model could achieve alone.

By emulating the layered structure of neural networks, moa-llm allows for complex information processing workflows. It can handle tasks that require diverse knowledge domains, multiple perspectives, or step-by-step reasoning. This makes it particularly suitable for applications in areas such as advanced question-answering systems, multi-faceted analysis, creative content generation, and complex problem-solving scenarios.

moa-llm is inspired by the work done by Together AI in their research blog [Together MoA — collective intelligence of open-source models pushing the frontier of LLM capabilities](https://www.together.ai/blog/together-moa). The Mixture of Agents (MoA) approach adopts a layered architecture where each layer comprises several LLM agents. These agents take the outputs from the previous layer as auxiliary information to generate refined responses, effectively integrating diverse capabilities and insights from various models.

For more technical details, refer to the arXiv paper: [Mixture-of-Agents: Architecting Large Language Models as Interacting Experts](https://arxiv.org/abs/2406.04692).

## Key Features

- **Flexible Multi-Layer Architecture**: Supports an arbitrary number of layers, including multiple proposal layers and a final aggregation layer, allowing for deep and complex query processing pipelines.
- **Weighted Model Inputs**: Each LLM "neuron" can be assigned a customizable weight, enabling fine-tuned control over the influence of different models on the final output. This feature allows for the prioritization of more reliable or task-appropriate models.
- **Intelligent Query Annotation**: An optional pre-processing step that can reformulate, expand, or contextualize user queries to optimize them for the subsequent layers. This can significantly improve the relevance and quality of the final output.
- **Asynchronous Processing**: Utilizes asynchronous calls for improved performance and concurrency, allowing multiple LLMs to process information simultaneously and reducing overall response times.
- **Broad Model Support**: Compatible with a wide range of LLM providers and models, offering flexibility in choosing the most suitable models for specific tasks or domains.
- **Customizable Prompts**: System prompts for individual neurons and aggregation prompts can be tailored for specific use cases, allowing for task-specific optimization and consistent output formatting.
- **Dynamic Response Handling**: Features like shuffling and dropout in the aggregation layer introduce controlled randomness, potentially improving output diversity and robustness.
- **Detailed Performance Metrics**: Provides comprehensive timing information for each step of the process, enabling performance analysis and optimization of the model architecture.

## System Architecture


moa-llm's architecture consists of several key components that work together to process queries:


1. **UserQueryAnnotator**: An optional component that pre-processes and optimizes user queries. It can expand, reformulate, or add context to the original query to improve the performance of subsequent layers.
2. **Neuron**: An abstract base class representing the fundamental processing units in the network. It defines the interface for all types of neurons in the system.
3. **LLMNeuron**: A concrete implementation of a Neuron that encapsulates an LLM. It handles the interaction with the LLM API, including sending prompts and receiving responses.
4. **Layer**: A collection of Neurons that process input in parallel. Layers can be composed of multiple LLMNeurons, potentially using different models or configurations.
5. **AggregationLayer**: A specialized Layer that combines outputs from previous layers. It supports advanced features like response shuffling and dropout to introduce controlled variability in the aggregation process.
6. **MixtureOfAgents**: The main orchestrator class that manages the entire query processing pipeline. It coordinates the flow of information through the UserQueryAnnotator, multiple Layer instances, and the final AggregationLayer.
7. **ConfigLoader**: A utility for loading and parsing configuration files, enabling easy setup and customization of the moa-llm architecture without code changes.

## Installation

If you want to install it as a local repo:
```bash
git clone https://github.com/catena-labs/moa.git
pip install -e .
```

## Setup

1. Create a `.env.local` file in your project root with your API key and OpenAI-compatible `base_url`. If you do not specify a `base_url`, Together AI will be used. For example:

   ```
   PROVIDER_API_KEY=your_api_key_here
   BASE_URL=https://api.together.xyz/v1

   ```

   Replace `your_api_key_here` with your actual Together AI API key.

2. Ensure `.env.local` is added to your `.gitignore` file to prevent accidentally committing sensitive information.

## Usage

Here's a basic example of how to set up and run moa-llm using a configuration file, demonstrating both single query and messages array inputs:

```python
import asyncio
from moa_llm import create_moa_from_config

async def run_moa():
    # Create MixtureOfAgents from a config file
    moa = create_moa_from_config('moa_config.yaml', is_file_path=True)

    # Process input
    user_query = "Explain the concept of quantum entanglement and its potential applications in quantum computing."
    result = await moa.process(user_query)

    # Print results
    print(result['content'])
    print(json.dumps(result['response_times'], indent=2))
    print(f"Total Completion Time: {result['total_completion_time']:.2f} seconds")

    # Example with messages array
    messages = [
        "What is the current state of quantum computing?",
        "Can you explain the concept of quantum entanglement?",
        "How does quantum computing differ from classical computing?"
    ]
    result = await moa.process(messages)
    print(result['content'])
    print(json.dumps(result['response_times'], indent=2))
    print(f"Total Completion Time: {result['total_completion_time']:.2f} seconds")

# Run moa-llm
asyncio.run(run_moa())
```

This example uses a configuration file (`moa_config.yaml`) to set up the moa-llm architecture. The configuration file allows for easy customization of the model structure, including the number and types of neurons, layer configurations, and aggregation settings.

## YAML Configuration

moa-llm uses YAML configuration files for easy setup and customization of the model architecture. This allows users to define complex multi-layer structures without modifying the code. Here's a detailed breakdown of the configuration options:

### Basic Structure

The configuration file consists of several main sections:

1. `use_annotator`: A boolean flag to enable or disable the query annotator.
2. `annotator`: Settings for the query annotator.
3. `proposal_layers`: An array of one or more proposal layers.
4. `aggregation_layer`: Settings for the final aggregation layer.

### Annotator Configuration

The annotator is an optional component that can preprocess and optimize user queries:

```yaml
use_annotator: true
annotator:
model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
temperature: 0.7
```


- `model`: Specifies the LLM to use for annotation.
- `temperature`: Controls the randomness of the model's output (0.0 to 1.0).

### Proposal Layers

You can define multiple proposal layers, each containing one or more neurons:

```yaml
proposal_layers:
  - neurons:
    - model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
      prompt: "You are a helpful assistant. Provide a concise response."
      temperature: 0.7
      weight: 8
    - model: "mistralai/Mixtral-8x22B-Instruct-v0.1"
      prompt: "You are a helpful assistant. Provide a concise response."
      temperature: 0.7
      weight: 7
  - neurons:
    - model: "Qwen/Qwen2-72B-Instruct"
      prompt: "You are an expert in the field. Provide a detailed analysis."
      temperature: 0.5
      weight: 9
```


For each neuron:
- `model`: Specifies the LLM to use.
- `prompt`: The system prompt for the model.
- `temperature`: Controls output randomness.
- `weight`: Determines the neuron's influence on the final result.

### Aggregation Layer

The aggregation layer combines outputs from the proposal layers:

```yaml
aggregation_layer:
  model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
  prompt: "You are an aggregator. Synthesize the following responses:"
  temperature: 0.7
  aggregation_prompt: |
    You are an advanced AI aggregator tasked with synthesizing multiple responses into a single, high-quality answer.
    USER QUERY: {user_query}
    RESPONSES:
    {responses}
    RESPONSE WEIGHTS:
    {response_weights}
  shuffle: true
  dropout_rate: 0.2
  ```

- `model`: The LLM used for aggregation.
- `prompt`: System prompt for the aggregator.
- `temperature`: Controls randomness of the aggregator's output.
- `aggregation_prompt`: A template for combining responses. It can include placeholders like `{user_query}`, `{responses}`, and `{response_weights}`.
- `shuffle`: When true, randomizes the order of input responses.
- `dropout_rate`: Probability (0.0 to 1.0) of dropping each input response.

### Advanced Features

1. **Multiple Proposal Layers**: You can define any number of proposal layers, allowing for complex, multi-step processing pipelines.

2. **Weighted Responses**: By assigning weights to neurons, you can control their influence on the final output.

3. **Customizable Prompts**: Each neuron and the aggregator can have tailored prompts, allowing for specialized roles within the network.

4. **Randomization**: The `shuffle` and `dropout_rate` options in the aggregation layer introduce controlled randomness, potentially improving output diversity and robustness.

### Example Usage

To use a YAML configuration file with moa-llm:
```python
from moa_llm import create_moa_from_config
moa = create_moa_from_config('moa_config.yaml', is_file_path=True)
result = await moa.process("Your query here")
```

This flexibility allows users to experiment with different architectures, model combinations, and processing strategies without changing the underlying code.

## Performance Considerations

- moa-llm uses asynchronous processing to improve performance, especially when dealing with multiple LLMs.
- Response times for each neuron and layer are logged, allowing for performance analysis and optimization.
- The total completion time for each query is calculated and reported.
- Shuffling and dropout in the aggregation layer can be used to introduce randomness and potentially improve the diversity of outputs.

## Extending the System

moa-llm is designed to be flexible and extensible. Some possible enhancements include:

- Adding support for more LLM providers and models.
- Implementing dynamic weight adjustment based on performance.
- Creating specialized layers for specific tasks (e.g., fact-checking, creativity enhancement).
- Integrating with other AI systems or data sources for enhanced capabilities.

## Limitations and Considerations

- The system's performance and output quality depend on the chosen LLMs and their respective capabilities.
- Proper prompt engineering is crucial for achieving optimal results.
- API costs may be a consideration when using multiple commercial LLM services.

## Contributing

Contributions to moa-llm are welcome! Please submit pull requests or open issues on the project's GitHub repository.

## License

MIT
