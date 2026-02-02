# DeepAgents 🤖

A comprehensive playground for exploring LangChain, LangGraph, and deep agent systems. This repository provides hands-on examples demonstrating various AI agent architectures and workflows.

## 🌟 Features

- **LangChain Examples**: Basic chatbots, tool-using agents, and more
- **LangGraph Workflows**: State machines, conditional routing, and complex workflows
- **Deep Agents**: Reasoning agents, multi-agent systems, and collaborative AI
- **Ready-to-Run**: All examples are self-contained and easy to execute

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API key (or other LLM provider API keys)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/AdhamWagih93/deepagents.git
cd deepagents
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=your-openai-api-key-here
```

## 📚 Examples

### LangChain Examples

Located in `examples/langchain/`:

1. **Basic Chatbot** (`01_basic_chatbot.py`)
   - Simple conversational AI using LangChain
   - Demonstrates basic chat model usage
   - System prompts and context management

   ```bash
   python examples/langchain/01_basic_chatbot.py
   ```

2. **Agent with Tools** (`02_agent_with_tools.py`)
   - Tool-using agent that can perform calculations
   - Custom tool creation and integration
   - Agent reasoning and execution

   ```bash
   python examples/langchain/02_agent_with_tools.py
   ```

### LangGraph Examples

Located in `examples/langgraph/`:

1. **Basic Workflow** (`01_basic_workflow.py`)
   - Sequential state machine workflow
   - State management across steps
   - Multi-step processing pipeline

   ```bash
   python examples/langgraph/01_basic_workflow.py
   ```

2. **Conditional Workflow** (`02_conditional_workflow.py`)
   - Conditional branching based on state
   - Dynamic routing to specialized handlers
   - Query classification and routing

   ```bash
   python examples/langgraph/02_conditional_workflow.py
   ```

### Deep Agents Examples

Located in `examples/deepagents/`:

1. **Reasoning Agent** (`01_reasoning_agent.py`)
   - Chain-of-thought reasoning
   - Step-by-step problem solving
   - Complex problem decomposition

   ```bash
   python examples/deepagents/01_reasoning_agent.py
   ```

2. **Multi-Agent System** (`02_multi_agent_system.py`)
   - Collaborative agent system
   - Specialized agents (Research, Analysis, Writing)
   - Agent orchestration and coordination

   ```bash
   python examples/deepagents/02_multi_agent_system.py
   ```

## 🏗️ Project Structure

```
deepagents/
├── examples/
│   ├── langchain/          # LangChain examples
│   │   ├── 01_basic_chatbot.py
│   │   └── 02_agent_with_tools.py
│   ├── langgraph/          # LangGraph examples
│   │   ├── 01_basic_workflow.py
│   │   └── 02_conditional_workflow.py
│   └── deepagents/         # Deep agent examples
│       ├── 01_reasoning_agent.py
│       └── 02_multi_agent_system.py
├── .env.example            # Example environment variables
├── .gitignore             # Git ignore rules
├── pyproject.toml         # Project configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Development

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

### Code Formatting

```bash
black .
ruff check .
```

## 📖 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add new examples
- Improve existing code
- Fix bugs
- Enhance documentation

## 📝 License

This project is open source and available for educational and experimental purposes.

## ⚠️ Notes

- Examples require API keys from LLM providers (e.g., OpenAI)
- Some examples use GPT-4 which may incur higher costs
- All examples are for educational purposes
- Ensure you handle API keys securely and never commit them to version control

## 🎯 Next Steps

After running the examples, consider:

1. Modifying prompts to see different behaviors
2. Adding your own custom tools
3. Creating more complex workflows
4. Experimenting with different LLM models
5. Building your own multi-agent systems

Happy exploring! 🚀
