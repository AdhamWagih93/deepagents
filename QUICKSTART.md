# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Your API Key

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run Your First Example

#### Option A: Using the Interactive Runner

```bash
python run_examples.py
```

Then select an example from the menu!

#### Option B: Run Examples Directly

**LangChain - Basic Chatbot:**
```bash
python examples/langchain/01_basic_chatbot.py
```

**LangChain - Agent with Tools:**
```bash
python examples/langchain/02_agent_with_tools.py
```

**LangGraph - Basic Workflow:**
```bash
python examples/langgraph/01_basic_workflow.py
```

**LangGraph - Conditional Workflow:**
```bash
python examples/langgraph/02_conditional_workflow.py
```

**Deep Agents - Reasoning Agent:**
```bash
python examples/deepagents/01_reasoning_agent.py
```

**Deep Agents - Multi-Agent System:**
```bash
python examples/deepagents/02_multi_agent_system.py
```

## What to Try

### 1. Start with LangChain Basics
Learn the fundamentals with the basic chatbot example.

### 2. Explore Agent Tools
See how agents can use tools to perform tasks.

### 3. Understand LangGraph Workflows
Learn about state management and workflow orchestration.

### 4. Experiment with Deep Agents
Explore advanced reasoning and multi-agent collaboration.

## Troubleshooting

### "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "AuthenticationError: No API key provided"
Make sure you've:
1. Created a `.env` file (copy from `.env.example`)
2. Added your OpenAI API key to the `.env` file

### "Rate limit exceeded"
If using the free tier, you may hit rate limits. Wait a moment and try again.

## Next Steps

1. **Modify the Examples**: Try changing prompts, temperatures, or system messages
2. **Create Custom Tools**: Add your own tools to the agent examples
3. **Design Workflows**: Create more complex LangGraph workflows
4. **Build Multi-Agent Systems**: Expand the multi-agent example with more specialized agents

## Resources

- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Docs](https://platform.openai.com/docs)

Happy coding! 🚀
