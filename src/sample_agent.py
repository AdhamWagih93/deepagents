from typing import List
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
### LLM handling for Ollama
from langchain_ollama.chat_models import ChatOllama
### Message handling
from langchain.messages import HumanMessage, AIMessage
### Tool handling
from langchain.tools import tool, ToolRuntime
#from langchain_mcp_adapters import MultiServerMCPClient --> Not working for some reason
### Langchain Agent creation
from langchain.agents import create_agent
### Built-in Middleware imports
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware import LLMToolEmulator
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from langchain.agents.middleware import ShellToolMiddleware, HostExecutionPolicy, DockerExecutionPolicy
from langchain.agents.middleware import FilesystemFileSearchMiddleware
### Short-term in-memory imports
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
 
# -------- Tools --------
@tool
def write_json(filepath: str, data: dict) -> str:
    """Write a Python dictionary as JSON to a file with pretty formatting."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Successfully wrote JSON data to '{filepath}' ({len(json.dumps(data))} characters)."
    except Exception as e:
        return f"Error writing JSON: {str(e)}"
 
 
@tool
def read_json(filepath: str) -> str:
    """Read and return the contents of a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error reading JSON: {str(e)}"
 
 
@tool
def generate_sample_users(
        first_names: List[str],
        last_names: List[str],
        domains: List[str],
        min_age: int,
        max_age: int
) -> dict:
    """
    Generate sample user data. Count is determined by the length of first_names.
    """
    if not first_names:
        return {"error": "first_names list cannot be empty"}
    if not last_names:
        return {"error": "last_names list cannot be empty"}
    if not domains:
        return {"error": "domains list cannot be empty"}
    if min_age > max_age:
        return {"error": f"min_age ({min_age}) cannot be greater than max_age ({max_age})"}
    if min_age < 0 or max_age < 0:
        return {"error": "ages must be non-negative"}
 
    users = []
    count = len(first_names)
 
    for i in range(count):
        first = first_names[i]
        last = last_names[i % len(last_names)]
        domain = domains[i % len(domains)]
        email = f"{first.lower()}.{last.lower()}@{domain}"
 
        user = {
            "id": i + 1,
            "firstName": first,
            "lastName": last,
            "email": email,
            "username": f"{first.lower()}{random.randint(100, 999)}",
            "age": random.randint(min_age, max_age),
            "registeredAt": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
        }
        users.append(user)
 
    return {"users": users, "count": len(users)}
 
 
tools = [write_json, read_json, generate_sample_users]
 
### Model definitions
model = ChatOllama(
    model="qwen3-coder:latest",  # or any Ollama model you pulled
    base_url="http://ef-nexus-02.efinance.com.eg:8081",
    temperature=0,
    num_ctx=50000,
)
 
model3 = ChatOllama(
    model="qwen2.5:7b-instruct-q6_K",  # or any Ollama model you pulled
    base_url="http://ef-nexus-02.efinance.com.eg:8081",
    temperature=0,
    num_ctx=50000,
)
 
model2 = ChatOllama(
    model="llama3.1:8b",  # or any Ollama model you pulled
    base_url="http://ef-nexus-02.efinance.com.eg:8081",
    temperature=0,
    num_ctx=50000,
)
 
SYSTEM_MESSAGE = (
    "You are a helpful assistant."
)
 
### Setup short-term memory in-memory
inmemory_checkpointer=InMemorySaver()
 
### Setup short-term memory in Postgres (Not working for some reason)
#DB_URI = "postgresql://{db_username}:{db_password}@{db_hostname}:{db_port}/{db_name}?sslmode=disable"
#with PostgresSaver.from_conn_string(DB_URI) as pg_checkpointer:
#    pg_checkpointer.setup()
 
### Forcing context variables
@dataclass
class UserContext:
    project: str
    environment: str
    user_role: str
 
 
### Create the agent
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        ### Automatically summarize
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 20000),
            keep=("messages", 20),
        ),
        ### Human approval
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_json": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "generate_sample_users": True,
            }
        ),
        ### Model calling limit
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
        ### Global tool calling limit
        ToolCallLimitMiddleware(
            thread_limit=20,
            run_limit=10,
        ),
        ### Tool-specific calling limit
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3,
        ),
        ### Model fallback instead of failure
        ModelFallbackMiddleware(
            model2,
            model3,
        ),
        ### Personal Identifiable Info handling
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        ### Maintain a to-do list
        TodoListMiddleware(),
        LLMToolSelectorMiddleware(
            model=model,
            max_tools=5,
            always_include=["glob_search","grep_search"],
        ),
        ### Tool retry mechanism
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            #tools=[tool1,tool2],
        ),
        ### Model retry mechanism
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
        ### Emulate specific tools instead of actually calling them
        LLMToolEmulator(
            model=model,
            tools=[generate_sample_users],
        ),
        ### Edit context at runtime
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=10000,
                    keep=5,
                    #exclude_tools=[tool1,tool2],
                    placeholder="[cleared]",
                ),
            ],
        ),
        ### Execute shell commands inside the environment
        ShellToolMiddleware(
            workspace_root="/app",
            execution_policy=HostExecutionPolicy(),
        ),
        ### Search file names and contents
        FilesystemFileSearchMiddleware(
            root_path="/app",
            use_ripgrep=True,
            max_file_size_mb=10,
        ),
    ],
    ### Store conversation history
    checkpointer=inmemory_checkpointer,
    ### Provide custom Context
    context_schema=UserContext,
    ### System prompt
    system_prompt=SYSTEM_MESSAGE,
)
 
 
### Testing the agent ###
test_query = "How many files are in current folder? and what are their names?"
test_query = "Does the file agent.py_allfeatures exist?"
test_query = "What is 2+2?"
test_query = "Based on existing agent.py implementation, create a streamlit.py that works as a comprehensive UI for interacting with this agent, covering Streamed output, humani in the loop approval, and visualizing responses (current and historical) beautifully."
test_query = "Delete file agent.py_allfeatures"
test_config = {"configurable": {"thread_id": "1"}}
test_context=UserContext(
        project="myproject",
        environment="dev",
        user_role="admin",
)
 
 
### Invoke Agent
print("Invoking Agent")
result = agent.invoke(
    {"messages": [{"role": "user", "content": test_query}]},
    config=test_config,
    context=test_context,
)
print(result)
 
### Stream Agent
print("Streaming Agent")
for chunk in agent.stream(
        {"messages": [{"role": "user", "content": test_query}]},
        config=test_config,
        context=test_context,
        stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data}")
        #print(f"content: {data['messages'][-1].content_blocks}")