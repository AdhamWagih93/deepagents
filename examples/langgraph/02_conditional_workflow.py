"""
LangGraph conditional workflow example.

This example demonstrates how to create a workflow with conditional branching
based on the state of the system.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()


class RouterState(TypedDict):
    """State for the routing workflow."""
    user_query: str
    query_type: str
    response: str


def classify_query(state: RouterState) -> RouterState:
    """Classify the type of user query."""
    print("\n--- Classifying Query ---")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = f"""Classify the following query into one of these categories:
- 'math' for mathematical questions
- 'code' for programming questions
- 'general' for general knowledge questions

Query: {state['user_query']}

Return only the category name, nothing else."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    query_type = response.content.strip().lower()
    
    state["query_type"] = query_type
    print(f"Query Type: {query_type}")
    return state


def handle_math_query(state: RouterState) -> RouterState:
    """Handle mathematical queries."""
    print("\n--- Handling Math Query ---")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = f"""You are a math expert. Solve this problem step by step:

{state['user_query']}

Provide a clear, detailed solution."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state["response"] = response.content
    print(f"Response: {state['response']}")
    return state


def handle_code_query(state: RouterState) -> RouterState:
    """Handle programming queries."""
    print("\n--- Handling Code Query ---")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = f"""You are a programming expert. Answer this question with code examples:

{state['user_query']}

Provide clear explanations and well-commented code."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state["response"] = response.content
    print(f"Response: {state['response']}")
    return state


def handle_general_query(state: RouterState) -> RouterState:
    """Handle general queries."""
    print("\n--- Handling General Query ---")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = f"""You are a helpful assistant. Answer this question clearly and concisely:

{state['user_query']}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state["response"] = response.content
    print(f"Response: {state['response']}")
    return state


def route_query(state: RouterState) -> Literal["math", "code", "general"]:
    """Route to the appropriate handler based on query type."""
    return state["query_type"]


def create_conditional_workflow():
    """Create a workflow with conditional routing."""
    # Create the graph
    workflow = StateGraph(RouterState)
    
    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("math", handle_math_query)
    workflow.add_node("code", handle_code_query)
    workflow.add_node("general", handle_general_query)
    
    # Define the flow
    workflow.set_entry_point("classify")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classify",
        route_query,
        {
            "math": "math",
            "code": "code",
            "general": "general"
        }
    )
    
    # All handlers lead to END
    workflow.add_edge("math", END)
    workflow.add_edge("code", END)
    workflow.add_edge("general", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def main():
    """Run the conditional workflow example."""
    print("=== LangGraph Conditional Workflow Example ===\n")
    
    # Create the workflow
    app = create_conditional_workflow()
    
    # Example queries
    queries = [
        "What is the derivative of x^2?",
        "How do I implement a binary search in Python?",
        "What is the capital of France?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = app.invoke({
            "user_query": query,
            "query_type": "",
            "response": ""
        })
        
        print(f"\n=== Final Response ===")
        print(result["response"])
        print()


if __name__ == "__main__":
    main()
