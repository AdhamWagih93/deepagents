"""
Basic LangGraph workflow example.

This example demonstrates how to create a simple state machine workflow
using LangGraph with sequential steps.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()


class WorkflowState(TypedDict):
    """State for the workflow."""
    input: str
    summary: str
    analysis: str
    output: str


def summarize_step(state: WorkflowState) -> WorkflowState:
    """Summarize the input text."""
    print("\n--- Step 1: Summarizing ---")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = f"Summarize the following text in 2-3 sentences:\n\n{state['input']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["summary"] = response.content
    print(f"Summary: {state['summary']}")
    return state


def analyze_step(state: WorkflowState) -> WorkflowState:
    """Analyze the summary."""
    print("\n--- Step 2: Analyzing ---")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    prompt = f"Analyze the sentiment and key themes in this summary:\n\n{state['summary']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["analysis"] = response.content
    print(f"Analysis: {state['analysis']}")
    return state


def format_output_step(state: WorkflowState) -> WorkflowState:
    """Format the final output."""
    print("\n--- Step 3: Formatting Output ---")
    
    output = f"""
=== Document Processing Results ===

Original Input:
{state['input'][:100]}...

Summary:
{state['summary']}

Analysis:
{state['analysis']}
"""
    state["output"] = output
    print(output)
    return state


def create_workflow():
    """Create a sequential workflow using LangGraph."""
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("summarize", summarize_step)
    workflow.add_node("analyze", analyze_step)
    workflow.add_node("format", format_output_step)
    
    # Define the flow
    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "analyze")
    workflow.add_edge("analyze", "format")
    workflow.add_edge("format", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def main():
    """Run the basic workflow example."""
    print("=== LangGraph Basic Workflow Example ===\n")
    
    # Create the workflow
    app = create_workflow()
    
    # Example input
    sample_text = """
    Artificial Intelligence has transformed the way we interact with technology.
    Machine learning models can now understand natural language, generate creative content,
    and assist in complex decision-making processes. LangChain and LangGraph are frameworks
    that make it easier to build sophisticated AI applications by providing tools for
    chaining LLM calls, managing state, and creating complex workflows. These tools are
    particularly useful for building agents that can reason, plan, and execute tasks
    autonomously.
    """
    
    # Run the workflow
    result = app.invoke({
        "input": sample_text,
        "summary": "",
        "analysis": "",
        "output": ""
    })
    
    print("\n=== Workflow Complete ===")


if __name__ == "__main__":
    main()
