"""
LangChain agent with tools example.

This example demonstrates how to create an agent that can use tools
to perform tasks like calculations and web searches.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2")
    
    Returns:
        The result of the calculation
    """
    try:
        # Use a safe subset of operations
        import ast
        import operator
        
        # Allowed operators for safe evaluation
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            elif isinstance(node, ast.Num):  # Fallback for older Python
                return node.n
            elif isinstance(node, ast.BinOp):
                return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](eval_expr(node.operand))
            else:
                raise ValueError(f"Unsupported operation: {type(node).__name__}")
        
        node = ast.parse(expression, mode='eval')
        result = eval_expr(node.body)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """
    Count the number of words in a text.
    
    Args:
        text: The text to count words in
    
    Returns:
        The word count
    """
    word_count = len(text.split())
    return f"The text contains {word_count} words."


@tool
def string_reverser(text: str) -> str:
    """
    Reverse a string.
    
    Args:
        text: The text to reverse
    
    Returns:
        The reversed text
    """
    return f"Reversed: {text[::-1]}"


def create_tool_agent():
    """Create an agent with tools."""
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Define the tools
    tools = [calculator, word_counter, string_reverser]
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools to help answer questions."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


def main():
    """Run the tool agent example."""
    print("=== LangChain Agent with Tools Example ===\n")
    
    # Create the agent
    agent = create_tool_agent()
    
    # Example tasks
    tasks = [
        "What is 157 * 23?",
        "Count the words in this sentence: 'LangChain is a powerful framework for building AI applications.'",
        "Reverse the string 'Hello World'",
    ]
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print('='*60)
        result = agent.invoke({"input": task})
        print(f"\nFinal Answer: {result['output']}\n")


if __name__ == "__main__":
    main()
