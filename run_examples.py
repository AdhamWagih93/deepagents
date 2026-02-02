#!/usr/bin/env python3
"""
DeepAgents Example Runner

This script helps you easily run any example in the playground.
"""

import sys
import os
from pathlib import Path


EXAMPLES = {
    "langchain": {
        "1": {
            "name": "Basic Chatbot",
            "file": "examples/langchain/01_basic_chatbot.py",
            "description": "Simple conversational AI using LangChain"
        },
        "2": {
            "name": "Agent with Tools",
            "file": "examples/langchain/02_agent_with_tools.py",
            "description": "Tool-using agent that can perform calculations"
        }
    },
    "langgraph": {
        "1": {
            "name": "Basic Workflow",
            "file": "examples/langgraph/01_basic_workflow.py",
            "description": "Sequential state machine workflow"
        },
        "2": {
            "name": "Conditional Workflow",
            "file": "examples/langgraph/02_conditional_workflow.py",
            "description": "Conditional branching and dynamic routing"
        }
    },
    "deepagents": {
        "1": {
            "name": "Reasoning Agent",
            "file": "examples/deepagents/01_reasoning_agent.py",
            "description": "Chain-of-thought reasoning agent"
        },
        "2": {
            "name": "Multi-Agent System",
            "file": "examples/deepagents/02_multi_agent_system.py",
            "description": "Collaborative multi-agent system"
        }
    }
}


def print_menu():
    """Print the main menu."""
    print("\n" + "="*70)
    print("DeepAgents Playground - Example Runner")
    print("="*70 + "\n")
    
    print("LangChain Examples:")
    for key, example in EXAMPLES["langchain"].items():
        print(f"  {key}. {example['name']}")
        print(f"     {example['description']}")
    
    print("\nLangGraph Examples:")
    for key, example in EXAMPLES["langgraph"].items():
        print(f"  {int(key) + 2}. {example['name']}")
        print(f"     {example['description']}")
    
    print("\nDeep Agents Examples:")
    for key, example in EXAMPLES["deepagents"].items():
        print(f"  {int(key) + 4}. {example['name']}")
        print(f"     {example['description']}")
    
    print("\n  0. Exit")
    print("\n" + "="*70)


def get_example_by_number(num: int):
    """Get example information by number."""
    if num == 1:
        return EXAMPLES["langchain"]["1"]
    elif num == 2:
        return EXAMPLES["langchain"]["2"]
    elif num == 3:
        return EXAMPLES["langgraph"]["1"]
    elif num == 4:
        return EXAMPLES["langgraph"]["2"]
    elif num == 5:
        return EXAMPLES["deepagents"]["1"]
    elif num == 6:
        return EXAMPLES["deepagents"]["2"]
    return None


def run_example(example_info):
    """Run a specific example."""
    print(f"\nRunning: {example_info['name']}")
    print(f"Description: {example_info['description']}")
    print("-" * 70 + "\n")
    
    # Import and run the example
    file_path = example_info['file']
    
    # Execute the file
    os.system(f"python {file_path}")


def check_environment():
    """Check if the environment is properly set up."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("\n⚠️  Warning: .env file not found!")
        print("Please copy .env.example to .env and add your API keys.")
        print("\nYou can continue, but examples will fail without proper API keys.\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True


def main():
    """Main function."""
    if not check_environment():
        return
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == "0":
                print("\nThanks for using DeepAgents Playground! 👋")
                break
            
            choice_num = int(choice)
            
            if choice_num < 1 or choice_num > 6:
                print("\n❌ Invalid choice. Please enter a number between 0 and 6.")
                continue
            
            example = get_example_by_number(choice_num)
            
            if example:
                run_example(example)
                input("\nPress Enter to continue...")
            else:
                print("\n❌ Example not found.")
        
        except ValueError:
            print("\n❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nThanks for using DeepAgents Playground! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
