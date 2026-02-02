#!/usr/bin/env python3
"""
DeepAgents Example Runner

This script helps you easily run any example in the playground.
"""

import sys
import os
import subprocess
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


def build_example_map():
    """Build a mapping from numbers to examples."""
    example_map = {}
    num = 1
    
    for category in ["langchain", "langgraph", "deepagents"]:
        for key in sorted(EXAMPLES[category].keys()):
            example_map[num] = EXAMPLES[category][key]
            num += 1
    
    return example_map


def print_menu():
    """Print the main menu."""
    print("\n" + "="*70)
    print("DeepAgents Playground - Example Runner")
    print("="*70 + "\n")
    
    example_map = build_example_map()
    
    current_category = None
    categories = {
        1: "LangChain Examples:",
        3: "\nLangGraph Examples:",
        5: "\nDeep Agents Examples:"
    }
    
    for num, example in example_map.items():
        if num in categories:
            print(categories[num])
        print(f"  {num}. {example['name']}")
        print(f"     {example['description']}")
    
    print("\n  0. Exit")
    print("\n" + "="*70)


def run_example(example_info):
    """Run a specific example."""
    print(f"\nRunning: {example_info['name']}")
    print(f"Description: {example_info['description']}")
    print("-" * 70 + "\n")
    
    # Execute the file safely using subprocess
    file_path = example_info['file']
    
    try:
        subprocess.run([sys.executable, file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running example: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Example interrupted by user.")


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
    
    example_map = build_example_map()
    max_choice = len(example_map)
    
    while True:
        print_menu()
        
        try:
            choice = input(f"\nEnter your choice (0-{max_choice}): ").strip()
            
            if choice == "0":
                print("\nThanks for using DeepAgents Playground! 👋")
                break
            
            choice_num = int(choice)
            
            if choice_num < 1 or choice_num > max_choice:
                print(f"\n❌ Invalid choice. Please enter a number between 0 and {max_choice}.")
                continue
            
            example = example_map.get(choice_num)
            
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
