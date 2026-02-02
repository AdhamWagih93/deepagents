"""
Basic LangChain chatbot example.

This example demonstrates how to create a simple chatbot using LangChain
with OpenAI's GPT model.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()


def create_basic_chatbot():
    """Create a basic chatbot using LangChain."""
    # Initialize the chat model
    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return chat


def chat_with_bot(chat_model, user_message: str, system_prompt: str = None):
    """
    Send a message to the chatbot and get a response.
    
    Args:
        chat_model: The LangChain chat model instance
        user_message: The user's message
        system_prompt: Optional system prompt to set context
    
    Returns:
        The chatbot's response
    """
    messages = []
    
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    messages.append(HumanMessage(content=user_message))
    
    response = chat_model.invoke(messages)
    return response.content


def main():
    """Run the basic chatbot example."""
    print("=== LangChain Basic Chatbot Example ===\n")
    
    # Create the chatbot
    chatbot = create_basic_chatbot()
    
    # Define a system prompt
    system_prompt = "You are a helpful AI assistant that specializes in explaining AI concepts."
    
    # Example conversations
    questions = [
        "What is LangChain?",
        "How does it differ from traditional programming?",
        "Give me a simple use case example."
    ]
    
    for question in questions:
        print(f"User: {question}")
        response = chat_with_bot(chatbot, question, system_prompt)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
