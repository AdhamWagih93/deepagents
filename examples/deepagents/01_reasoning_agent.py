"""
Reasoning agent example.

This example demonstrates a deep agent that can reason through
complex problems using chain-of-thought prompting.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()


class ReasoningAgent:
    """An agent that uses chain-of-thought reasoning."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.system_prompt = """You are an expert reasoning agent. When given a problem:
1. Break down the problem into smaller steps
2. Think through each step carefully
3. Consider multiple approaches
4. Arrive at a well-reasoned conclusion

Always explain your reasoning process."""
    
    def reason(self, problem: str) -> dict:
        """
        Reason through a problem.
        
        Args:
            problem: The problem to solve
            
        Returns:
            A dictionary with reasoning steps and conclusion
        """
        print(f"\n=== Reasoning Agent ===")
        print(f"Problem: {problem}\n")
        
        # Step 1: Understand the problem
        understand_prompt = f"{self.system_prompt}\n\nFirst, rephrase this problem to ensure you understand it:\n{problem}"
        understanding = self.llm.invoke([HumanMessage(content=understand_prompt)])
        
        print("Step 1 - Understanding:")
        print(understanding.content)
        print()
        
        # Step 2: Break down the problem
        breakdown_prompt = f"{self.system_prompt}\n\nBreak this problem into steps:\n{problem}"
        breakdown = self.llm.invoke([HumanMessage(content=breakdown_prompt)])
        
        print("Step 2 - Breakdown:")
        print(breakdown.content)
        print()
        
        # Step 3: Solve step by step
        solve_prompt = f"""{self.system_prompt}

Problem: {problem}

Understanding: {understanding.content}

Steps: {breakdown.content}

Now solve the problem step by step, showing your work."""
        
        solution = self.llm.invoke([HumanMessage(content=solve_prompt)])
        
        print("Step 3 - Solution:")
        print(solution.content)
        print()
        
        return {
            "problem": problem,
            "understanding": understanding.content,
            "breakdown": breakdown.content,
            "solution": solution.content
        }


def main():
    """Run the reasoning agent example."""
    print("=== Deep Agents: Reasoning Agent Example ===\n")
    
    # Create the agent
    agent = ReasoningAgent()
    
    # Example problems
    problems = [
        """A farmer has chickens and rabbits. There are 20 heads and 56 legs total.
        How many chickens and how many rabbits are there?""",
        
        """If it takes 5 machines 5 minutes to make 5 widgets,
        how long would it take 100 machines to make 100 widgets?"""
    ]
    
    for problem in problems:
        print("\n" + "="*80)
        result = agent.reason(problem)
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
