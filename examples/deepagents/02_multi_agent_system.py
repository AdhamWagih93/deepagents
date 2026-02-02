"""
Multi-agent system example.

This example demonstrates a collaborative system where multiple specialized
agents work together to solve a complex task.
"""

import os
from dotenv import load_dotenv
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()


class ResearchAgent:
    """Agent specialized in research and information gathering."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.name = "Research Agent"
    
    def research(self, topic: str) -> str:
        """Research a topic and provide key information."""
        prompt = f"""You are a research specialist. Provide key facts and information about: {topic}
        
Focus on the most important and relevant information."""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert researcher."),
            HumanMessage(content=prompt)
        ])
        
        return response.content


class AnalysisAgent:
    """Agent specialized in analysis and critical thinking."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.name = "Analysis Agent"
    
    def analyze(self, information: str) -> str:
        """Analyze information and identify key insights."""
        prompt = f"""You are an analysis specialist. Analyze this information and identify:
1. Key patterns
2. Important insights
3. Potential implications

Information:
{information}"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert analyst."),
            HumanMessage(content=prompt)
        ])
        
        return response.content


class WriterAgent:
    """Agent specialized in writing and communication."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.8,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.name = "Writer Agent"
    
    def write(self, research: str, analysis: str, topic: str) -> str:
        """Write a comprehensive report based on research and analysis."""
        prompt = f"""You are a professional writer. Create a well-structured report about: {topic}

Research findings:
{research}

Analysis:
{analysis}

Write a clear, engaging report that combines the research and analysis."""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert writer and communicator."),
            HumanMessage(content=prompt)
        ])
        
        return response.content


class MultiAgentOrchestrator:
    """Orchestrator that coordinates multiple agents."""
    
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.writer_agent = WriterAgent()
    
    def collaborate(self, task: str) -> Dict[str, str]:
        """
        Coordinate multiple agents to complete a task.
        
        Args:
            task: The task description
            
        Returns:
            Results from each agent
        """
        print(f"\n{'='*80}")
        print(f"Multi-Agent System: {task}")
        print('='*80 + "\n")
        
        # Step 1: Research
        print(f"[{self.research_agent.name}] Starting research...")
        research_results = self.research_agent.research(task)
        print(f"[{self.research_agent.name}] Research complete.\n")
        print(f"Research Results:\n{research_results}\n")
        
        # Step 2: Analysis
        print(f"[{self.analysis_agent.name}] Analyzing research findings...")
        analysis_results = self.analysis_agent.analyze(research_results)
        print(f"[{self.analysis_agent.name}] Analysis complete.\n")
        print(f"Analysis Results:\n{analysis_results}\n")
        
        # Step 3: Writing
        print(f"[{self.writer_agent.name}] Writing final report...")
        final_report = self.writer_agent.write(research_results, analysis_results, task)
        print(f"[{self.writer_agent.name}] Report complete.\n")
        
        return {
            "task": task,
            "research": research_results,
            "analysis": analysis_results,
            "report": final_report
        }


def main():
    """Run the multi-agent system example."""
    print("=== Deep Agents: Multi-Agent System Example ===\n")
    
    # Create the orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Example tasks
    tasks = [
        "The impact of AI on healthcare",
        "Benefits and challenges of remote work"
    ]
    
    for task in tasks:
        result = orchestrator.collaborate(task)
        
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80)
        print(result["report"])
        print("\n")


if __name__ == "__main__":
    main()
