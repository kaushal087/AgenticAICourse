"""
main.py — Entry Point for the Multi-Agent Research System

This script:
  1. Loads environment variables (OPENAI_API_KEY, TAVILY_API_KEY)
  2. Initializes the RAG pipeline with the local knowledge base
  3. Builds and compiles the LangGraph agent workflow
  4. Runs the multi-agent pipeline on a sample query
  5. Streams and prints intermediate results + final report

Usage:
    # From the project root:
    cd code/
    python main.py
    
    # Or with a custom query:
    python main.py --query "What are the best practices for LLM deployment?"

Environment variables:
    OPENAI_API_KEY  — Required for LLM calls
    TAVILY_API_KEY  — Optional, enables live web search (falls back to mock)
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load .env file from parent directory or current directory
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def build_graph():
    """
    Construct and compile the multi-agent LangGraph workflow.

    Graph topology:
        planner → researcher → analyst → [conditional edge]
                                              ├── "continue" → researcher (loop)
                                              └── "done"     → writer → END

    Returns:
        A compiled LangGraph application ready to invoke/stream.
    """
    from agent import (
        GraphState,
        planner_node,
        researcher_node,
        analyst_node,
        writer_node,
        should_continue,
    )

    # ---- Define the graph ------------------------------------------------
    workflow = StateGraph(GraphState)

    # Add agent nodes — each is a plain Python function
    workflow.add_node("planner",    planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst",    analyst_node)
    workflow.add_node("writer",     writer_node)

    # ---- Wire up the edges -----------------------------------------------
    workflow.set_entry_point("planner")

    # Planner always goes to Researcher first
    workflow.add_edge("planner", "researcher")

    # Researcher always goes to Analyst for quality check
    workflow.add_edge("researcher", "analyst")

    # Analyst decides: loop for more research, or write the report
    workflow.add_conditional_edges(
        "analyst",           # source node
        should_continue,     # routing function  → returns "continue" or "done"
        {
            "continue": "researcher",   # loop: analyst → researcher
            "done":     "writer",       # proceed: analyst → writer
        },
    )

    # Writer is the terminal node
    workflow.add_edge("writer", END)

    return workflow.compile()


def run_pipeline(query: str, knowledge_base_path: str | None = None) -> str:
    """
    Execute the complete multi-agent research pipeline.

    Args:
        query:               The research question to answer.
        knowledge_base_path: Optional path to a text file for RAG ingestion.

    Returns:
        The final research report as a markdown string.
    """
    # ---- Initialize RAG pipeline -----------------------------------------
    from rag_pipeline import RAGPipeline
    from tools import set_rag_pipeline

    rag = RAGPipeline()

    # Try to load the sample knowledge base
    kb_path = knowledge_base_path or os.path.join(
        os.path.dirname(__file__), "..", "demo", "sample_data", "knowledge_base.txt"
    )
    if os.path.exists(kb_path):
        rag.ingest(kb_path)
    else:
        # Seed with some inline content so RAG works even without a file
        rag.ingest_text(
            """
LangGraph is a framework for building stateful multi-agent systems as directed graphs.
Key features: typed state (TypedDict), conditional edges for routing, 
checkpointing for resumable workflows, and streaming for real-time output.

Multi-agent systems divide complex tasks across specialized agents, improving 
quality, parallelism, and maintainability compared to single-agent approaches.

The ReAct pattern (Reasoning + Acting) is the foundation of most agents:
Thought → Action → Observation → repeat until goal is achieved.

RAG (Retrieval-Augmented Generation) enhances LLM responses by grounding them
in retrieved documents, reducing hallucination on knowledge-intensive tasks.
            """,
            source="inline-knowledge"
        )
        print("[RAG] Using inline knowledge base (knowledge_base.txt not found)")

    # Inject the RAG pipeline into the tools module
    set_rag_pipeline(rag)

    # ---- Build the graph -------------------------------------------------
    print("\n" + "="*60)
    print("🤖 MULTI-AGENT RESEARCH SYSTEM")
    print("="*60)
    print(f"📝 Query: {query}")
    print("="*60)

    app = build_graph()

    # ---- Initial state ---------------------------------------------------
    initial_state = {
        "query": query,
        "plan": [],
        "research_results": [],
        "additional_queries": [],
        "analysis": "",
        "final_report": "",
        "iteration_count": 0,
        "approved": False,
    }

    # ---- Stream execution ------------------------------------------------
    final_report = ""

    for event in app.stream(initial_state):
        for node_name, node_output in event.items():
            # Print state updates for each node (demo transparency)
            if "final_report" in node_output and node_output["final_report"]:
                final_report = node_output["final_report"]

    # ---- Output ----------------------------------------------------------
    print("\n" + "="*60)
    print("📄 FINAL REPORT")
    print("="*60)
    print(final_report)
    print("="*60)

    return final_report


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System — LangGraph Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --query "What are the top use cases for LLMs in fintech?"
  python main.py --query "Explain LangGraph vs CrewAI tradeoffs" --kb ../demo/sample_data/knowledge_base.txt
        """,
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the best practices for building production-ready LLM applications in 2025?",
        help="The research question to investigate",
    )
    parser.add_argument(
        "--kb",
        type=str,
        default=None,
        help="Path to a text file to use as the local knowledge base",
    )
    args = parser.parse_args()

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable is not set.")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    run_pipeline(query=args.query, knowledge_base_path=args.kb)


if __name__ == "__main__":
    main()
