"""
tools.py — Tool integrations for the Multi-Agent Research System

Each tool is a LangChain @tool-decorated function that agents can call.
Tools are the "hands" of the agents — they interact with the outside world.

Tools defined here:
  - search_web: Tavily-powered web search (falls back to mock if no API key)
  - rag_retrieve: Local FAISS vector store retrieval
  - calculate: Safe expression evaluator for math tasks
"""

import os
import re
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


# ---------------------------------------------------------------------------
# Web Search Tool
# ---------------------------------------------------------------------------

def _build_search_tool():
    """Build the Tavily search tool, or return a mock if key is absent."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        return TavilySearchResults(max_results=3)
    return None


_tavily_search = _build_search_tool()


@tool
def search_web(query: str) -> str:
    """
    Search the web for up-to-date information on a given topic.

    Use this to find recent news, statistics, research findings, or any
    information that might not be in the local knowledge base.

    Args:
        query: A clear, specific search query (e.g. "AI impact on software jobs 2025")

    Returns:
        Formatted string of search results with snippets and URLs.
    """
    if _tavily_search is not None:
        try:
            results = _tavily_search.invoke(query)
            if not results:
                return f"[search_web] No results found for: {query}"
            formatted = []
            for r in results:
                title = r.get("title", "No title")
                url = r.get("url", "No URL")
                content = r.get("content", "")[:400]
                formatted.append(f"Source: {title}\nURL: {url}\nSnippet: {content}\n")
            return "\n---\n".join(formatted)
        except Exception as e:  # network / API errors
            return f"[search_web] Error during search: {e}"

    # ---- Fallback mock (no Tavily API key) --------------------------------
    mock_data = {
        "ai software engineering": (
            "AI tools like GitHub Copilot now generate ~46% of code "
            "in enabled repositories (GitHub Octoverse 2024). "
            "Developer productivity has increased 35% on average for "
            "routine coding tasks."
        ),
        "llm production": (
            "Best practices for production LLMs include: robust eval pipelines, "
            "output validation, rate limiting, caching frequent queries, "
            "and dedicated monitoring (LangSmith, Weights & Biases)."
        ),
        "langgraph": (
            "LangGraph is a framework for building stateful multi-agent systems "
            "as directed graphs. Key features: typed state, conditional edges, "
            "checkpointing, and streaming support."
        ),
        "multi-agent systems": (
            "Multi-agent architectures show 40% quality improvement over "
            "single-agent on complex research tasks (Stanford HELM 2024). "
            "Common patterns: Supervisor-Worker, Sequential Pipeline, "
            "and Graph-based workflows."
        ),
    }
    # Find best-matching mock result
    query_lower = query.lower()
    for key, value in mock_data.items():
        if any(word in query_lower for word in key.split()):
            return f"[MOCK SEARCH] {value}"
    return (
        f"[MOCK SEARCH] No specific data found for '{query}'. "
        "In production, this would return real web search results. "
        "Set TAVILY_API_KEY for live search."
    )


# ---------------------------------------------------------------------------
# RAG Retrieval Tool (uses the pipeline singleton injected at runtime)
# ---------------------------------------------------------------------------

# This module-level reference is populated by main.py after the RAG pipeline
# is initialized. Agents call rag_retrieve() without knowing the internals.
_rag_pipeline_instance = None


def set_rag_pipeline(pipeline):
    """Inject the initialized RAGPipeline instance so tools can use it."""
    global _rag_pipeline_instance
    _rag_pipeline_instance = pipeline


@tool
def rag_retrieve(query: str) -> str:
    """
    Retrieve relevant information from the local knowledge base using semantic search.

    Use this to look up domain-specific information, company policies, or any
    content that has been loaded into the local vector store.

    Args:
        query: A natural-language query describing what you're looking for.

    Returns:
        Top-K most relevant document chunks from the local knowledge base.
    """
    if _rag_pipeline_instance is None:
        return (
            "[rag_retrieve] RAG pipeline not initialized. "
            "Call set_rag_pipeline() before using this tool."
        )
    try:
        return _rag_pipeline_instance.retrieve(query)
    except Exception as e:
        return f"[rag_retrieve] Retrieval error: {e}"


# ---------------------------------------------------------------------------
# Calculator Tool
# ---------------------------------------------------------------------------

# Allowed characters for safe eval — digits, operators, parens, dots
_SAFE_PATTERN = re.compile(r"^[\d\s\+\-\*\/\.\(\)\%\*\*]+$")


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Use this for arithmetic, percentages, or numerical reasoning.
    Supports: +, -, *, /, **, %, parentheses, and decimal numbers.

    Args:
        expression: A mathematical expression string (e.g. "3.14 * 5**2")

    Returns:
        The numeric result as a string, or an error message.

    Examples:
        calculate("100 * 1.35")    → "135.0"
        calculate("(42 + 58) / 5") → "20.0"
    """
    # Strip whitespace for validation
    clean = expression.strip()
    if not _SAFE_PATTERN.match(clean):
        return (
            f"[calculate] Unsafe expression rejected: '{expression}'. "
            "Only numeric characters and operators (+,-,*,/,**,%,.) are allowed."
        )
    try:
        result = eval(clean, {"__builtins__": {}})  # no builtins = safe sandbox
        return str(result)
    except ZeroDivisionError:
        return "[calculate] Error: Division by zero."
    except Exception as e:
        return f"[calculate] Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Tool Registry (used by agents to bind tools to LLM)
# ---------------------------------------------------------------------------

ALL_TOOLS = [search_web, rag_retrieve, calculate]
RESEARCHER_TOOLS = [search_web, rag_retrieve]
ANALYST_TOOLS = [calculate]
