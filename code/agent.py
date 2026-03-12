"""
agent.py — Agent Node Definitions for the Multi-Agent Research System

This module defines the four agent nodes used in the LangGraph workflow:

  1. planner_node    — Decomposes the user query into focused subtasks
  2. researcher_node — Gathers information for each subtask using tools
  3. analyst_node    — Evaluates research quality and synthesizes findings
  4. writer_node     — Produces the final structured report

Each node is a plain Python function that:
  - Takes a GraphState dict as input
  - Returns a dict with only the fields that changed
  - Has no side effects beyond updating state

Architecture note:
  Agents are deliberately stateless functions. All state lives in GraphState.
  This makes the system testable, resumable (via checkpointing), and debuggable.
"""

import json
from typing import TypedDict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from tools import RESEARCHER_TOOLS, ANALYST_TOOLS, search_web, rag_retrieve, calculate


# ---------------------------------------------------------------------------
# Shared State Definition
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """
    The single shared data structure that flows through every node in the graph.

    Think of this as the "whiteboard" in a team room — every agent can read
    anything on it and write their own updates.
    """
    query: str                          # Original user question
    plan: List[str]                     # Subtasks from Planner
    research_results: List[dict]        # Findings from Researcher
    additional_queries: List[str]       # Analyst-requested follow-up searches
    analysis: str                       # Synthesized insights from Analyst
    final_report: str                   # Final output from Writer
    iteration_count: int                # Guard against infinite loops
    approved: bool                      # Analyst's verdict on research quality


# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

# temperature=0 for Planner/Analyst: we want deterministic, structured output
# Writer gets temperature=0.3 for slightly more natural writing
_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_llm_writer = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Bind tools to the researcher LLM so it can call them via function calling
_llm_with_tools = _llm.bind_tools([search_web, rag_retrieve])


# ---------------------------------------------------------------------------
# Node 1: Planner Agent
# ---------------------------------------------------------------------------

def planner_node(state: GraphState) -> dict:
    """
    Decompose the user's query into 3-5 focused, searchable subtasks.

    Role:    Project Manager / Tech Lead
    Input:   state["query"]
    Output:  state["plan"] — list of research subtasks

    Instructor note:
      This is the "divide and conquer" step. A well-formed plan ensures each
      downstream agent works on a focused, answerable question rather than
      trying to answer everything at once.
    """
    print("\n🧠 [PLANNER] Analyzing query and creating research plan...")

    system_prompt = """You are an expert research planner and project manager.

Given a user's research question, break it down into 3-5 specific, 
searchable sub-questions that together will comprehensively answer it.

Rules:
- Each sub-question should be independently searchable
- Cover different angles: facts, data, expert opinions, trends
- Be specific enough to get focused search results
- Avoid overlap between sub-questions

Return ONLY a valid JSON object in this exact format:
{
    "plan": [
        "specific sub-question 1",
        "specific sub-question 2",
        "specific sub-question 3"
    ]
}"""

    response = _llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research Question: {state['query']}")
    ])

    try:
        # Parse the JSON response
        content = response.content.strip()
        # Strip markdown code fences if present (```json ... ```)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content.strip())
        plan = data["plan"]
    except (json.JSONDecodeError, KeyError) as e:
        # Graceful fallback: wrap the whole response as a single item
        print(f"  ⚠️  JSON parse error ({e}), using fallback plan")
        plan = [state["query"]]

    print(f"  ✅ Plan created with {len(plan)} subtasks:")
    for i, task in enumerate(plan, 1):
        print(f"     {i}. {task}")

    return {
        "plan": plan,
        "iteration_count": 0,
        "approved": False,
    }


# ---------------------------------------------------------------------------
# Node 2: Researcher Agent
# ---------------------------------------------------------------------------

def researcher_node(state: GraphState) -> dict:
    """
    Gather information for each subtask using web search and RAG retrieval.

    Role:    Senior Developer / Research Analyst
    Input:   state["plan"] + state["additional_queries"] (on subsequent iterations)
    Output:  state["research_results"] — list of {subtask, findings, sources}
    Tools:   search_web, rag_retrieve

    Instructor note:
      This is where the agent actively uses tools. Each subtask gets its own
      web search + RAG lookup. Notice the Researcher doesn't decide if the
      results are "good enough" — that's the Analyst's job (separation of concerns).
    """
    iteration = state.get("iteration_count", 0) + 1
    print(f"\n🔍 [RESEARCHER] Iteration {iteration} — Gathering information...")

    # On the first pass, use the plan; on subsequent passes, use analyst's follow-ups
    if iteration == 1 or not state.get("additional_queries"):
        tasks_to_research = state["plan"]
        print(f"  📋 Working through {len(tasks_to_research)} planned subtasks")
    else:
        tasks_to_research = state["additional_queries"]
        print(f"  🔄 Working through {len(tasks_to_research)} analyst-requested follow-ups")

    results = list(state.get("research_results", []))  # preserve existing results

    for i, subtask in enumerate(tasks_to_research, 1):
        print(f"\n  [{i}/{len(tasks_to_research)}] Researching: {subtask[:70]}...")

        # --- Web Search ---
        print(f"    🔧 Tool: search_web('{subtask[:50]}...')")
        web_findings = search_web.invoke(subtask)

        # --- RAG Retrieval ---
        print(f"    🔧 Tool: rag_retrieve('{subtask[:50]}...')")
        rag_findings = rag_retrieve.invoke(subtask)

        # --- Synthesize with LLM ---
        synthesis_prompt = f"""You are a research assistant. Synthesize the following 
search results into a clear, factual summary for this subtask:

Subtask: {subtask}

Web Search Results:
{web_findings}

Local Knowledge Base Results:
{rag_findings}

Write a 2-4 sentence factual summary. Include specific data points if available.
End with 'Sources: [list any URLs mentioned]'"""

        synthesis = _llm.invoke([HumanMessage(content=synthesis_prompt)])

        results.append({
            "subtask": subtask,
            "findings": synthesis.content,
            "web_raw": web_findings[:500],  # truncate for state efficiency
            "rag_raw": rag_findings[:500],
        })
        print(f"    ✅ Subtask {i} complete")

    return {
        "research_results": results,
        "iteration_count": iteration,
    }


# ---------------------------------------------------------------------------
# Node 3: Analyst Agent
# ---------------------------------------------------------------------------

def analyst_node(state: GraphState) -> dict:
    """
    Evaluate research quality and synthesize findings into key insights.

    Role:    Code Reviewer / Quality Analyst
    Input:   state["research_results"]
    Output:  state["analysis"], state["approved"], state["additional_queries"]

    Decision logic:
      - approved=True  → proceed to Writer
      - approved=False → loop back to Researcher with additional_queries

    Instructor note:
      This is the "self-correcting" node. Without it, the pipeline would
      blindly pass whatever the Researcher found to the Writer, even if
      coverage was incomplete. The Analyst is what makes the system reliable.
    """
    print("\n🧪 [ANALYST] Evaluating research quality...")

    results_text = "\n\n".join([
        f"Subtask: {r['subtask']}\nFindings: {r['findings']}"
        for r in state["research_results"]
    ])

    system_prompt = f"""You are a rigorous research quality analyst.

Original Query: {state['query']}

Review the research findings below and:
1. Identify the key themes and patterns across all findings
2. Assess coverage: does the research adequately address the original query?
3. Identify any critical gaps or conflicting information
4. Decide: is this research complete enough to write a high-quality report?

Research Findings:
{results_text}

Return ONLY a valid JSON object in this exact format:
{{
    "analysis": "2-4 paragraph synthesis of key findings and themes",
    "approved": true or false,
    "coverage_score": 0-100,
    "additional_queries": ["follow-up question if approved=false", "..."],
    "gaps": "brief description of what's missing, if anything"
}}

Set approved=true if coverage_score >= 70 or iteration_count >= 3."""

    response = _llm.invoke([HumanMessage(content=system_prompt)])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content.strip())

        analysis = data.get("analysis", "Synthesis not available.")
        approved = data.get("approved", False)
        coverage = data.get("coverage_score", 0)
        additional = data.get("additional_queries", [])
        gaps = data.get("gaps", "")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ⚠️  JSON parse error ({e}), approving by default")
        analysis = response.content
        approved = True
        coverage = 75
        additional = []
        gaps = ""

    # Force approval after max iterations to prevent infinite loops
    if state.get("iteration_count", 0) >= 3:
        approved = True
        print("  ⚠️  Max iterations reached — forcing approval")

    status = "✅ APPROVED" if approved else "❌ NEEDS MORE RESEARCH"
    print(f"  {status} (Coverage: {coverage}%)")
    if not approved and additional:
        print(f"  📎 Requesting {len(additional)} follow-up searches")
    if gaps:
        print(f"  📌 Gaps identified: {gaps[:100]}")

    return {
        "analysis": analysis,
        "approved": approved,
        "additional_queries": additional if not approved else [],
    }


# ---------------------------------------------------------------------------
# Node 4: Writer Agent
# ---------------------------------------------------------------------------

def writer_node(state: GraphState) -> dict:
    """
    Produce a well-structured, cited final report from the synthesized research.

    Role:    Technical Writer / Documentation Specialist
    Input:   state["query"], state["analysis"], state["research_results"]
    Output:  state["final_report"] — formatted Markdown report

    Instructor note:
      The Writer gets the highest-quality context: the Analyst's synthesis
      plus the original research results. It uses a slightly higher temperature
      (0.3) for more natural, readable writing.
    """
    print("\n✍️  [WRITER] Generating final report...")

    results_text = "\n\n".join([
        f"### {r['subtask']}\n{r['findings']}"
        for r in state["research_results"]
    ])

    system_prompt = """You are an expert technical writer producing a research report.

Write in clear, professional Markdown. The report must include:
1. # Title (derived from the query)
2. ## Executive Summary (2-3 sentences overview)
3. ## Key Findings (one subsection per research subtask)
4. ## Analysis & Insights (synthesized takeaways)
5. ## Recommendations (3-5 actionable items)
6. ## Sources (list any URLs or references mentioned)

Style guidelines:
- Use concrete data points and statistics where available
- Write for a technical audience (senior engineers)
- Use bullet points liberally for scannability
- Cite sources inline (e.g., "According to GitHub Octoverse 2024, ...")"""

    user_prompt = f"""Original Query: {state['query']}

Analyst Synthesis:
{state['analysis']}

Detailed Research Findings:
{results_text}

Write the complete research report now."""

    response = _llm_writer.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    word_count = len(response.content.split())
    print(f"  ✅ Report complete ({word_count} words)")

    return {"final_report": response.content}


# ---------------------------------------------------------------------------
# Routing Function (Conditional Edge Logic)
# ---------------------------------------------------------------------------

def should_continue(state: GraphState) -> str:
    """
    Decide whether to loop back to the Researcher or proceed to the Writer.

    This function is the "routing logic" attached to the Analyst node's
    conditional edge. LangGraph calls this after every Analyst execution.

    Returns:
        "continue" → Researcher node (more research needed)
        "done"     → Writer node (research quality approved)
    """
    # Hard stop to prevent infinite loops
    if state.get("iteration_count", 0) >= 3:
        print("  🛑 [ROUTER] Max iterations reached → proceeding to Writer")
        return "done"

    if state.get("approved", False):
        print("  ✅ [ROUTER] Research approved → proceeding to Writer")
        return "done"

    print("  🔄 [ROUTER] Research needs improvement → looping back to Researcher")
    return "continue"
