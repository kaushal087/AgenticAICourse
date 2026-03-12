"""
demo_app.py — Streamlit Interactive Demo

Run with: streamlit run demo_app.py

This Streamlit app provides an interactive UI for the Multi-Agent Research System.
Users can:
  - Enter a research question
  - Watch agents work in real time (streamed output)
  - See each agent's intermediate output in expandable sections
  - Read the final formatted report

Architecture:
  This file is intentionally kept simple — it's just a UI wrapper around the
  same main.py pipeline used in the CLI demo. The actual agent logic lives in
  code/agent.py and code/tools.py.
"""

import os
import sys
import time
import threading
from pathlib import Path

import streamlit as st

# Add the code/ directory to Python path so we can import agent modules
CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar — configuration and instructions
# ------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Your OpenAI API key (sk-...)",
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    tavily_key = st.text_input(
        "Tavily API Key (optional)",
        type="password",
        value=os.getenv("TAVILY_API_KEY", ""),
        help="For live web search. Leave blank to use mock search.",
    )
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    st.markdown("### 📖 How It Works")
    st.markdown("""
This demo runs a **4-agent LangGraph pipeline**:

1. 🧠 **Planner** — Breaks your question into subtasks
2. 🔍 **Researcher** — Gathers info via web search + RAG
3. 🧪 **Analyst** — Evaluates quality, may loop back
4. ✍️  **Writer** — Produces the final report

The agents share a **GraphState** dictionary that
flows through the LangGraph directed graph.
    """)

    st.divider()
    st.markdown("### 💡 Sample Queries")
    sample_queries = [
        "What are best practices for building production-ready LLM applications?",
        "How does LangGraph compare to CrewAI for multi-agent systems?",
        "What are the top 5 use cases for AI agents in enterprise software?",
        "Explain the ReAct pattern and its role in AI agent design.",
    ]
    for q in sample_queries:
        if st.button(f"→ {q[:50]}...", key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state["preset_query"] = q

# ------------------------------------------------------------------
# Main content
# ------------------------------------------------------------------
st.title("🤖 Multi-Agent Research System")
st.caption("Built with LangGraph | 4-Agent Pipeline: Planner → Researcher → Analyst → Writer")

st.divider()

# Query input
default_query = st.session_state.get(
    "preset_query",
    "What are the best practices for building production-ready LLM applications in 2025?",
)
query = st.text_area(
    "🔎 Enter your research question:",
    value=default_query,
    height=80,
    placeholder="e.g., What are the key differences between LangGraph and CrewAI?",
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)
with col2:
    st.caption("⏱ Typically takes 20–40 seconds with GPT-4o")

# ------------------------------------------------------------------
# Pipeline execution
# ------------------------------------------------------------------
if run_button:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Please enter your OpenAI API key in the sidebar.")
        st.stop()

    if not query.strip():
        st.warning("⚠️ Please enter a research question.")
        st.stop()

    # Clear preset query
    st.session_state.pop("preset_query", None)

    # Setup containers for real-time output
    st.divider()
    st.subheader("🔄 Agent Execution")

    planner_container   = st.expander("🧠 Planner Agent", expanded=True)
    researcher_container = st.expander("🔍 Researcher Agent", expanded=True)
    analyst_container   = st.expander("🧪 Analyst Agent", expanded=True)
    writer_container    = st.expander("✍️ Writer Agent", expanded=True)
    report_container    = st.container()

    planner_ph   = planner_container.empty()
    researcher_ph = researcher_container.empty()
    analyst_ph   = analyst_container.empty()
    writer_ph    = writer_container.empty()

    planner_ph.info("⏳ Waiting...")
    researcher_ph.info("⏳ Waiting...")
    analyst_ph.info("⏳ Waiting...")
    writer_ph.info("⏳ Waiting...")

    # ---- Run the pipeline ------------------------------------------------
    try:
        from rag_pipeline import RAGPipeline
        from tools import set_rag_pipeline
        from agent import (
            GraphState, planner_node, researcher_node,
            analyst_node, writer_node, should_continue,
        )
        from langgraph.graph import StateGraph, END

        # Initialize RAG
        with st.spinner("📚 Loading knowledge base..."):
            rag = RAGPipeline()
            kb_path = Path(__file__).parent / "sample_data" / "knowledge_base.txt"
            if kb_path.exists():
                rag.ingest(str(kb_path))
            else:
                rag.ingest_text(
                    "LangGraph is a framework for stateful multi-agent AI systems. "
                    "It uses directed graphs with typed state, conditional edges, "
                    "and checkpointing.",
                    source="inline"
                )
            set_rag_pipeline(rag)

        # Build the graph
        workflow = StateGraph(GraphState)
        workflow.add_node("planner",    planner_node)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("analyst",    analyst_node)
        workflow.add_node("writer",     writer_node)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_conditional_edges(
            "analyst", should_continue,
            {"continue": "researcher", "done": "writer"}
        )
        workflow.add_edge("writer", END)
        app = workflow.compile()

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

        # ---- Stream graph events -----------------------------------------
        with st.spinner("🤖 Agents working..."):
            for event in app.stream(initial_state):
                for node_name, node_output in event.items():

                    if node_name == "planner" and "plan" in node_output:
                        plan = node_output["plan"]
                        plan_md = "\n".join([f"- {t}" for t in plan])
                        planner_ph.success(
                            f"✅ Plan created: {len(plan)} subtasks\n\n{plan_md}"
                        )

                    elif node_name == "researcher" and "research_results" in node_output:
                        results = node_output["research_results"]
                        iteration = node_output.get("iteration_count", "?")
                        summary = f"✅ Iteration {iteration}: {len(results)} subtasks researched"
                        details = "\n\n".join([
                            f"**{r['subtask'][:60]}...**\n{r['findings'][:200]}..."
                            for r in results
                        ])
                        researcher_ph.success(summary + "\n\n" + details)

                    elif node_name == "analyst":
                        analysis = node_output.get("analysis", "")
                        approved = node_output.get("approved", False)
                        status = "✅ Approved" if approved else "🔄 Looping for more research"
                        analyst_ph.success(
                            f"{status}\n\n**Analysis:**\n{analysis[:400]}..."
                        )

                    elif node_name == "writer" and "final_report" in node_output:
                        word_count = len(node_output["final_report"].split())
                        writer_ph.success(f"✅ Report complete ({word_count} words)")
                        final_report = node_output["final_report"]

        # ---- Display final report ----------------------------------------
        st.divider()
        report_container.subheader("📄 Final Research Report")
        report_container.markdown(final_report)

        # Download button
        report_container.download_button(
            label="⬇️ Download Report (Markdown)",
            data=final_report,
            file_name="research_report.md",
            mime="text/markdown",
        )

    except ImportError as e:
        st.error(f"❌ Import error: {e}\n\nMake sure you've installed requirements.txt")
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        st.exception(e)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
st.caption(
    "Multi-Agent Research System | Interview Kickstart — Agentic AI Series | "
    "Built with LangGraph + LangChain + OpenAI"
)
