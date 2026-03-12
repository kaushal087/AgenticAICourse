import glob

ALL_TOPICS_BLOCK = (
    '  <div class="group">📋 All Topics</div>\n'
    '  <a href="topic1_fundamentals.html">1. LLMs, RAG &amp; Agents</a>\n'
    '  <a href="topic2_llm_apps.html">2. Building LLM Apps</a>\n'
    '  <a href="topic3_langchain_agent.html">3. First Agent: LangChain</a>\n'
    '  <a href="home.html">4. Multi-Agent LangGraph</a>\n'
    '  <a href="topic5_mcp_a2a.html">5. MCP &amp; A2A Protocol</a>\n'
    '  <a href="topic6_multimodal.html">6. Multimodal Audio Agents</a>\n'
    '  <a href="topic7_specialized.html">7. Specialized Agents</a>\n'
    '  <a href="topic8_evaluation.html">8. Evaluation &amp; LangSmith</a>\n'
    '  <a href="topic9_python_genai.html">9. Python for GenAI</a>\n'
    '  <div class="group">📚 Topic 4 Deep-Dive</div>'
)

OLD_GROUP = '  <div class="group">📚 Documentation</div>'

files = [f for f in glob.glob('*.html') if not f.startswith('topic')]
updated = []
skipped = []

for fname in sorted(files):
    with open(fname, encoding='utf-8') as f:
        content = f.read()
    if '📋 All Topics' in content:
        skipped.append(fname + ' (already done)')
        continue
    if OLD_GROUP not in content:
        skipped.append(fname + ' (no match)')
        continue
    new_content = content.replace(OLD_GROUP, ALL_TOPICS_BLOCK)
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(new_content)
    updated.append(fname)

print('Updated:', updated)
print('Skipped:', skipped)
