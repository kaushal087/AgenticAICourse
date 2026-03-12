"""
convert_to_html.py — Convert ALL .md files to interlinked HTML pages.

Run:  python convert_to_html.py
Then: open html/home.html   (or double-click in Finder)

All HTML is written to a single html/ folder at the project root.
Every page shares the same sidebar so you can navigate freely.
After conversion all .md files are deleted.
"""

import os
import markdown
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "html"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Registry of every md file → (output html filename, nav label, nav group) ──
# Groups are rendered as section headers in the sidebar.
REGISTRY = [
    # (source_md_relative_to_BASE, output_html, nav_label, nav_group)
    ("README.md",                           "home.html",               "🏠 Home",               ""),
    ("website/index.md",                    "index.html",              "Overview",              "📚 Documentation"),
    ("website/concepts.md",                 "concepts.html",           "Core Concepts",         "📚 Documentation"),
    ("website/architecture.md",             "architecture.html",       "Architecture",          "📚 Documentation"),
    ("website/workflow.md",                 "workflow.html",           "Workflow",              "📚 Documentation"),
    ("website/demos.md",                    "demos.html",              "Demo Walkthrough",      "📚 Documentation"),
    ("website/quiz.md",                     "quiz.html",               "Quiz",                  "📚 Documentation"),
    ("slides/lecture_slides.md",            "lecture_slides.html",     "Lecture Slides",        "🎓 Slides"),
    ("slides/diagrams/agent_flow_slide.md", "agent_flow_slide.html",   "Agent Flow Diagram",    "🎓 Slides"),
    ("diagrams/architecture.md",            "diag_architecture.html",  "System Architecture",   "📊 Diagrams"),
    ("diagrams/agent_flow.md",              "diag_agent_flow.html",    "Agent Flow",            "📊 Diagrams"),
    ("diagrams/rag_pipeline.md",            "diag_rag_pipeline.html",  "RAG Pipeline",          "📊 Diagrams"),
    ("quizzes/quiz_questions.md",           "quiz_questions.html",     "Quiz Reference",        "📝 Quizzes"),
]

# ── CSS (light theme) ─────────────────────────────────────────────────────────
CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa;
    color: #1a202c;
    display: flex;
    min-height: 100vh;
}

/* ── Sidebar ── */
nav {
    width: 260px;
    min-width: 260px;
    background: #ffffff;
    padding: 24px 14px 40px;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    border-right: 1px solid #e2e8f0;
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}
nav .brand {
    font-size: 15px;
    font-weight: 800;
    color: #4f46e5;
    margin-bottom: 2px;
    letter-spacing: -0.3px;
}
nav .sub {
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 28px;
}
nav .group {
    font-size: 10px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 20px 0 6px 10px;
}
nav a {
    display: block;
    padding: 8px 12px;
    margin-bottom: 2px;
    border-radius: 7px;
    color: #475569;
    text-decoration: none;
    font-size: 13.5px;
    transition: background 0.12s, color 0.12s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
nav a:hover { background: #f1f5f9; color: #1a202c; }
nav a.active {
    background: #eef2ff;
    color: #4f46e5;
    font-weight: 600;
    border-left: 3px solid #4f46e5;
    padding-left: 9px;
}

/* ── Main content ── */
main {
    flex: 1;
    padding: 52px 68px 80px;
    max-width: 960px;
    min-width: 0;
    background: #f5f7fa;
}

/* ── Typography ── */
h1 { font-size: 2rem; color: #111827; margin-bottom: 18px; margin-top: 44px; line-height: 1.25; }
h2 { font-size: 1.45rem; color: #1e3a8a; margin-top: 40px; margin-bottom: 14px;
     padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }
h3 { font-size: 1.1rem; color: #3730a3; margin-top: 28px; margin-bottom: 10px; }
h4 { font-size: 0.95rem; color: #4b5563; margin-top: 18px; margin-bottom: 6px; font-weight: 600; }
h1:first-child, h2:first-child { margin-top: 0; }
p  { line-height: 1.8; color: #374151; margin-bottom: 14px; }
ul, ol { padding-left: 24px; margin-bottom: 14px; }
li { line-height: 1.75; color: #374151; margin-bottom: 5px; }
a  { color: #4f46e5; text-decoration: none; }
a:hover { text-decoration: underline; }
strong { color: #111827; }

/* ── Code ── */
code {
    background: #f1f5f9;
    color: #be185d;
    padding: 2px 7px;
    border-radius: 5px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 0.855em;
    border: 1px solid #e2e8f0;
}
pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 22px 24px;
    overflow-x: auto;
    margin: 18px 0 24px;
}
pre code {
    background: none;
    color: #1e293b;
    padding: 0;
    font-size: 0.83em;
    line-height: 1.65;
    border: none;
}

/* ── Blockquote ── */
blockquote {
    border-left: 4px solid #6366f1;
    background: #eef2ff;
    padding: 14px 20px;
    border-radius: 0 8px 8px 0;
    margin: 18px 0;
    color: #4338ca;
    font-style: italic;
}

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 18px 0 26px;
    font-size: 0.88em;
    background: #ffffff;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
th {
    background: #eef2ff;
    color: #3730a3;
    padding: 11px 16px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #c7d2fe;
}
td {
    padding: 10px 16px;
    border-bottom: 1px solid #f1f5f9;
    color: #374151;
    vertical-align: top;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }

/* ── Misc ── */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 30px 0; }
img { max-width: 100%; border-radius: 8px; margin: 16px 0; }
"""

# ── HTML page template ────────────────────────────────────────────────────────
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title} — Multi-Agent LangGraph</title>
  <style>{css}</style>
</head>
<body>
<nav>
  <div class="brand">🤖 LangGraph</div>
  <div class="sub">Multi-Agent Systems · IK Agentic AI</div>
  {nav_html}
</nav>
<main>
{body_html}
</main>
<script>
  // Mark the current page active in nav
  const current = location.pathname.split('/').pop() || 'home.html';
  document.querySelectorAll('nav a').forEach(a => {{
    if (a.getAttribute('href') === current) a.classList.add('active');
  }});
</script>
</body>
</html>
"""

# ── Build sidebar nav HTML (shared, with group headers) ──────────────────────
def build_nav(active_html: str) -> str:
    seen_groups = []
    parts = []
    for _src, out_html, label, group in REGISTRY:
        if group and group not in seen_groups:
            seen_groups.append(group)
            parts.append(f'<div class="group">{group}</div>')
        active = ' class="active"' if out_html == active_html else ""
        parts.append(f'<a href="{out_html}"{active}>{label}</a>')
    return "\n  ".join(parts)

# ── Markdown converter ────────────────────────────────────────────────────────
def make_md():
    return markdown.Markdown(
        extensions=["fenced_code", "tables", "toc", "nl2br", "sane_lists"],
    )

# ── Convert every file ────────────────────────────────────────────────────────
converted = []
skipped = []

for src_rel, out_html, label, _group in REGISTRY:
    src = BASE / src_rel
    if not src.exists():
        skipped.append(src_rel)
        continue

    raw = src.read_text(encoding="utf-8")
    md = make_md()
    body = md.convert(raw)

    html = HTML_TEMPLATE.format(
        title=label,
        css=CSS,
        nav_html=build_nav(out_html),
        body_html=body,
    )

    out_path = OUT_DIR / out_html
    out_path.write_text(html, encoding="utf-8")
    converted.append((src_rel, out_html))
    print(f"  ✅  {src_rel}  →  html/{out_html}")

if skipped:
    for s in skipped:
        print(f"  ⚠️   SKIP (not found): {s}")

# ── Delete all .md files ──────────────────────────────────────────────────────
print("\n🗑️  Removing .md files...")
deleted = []
for md_path in BASE.rglob("*.md"):
    md_path.unlink()
    deleted.append(str(md_path.relative_to(BASE)))
    print(f"  🗑️   deleted: {md_path.relative_to(BASE)}")

print(f"\n🎉  Done!")
print(f"   Converted : {len(converted)} files")
print(f"   Deleted   : {len(deleted)} .md files")
print(f"\n   Open in browser:")
print(f'   open "{OUT_DIR / "home.html"}"')
