#!/usr/bin/env bash
# run_demo.sh — Launch the Multi-Agent Research System demo
#
# Usage:
#   bash run_demo.sh                     # Streamlit web UI
#   bash run_demo.sh --cli               # CLI demo
#   bash run_demo.sh --cli --query "..."  # CLI with custom query
#
# Prerequisites:
#   - Python 3.10+ installed
#   - OPENAI_API_KEY environment variable set
#   - (optional) TAVILY_API_KEY for live web search

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_ROOT/code"

# ---- Color output --------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         🤖 Multi-Agent Research System — Demo Launcher       ║"
echo "║         Interview Kickstart | Agentic AI Series              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ---- Check for OpenAI API key --------------------------------------------
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: OPENAI_API_KEY is not set.${NC}"
    echo ""
    echo "Set it with:"
    echo "  export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
    echo "Or create a .env file in the code/ directory:"
    echo "  echo 'OPENAI_API_KEY=sk-your-key' > $CODE_DIR/.env"
    exit 1
fi

echo -e "${GREEN}✅ OpenAI API key found${NC}"

if [ -n "$TAVILY_API_KEY" ]; then
    echo -e "${GREEN}✅ Tavily API key found (live web search enabled)${NC}"
else
    echo -e "${YELLOW}⚠️  TAVILY_API_KEY not set — using mock web search${NC}"
    echo "   Get a free key: https://app.tavily.com"
fi

# ---- Check Python --------------------------------------------------------
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo -e "${RED}❌ Python not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "${GREEN}✅ Python: $PYTHON_VERSION${NC}"

# ---- Install dependencies ------------------------------------------------
echo ""
echo -e "${BLUE}📦 Checking dependencies...${NC}"

if ! $PYTHON -c "import langchain" 2>/dev/null; then
    echo "Installing requirements (first run only)..."
    $PYTHON -m pip install -r "$CODE_DIR/requirements.txt" -q
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

# ---- Parse arguments -----------------------------------------------------
CLI_MODE=false
QUERY="What are the best practices for building production-ready LLM applications in 2025?"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cli)
            CLI_MODE=true
            shift
            ;;
        --query)
            QUERY="$2"
            shift 2
            ;;
        *)
            echo -e "${YELLOW}Unknown argument: $1${NC}"
            shift
            ;;
    esac
done

# ---- Launch demo ---------------------------------------------------------
echo ""

if [ "$CLI_MODE" = true ]; then
    echo -e "${BLUE}🖥️  Launching CLI demo...${NC}"
    echo -e "Query: ${YELLOW}$QUERY${NC}"
    echo ""
    cd "$CODE_DIR"
    $PYTHON main.py --query "$QUERY"
else
    echo -e "${BLUE}🌐 Launching Streamlit web UI...${NC}"
    echo ""
    echo "Opening: http://localhost:8501"
    echo "Press Ctrl+C to stop"
    echo ""

    if ! $PYTHON -c "import streamlit" 2>/dev/null; then
        $PYTHON -m pip install streamlit -q
    fi

    cd "$SCRIPT_DIR"
    $PYTHON -m streamlit run demo_app.py \
        --server.headless false \
        --server.port 8501 \
        --browser.gatherUsageStats false
fi
