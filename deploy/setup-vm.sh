#!/usr/bin/env bash
# =============================================================================
# BerjQuant VM Setup Script
# Run this on a fresh Ubuntu 22.04+ VM (Oracle Cloud / GCP / any Linux)
# Usage: bash setup-vm.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/bngapple/berjquant.git"
APP_DIR="$HOME/berjquant"
PYTHON_VERSION="3.12"

echo "========================================="
echo " BerjQuant VM Setup"
echo "========================================="

# --- System packages ---
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    software-properties-common \
    git curl wget htop tmux \
    build-essential libssl-dev libffi-dev \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    2>/dev/null || {
        # If python3.12 not available, add deadsnakes PPA
        echo "Adding deadsnakes PPA for Python ${PYTHON_VERSION}..."
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
    }

# --- Clone repo ---
echo "[2/7] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$APP_DIR" && git pull
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# --- Python venv ---
echo "[3/7] Setting up Python virtual environment..."
python${PYTHON_VERSION} -m venv .venv
source .venv/bin/activate

# --- Install dependencies ---
echo "[4/7] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e ".[dev]" -q

# --- Create data directories ---
echo "[5/7] Creating data directories..."
mkdir -p data/raw data/processed data/features
mkdir -p results reports logs

# --- Run tests to verify ---
echo "[6/7] Running tests to verify installation..."
python -m pytest tests/ -v --tb=short || echo "Some tests may fail without data — that's OK"

# --- Install Claude Code (Node.js + npm) ---
echo "[7/7] Installing Claude Code CLI..."
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y -qq nodejs
fi
npm install -g @anthropic-ai/claude-code 2>/dev/null || echo "Claude Code install requires auth — run 'npm install -g @anthropic-ai/claude-code' manually"

echo ""
echo "========================================="
echo " Setup Complete!"
echo "========================================="
echo ""
echo "Quick start:"
echo "  cd $APP_DIR"
echo "  source .venv/bin/activate"
echo "  python run_backtest.py"
echo ""
echo "To keep processes running after SSH disconnect:"
echo "  tmux new -s quant"
echo "  # run your commands, then Ctrl+B, D to detach"
echo "  # tmux attach -t quant  to reconnect"
echo ""
echo "To use Claude Code:"
echo "  claude  (in the project directory)"
echo ""
