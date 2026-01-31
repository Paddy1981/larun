#!/bin/bash
#
# LARUN TinyML - Installation Script
# =================================
# One-command installation for Linux and macOS
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Paddy1981/larun/main/install.sh | bash
#   OR
#   ./install.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                      ║${NC}"
echo -e "${CYAN}║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                      ║${NC}"
echo -e "${CYAN}║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║                      ║${NC}"
echo -e "${CYAN}║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║                      ║${NC}"
echo -e "${CYAN}║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║                      ║${NC}"
echo -e "${CYAN}║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                      ║${NC}"
echo -e "${CYAN}║                                                                      ║${NC}"
echo -e "${CYAN}║     TinyML for Astronomical Data Analysis                            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
echo -e "${CYAN}Checking prerequisites...${NC}"

check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}✗ Python not found!${NC}"
        echo "Please install Python 3.8 or newer:"
        echo "  - macOS: brew install python"
        echo "  - Ubuntu/Debian: sudo apt install python3 python3-pip"
        echo "  - Fedora: sudo dnf install python3 python3-pip"
        exit 1
    fi

    # Check version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info[1])')

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        echo -e "${RED}✗ Python $PYTHON_VERSION is installed, but Python 3.8+ is required${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
}

check_pip() {
    if $PYTHON_CMD -m pip --version &> /dev/null; then
        PIP_VERSION=$($PYTHON_CMD -m pip --version | cut -d' ' -f2)
        echo -e "${GREEN}✓ pip $PIP_VERSION${NC}"
    else
        echo -e "${RED}✗ pip not found!${NC}"
        echo "Installing pip..."
        curl -fsSL https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD
    fi
}

check_git() {
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        echo -e "${GREEN}✓ git $GIT_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠ git not found (optional, for development)${NC}"
    fi
}

# Run checks
check_python
check_pip
check_git

echo ""

# Installation options
echo -e "${CYAN}Choose installation method:${NC}"
echo "  1) Install from PyPI (recommended)"
echo "  2) Install from GitHub (latest)"
echo "  3) Install in development mode (for contributors)"
echo ""
read -p "Enter choice [1]: " CHOICE
CHOICE=${CHOICE:-1}

case $CHOICE in
    1)
        echo ""
        echo -e "${CYAN}Installing from PyPI...${NC}"
        $PYTHON_CMD -m pip install --upgrade larun
        ;;
    2)
        echo ""
        echo -e "${CYAN}Installing from GitHub...${NC}"
        $PYTHON_CMD -m pip install --upgrade git+https://github.com/Paddy1981/larun.git
        ;;
    3)
        echo ""
        echo -e "${CYAN}Cloning repository...${NC}"
        if [ -d "larun" ]; then
            echo "Directory 'larun' already exists. Using existing directory."
            cd larun
            git pull || true
        else
            git clone https://github.com/Paddy1981/larun.git
            cd larun
        fi
        echo ""
        echo -e "${CYAN}Installing in development mode...${NC}"
        $PYTHON_CMD -m pip install -e ".[dev]"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""

# Verify installation
echo -e "${CYAN}Verifying installation...${NC}"

if command -v larun &> /dev/null; then
    echo -e "${GREEN}✓ larun command is available${NC}"
else
    echo -e "${YELLOW}⚠ larun command not found in PATH${NC}"
    echo "  You may need to add ~/.local/bin to your PATH:"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
fi

if command -v larun-chat &> /dev/null; then
    echo -e "${GREEN}✓ larun-chat command is available${NC}"
else
    echo -e "${YELLOW}⚠ larun-chat command not found in PATH${NC}"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Installation Complete!                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Quick start:"
echo "  larun          # Start interactive CLI"
echo "  larun-chat     # Start chat interface"
echo ""
echo "Documentation: https://github.com/Paddy1981/larun"
echo ""
