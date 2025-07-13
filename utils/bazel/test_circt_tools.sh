#!/bin/bash

# Test build status of all CIRCT tools
# Use bazel run <tool> -- --help to test each tool

set -e

# Define color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Tool list
tools=(
    "arcilator"
    "circt-as"
    "circt-bmc"
    "circt-dis"
    "circt-lec"
    "circt-lsp-server"
    "circt-opt"
    "circt-reduce"
    "circt-rtl-sim"
    "circt-synth"
    "circt-test"
    "circt-translate"
    "circt-verilog"
    "circt-verilog-lsp-server"
    "firld"
    "firtool"
    "handshake-runner"
    "hlstool"
    "kanagawatool"
    "om-linker"
)

# Store results
failed_tools=()
passed_tools=()

echo -e "${YELLOW}Testing CIRCT tool build status...${NC}"
echo "========================================"

# Iterate over all tools
for tool in "${tools[@]}"; do
    echo -n "Testing $tool... "
    
    # Try to run the tool
    if bazel run "//:$tool" -- --help > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Passed${NC}"
        passed_tools+=("$tool")
    else
        echo -e "${RED}✗ Failed${NC}"
        failed_tools+=("$tool")
    fi
done

echo ""
echo "========================================"
echo "Test Summary:"
echo ""

# Output passed tools
if [ ${#passed_tools[@]} -gt 0 ]; then
    echo -e "${GREEN}Tools built successfully (${#passed_tools[@]}/${#tools[@]}):${NC}"
    for tool in "${passed_tools[@]}"; do
        echo -e "  ${GREEN}✓${NC} $tool"
    done
    echo ""
fi

# Output failed tools
if [ ${#failed_tools[@]} -gt 0 ]; then
    echo -e "${RED}Tools failed to build (${#failed_tools[@]}/${#tools[@]}):${NC}"
    for tool in "${failed_tools[@]}"; do
        echo -e "  ${RED}✗${NC} $tool"
    done
    echo ""
    
    # Show detailed error messages
    echo -e "${YELLOW}Detailed error messages:${NC}"
    echo "----------------------"
    for tool in "${failed_tools[@]}"; do
        echo -e "${RED}$tool error details:${NC}"
        bazel run "//:$tool" -- --help 2>&1 | head -20
        echo "----------------------"
    done
    
    # Set exit status
    exit 1
else
    echo -e "${GREEN}All tools built successfully!${NC}"
    exit 0
fi 