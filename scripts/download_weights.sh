#!/bin/bash
# ==============================================================================
# JEPA-WM Pretrained Weights Download Script
# ==============================================================================
# This script downloads pretrained JEPA-WM weights for MetaWorld experiments.
# 
# Usage:
#   ./download_weights.sh [--force]
#
# Options:
#   --force   Overwrite existing files
#
# Prerequisites:
#   Set environment variable JEPAWM_CKPT to specify the download directory.
#   Example: export JEPAWM_CKPT=$HOME/checkpoints/jepa_lang_goal
# ==============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
FORCE=false
for arg in "$@"; do
    case $arg in
        --force)
            FORCE=true
            shift
            ;;
    esac
done

# Check environment variable
if [ -z "$JEPAWM_CKPT" ]; then
    echo -e "${RED}Error: JEPAWM_CKPT environment variable is not set.${NC}"
    echo ""
    echo "Please set it to your checkpoints directory:"
    echo "  export JEPAWM_CKPT=\$HOME/checkpoints/jepa_lang_goal"
    echo ""
    echo "Or run with all environment variables:"
    echo "  export JEPAWM_HOME=\$HOME/workspace/jepa_lang_goal"
    echo "  export JEPAWM_DSET=\$HOME/datasets/jepa_lang_goal"
    echo "  export JEPAWM_LOGS=\$HOME/logs/jepa_lang_goal"
    echo "  export JEPAWM_CKPT=\$HOME/checkpoints/jepa_lang_goal"
    echo "  export JEPAWM_OSSCKPT=\$HOME/oss_checkpoints/jepa_lang_goal"
    exit 1
fi

# Create pretrained directory
PRETRAINED_DIR="$JEPAWM_CKPT/pretrained"
mkdir -p "$PRETRAINED_DIR"

echo "=============================================="
echo "JEPA-WM Pretrained Weights Downloader"
echo "=============================================="
echo "Download directory: $PRETRAINED_DIR"
echo ""

# MetaWorld checkpoint
MW_URL="https://dl.fbaipublicfiles.com/jepa-wms/mw_jepa-wm.pth.tar"
MW_FILE="$PRETRAINED_DIR/mw_jepa-wm.pth.tar"

download_file() {
    local url=$1
    local output=$2
    local name=$3
    
    if [ -f "$output" ] && [ "$FORCE" = false ]; then
        file_size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
        if [ "$file_size" -gt 0 ]; then
            echo -e "${GREEN}✓ $name already exists ($(numfmt --to=iec-i --suffix=B $file_size))${NC}"
            echo "  Location: $output"
            echo "  (Use --force to re-download)"
            return 0
        fi
    fi
    
    echo -e "${YELLOW}⬇ Downloading $name...${NC}"
    echo "  URL: $url"
    echo "  Destination: $output"
    
    if curl -L -o "$output" "$url" --progress-bar; then
        file_size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
        if [ "$file_size" -gt 0 ]; then
            echo -e "${GREEN}✓ Downloaded successfully ($(numfmt --to=iec-i --suffix=B $file_size))${NC}"
        else
            echo -e "${RED}✗ Download failed: File is empty${NC}"
            rm -f "$output"
            return 1
        fi
    else
        echo -e "${RED}✗ Download failed${NC}"
        return 1
    fi
}

# Download MetaWorld checkpoint
echo ""
echo "--- MetaWorld JEPA-WM Checkpoint ---"
download_file "$MW_URL" "$MW_FILE" "MetaWorld JEPA-WM"

# Verify downloads
echo ""
echo "=============================================="
echo "Download Summary"
echo "=============================================="

verify_file() {
    local file=$1
    local name=$2
    
    if [ -f "$file" ]; then
        file_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
        if [ "$file_size" -gt 0 ]; then
            echo -e "${GREEN}✓ $name: $file${NC}"
            return 0
        fi
    fi
    echo -e "${RED}✗ $name: NOT FOUND${NC}"
    return 1
}

verify_file "$MW_FILE" "MetaWorld"

echo ""
echo "=============================================="
echo -e "${GREEN}✓ Download complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Create data index:"
echo "     python -m app.lang_goal.data_index --config configs/lang_goal/data_index.yaml"
echo ""
echo "  2. Train GoalHead:"
echo "     python -m app.lang_goal.train_goalhead --config configs/lang_goal/goalhead_train.yaml"
echo ""
echo "  3. Run evaluation:"
echo "     python -m evals.main --fname configs/lang_goal/mw_lang_goal_eval.yaml --debug"
echo ""
