#!/bin/bash
# Commit Strategy Script for JEPA-WMs Fork
# Usage: bash commit_changes.sh

set -e  # Exit on error

echo "Starting granular commit process..."

# 1. Config & Cleanup
echo "Step 1: Committing .gitignore and project config..."
git add .gitignore pyproject.toml
git commit -m "chore: update gitignore and project dependencies for language goal fork"

# 2. Documentation
echo "Step 2: Committing Documentation..."
git add README.md
git add docs/ 2>/dev/null || true
git commit -m "docs: add language goal fork documentation and experiment logs"

# ... (Step 3, 4, 5 same) ...



# 3. Core Feature: Language Goal Head
echo "Step 3: Committing Language Goal Head implementation..."
git add app/lang_goal/
git add evals/lang_goal/ # If exists
# Add specific model changes if they are new files, or stage modified src files
# Assuming src/models changes are related to this:
git add src/models/
git commit -m "feat: implement Language Goal Head and text encoder integration"

# 4. Agent Logic
echo "Step 4: Committing Agent modifications..."
git add evals/simu_env_planning/planning/gc_agent.py
git add evals/utils.py
git commit -m "feat(agent): update GC_Agent to support latent language goals"

# 5. Configurations
echo "Step 5: Committing Configurations..."
git add configs/lang_goal/
# Explicitly add specific config files if they are not caught by directory add
git add configs/lang_goal/*.yaml
git add assets/lang_goal/ 2>/dev/null || true # If exists
git commit -m "config: add training and evaluation configs for language goals (CLIP, SigLIP 2)"

# 6. Utilities
echo "Step 6: Committing Utility scripts..."
git add simple_check.py debug_render.py save_image.py
# Check if scripts/ directory has new files
git add scripts/eval_lang_goal_quick.py 2>/dev/null || true
git commit -m "tool: add verification and debugging scripts"

# 7. Final Verification
echo "Step 7: Verify status..."
git status

echo "----------------------------------------"
echo "Granular commits completed!"
echo "Please review 'git log' to ensure everything looks correct."
echo "Then push to your repository with: git push origin main"
