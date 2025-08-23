#!/bin/bash

# Setup script for git hooks
# This configures git to use the hooks in .githooks directory

echo "Setting up git hooks..."

# Set git hooks path to .githooks directory
git config core.hooksPath .githooks

# Make hooks executable
chmod +x .githooks/pre-commit
chmod +x .githooks/commit-msg
chmod +x .githooks/prepare-commit-msg

echo "✅ Git hooks configured successfully!"
echo ""
echo "Hooks installed:"
echo "  - pre-commit: Checks for AI/Claude references"
echo "  - commit-msg: Validates commit messages"
echo "  - prepare-commit-msg: Removes Co-Authored-By lines"
echo ""
echo "To bypass hooks (emergency only): git commit --no-verify"