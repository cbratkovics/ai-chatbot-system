#!/bin/bash

# Validation script for demo readiness

echo "🔍 Validating Demo Setup..."
echo "=========================="

ERRORS=0
WARNINGS=0

# Check required files
echo -n "Checking required files... "
REQUIRED_FILES=(
    "docker-compose.demo.yml"
    "backend/Dockerfile.demo"
    "backend/requirements.demo.txt"
    "backend/.env.example"
    "backend/app/demo_config.py"
    "backend/app/main_demo.py"
    "frontend/Dockerfile.demo"
    "setup_demo.sh"
    "README.demo.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo "✅"
else
    echo "❌ Missing $ERRORS files"
fi

# Check removed directories
echo -n "Checking cleanup... "
REMOVED_DIRS=(
    "ci"
    "benchmarks"
    "finops"
    "k8s"
    "backend/tests/contract"
    "backend/tests/load_testing"
)

for dir in "${REMOVED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "⚠️  Directory still exists: $dir"
        WARNINGS=$((WARNINGS + 1))
    fi
done

if [ $WARNINGS -eq 0 ]; then
    echo "✅"
fi

# Check Docker
echo -n "Checking Docker... "
if command -v docker &> /dev/null; then
    echo "✅"
else
    echo "❌"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "=========================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ Demo is ready for deployment!"
    echo ""
    echo "Next steps:"
    echo "1. Copy backend/.env.example to backend/.env"
    echo "2. Add your API keys to backend/.env"
    echo "3. Run ./setup_demo.sh"
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  Demo is ready with $WARNINGS warnings"
else
    echo "❌ Demo has $ERRORS errors and $WARNINGS warnings"
    echo "Please fix the issues above before proceeding"
fi