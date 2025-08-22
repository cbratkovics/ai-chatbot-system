#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AI Chatbot Demo Setup${NC}"
echo "========================"
echo ""

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose detected${NC}"

# Check for .env file
if [ ! -f backend/.env ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
    cp backend/.env.example backend/.env
    echo ""
    echo -e "${YELLOW}📝 IMPORTANT: Please edit backend/.env and add your API keys:${NC}"
    echo "   1. Open backend/.env in your text editor"
    echo "   2. Replace 'your-openai-key-here' with your OpenAI API key"
    echo "   3. (Optional) Add your Anthropic API key"
    echo "   4. Save the file and run this script again"
    echo ""
    echo -e "${BLUE}Get your API keys from:${NC}"
    echo "   • OpenAI: https://platform.openai.com/api-keys"
    echo "   • Anthropic: https://console.anthropic.com/account/keys"
    exit 1
fi

# Validate API keys
echo -e "${BLUE}🔍 Checking API keys...${NC}"

OPENAI_KEY=$(grep "^OPENAI_API_KEY=" backend/.env | cut -d '=' -f2)
ANTHROPIC_KEY=$(grep "^ANTHROPIC_API_KEY=" backend/.env | cut -d '=' -f2)

if [[ "$OPENAI_KEY" == "your-openai-key-here" || "$OPENAI_KEY" == "" ]] && \
   [[ "$ANTHROPIC_KEY" == "your-anthropic-key-here" || "$ANTHROPIC_KEY" == "" ]]; then
    echo -e "${RED}❌ No valid API keys found in backend/.env${NC}"
    echo -e "${YELLOW}📝 Please add at least one API key to backend/.env and run again${NC}"
    exit 1
fi

if [[ "$OPENAI_KEY" != "your-openai-key-here" && "$OPENAI_KEY" != "" ]]; then
    echo -e "${GREEN}✅ OpenAI API key detected${NC}"
fi

if [[ "$ANTHROPIC_KEY" != "your-anthropic-key-here" && "$ANTHROPIC_KEY" != "" ]]; then
    echo -e "${GREEN}✅ Anthropic API key detected${NC}"
fi

# Stop any existing containers
echo ""
echo -e "${BLUE}🧹 Cleaning up existing containers...${NC}"
docker-compose -f docker-compose.demo.yml down 2>/dev/null

# Build and start services
echo ""
echo -e "${BLUE}🐳 Building and starting services...${NC}"
echo "This may take a few minutes on first run..."
echo ""

docker-compose -f docker-compose.demo.yml up --build -d

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start services${NC}"
    echo "Please check the error messages above"
    exit 1
fi

# Wait for services to be ready
echo ""
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service is ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${YELLOW}⚠️  $service is taking longer than expected${NC}"
    return 1
}

# Check backend
echo -n "Checking backend API"
check_service "Backend API" "http://localhost:8000/api/health/ready"

# Check frontend
echo -n "Checking frontend UI"
check_service "Frontend UI" "http://localhost:3000"

# Display status
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
docker-compose -f docker-compose.demo.yml ps

# Success message
echo ""
echo "========================================="
echo -e "${GREEN}🎉 Demo is ready!${NC}"
echo "========================================="
echo ""
echo -e "${BLUE}Access your demo at:${NC}"
echo "  📱 Chat UI:    http://localhost:3000"
echo "  📚 API Docs:   http://localhost:8000/docs"
echo "  🔧 API Base:   http://localhost:8000"
echo ""
echo -e "${BLUE}Quick test:${NC}"
echo "  curl http://localhost:8000/api/demo/status"
echo ""
echo -e "${BLUE}To view logs:${NC}"
echo "  docker-compose -f docker-compose.demo.yml logs -f"
echo ""
echo -e "${BLUE}To stop the demo:${NC}"
echo "  docker-compose -f docker-compose.demo.yml down"
echo ""
echo -e "${GREEN}Happy chatting! 🤖${NC}"