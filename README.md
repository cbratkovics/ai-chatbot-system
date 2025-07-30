# AI-Powered Conversational Intelligence Platform

[![CI Pipeline](https://github.com/cbratkovics/ai-chatbot-system/actions/workflows/ci.yml/badge.svg)](https://github.com/cbratkovics/ai-chatbot-system/actions/workflows/ci.yml)
[![Deploy to Production](https://github.com/cbratkovics/ai-chatbot-system/actions/workflows/deploy.yml/badge.svg)](https://github.com/cbratkovics/ai-chatbot-system/actions/workflows/deploy.yml)
[![Load Testing](https://github.com/cbratkovics/ai-chatbot-system/actions/workflows/load-test.yml/badge.svg)](https://github.com/cbratkovics/ai-chatbot-system/actions/workflows/load-test.yml)

A production-grade, multi-tenant conversational AI platform demonstrating enterprise-level AI engineering practices. This system handles 100+ concurrent users with real-time streaming responses, intelligent caching, and comprehensive observability.

## 🚀 Features

### Core Capabilities
- **Multi-Model Orchestration**: Seamless integration with OpenAI and Anthropic models
- **Function Calling**: Built-in calculator, data analysis, web search, and scraping functions
- **Multi-Modal Support**: GPT-4 Vision integration for image understanding
- **Intelligent Failover**: Automatic fallback between providers for high availability
- **Real-time Streaming**: WebSocket-based streaming for responsive user experience
- **Semantic Caching**: Embedding-based cache for intelligent response reuse
- **Cost Optimization**: Query complexity analysis for optimal model selection

### Production Features
- **Rate Limiting**: Token bucket algorithm protecting against abuse
- **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards
- **Session Management**: Redis-backed persistent conversation history
- **Quality Evaluation**: Real-time response quality assessment
- **Error Handling**: Graceful error recovery with detailed logging

### Architecture Highlights
- **Scalable Design**: Microservices-ready architecture
- **Async-First**: Built on FastAPI with full async/await support
- **Container-Ready**: Docker and docker-compose configurations
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Anthropic API key (optional)

## 🛠️ Installation

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/cbratkovics/ai-chatbot-system.git
cd ai-chatbot-system
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Start the services:
```bash
docker-compose up -d
```

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd api
pip install -r requirements.txt
```

3. Start Redis:
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## 🔧 Configuration

Edit `.env` file with your configurations:

```env
# API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key  # Optional

# Redis
REDIS_URL=redis://localhost:6379

# API Settings
API_TITLE=AI Conversational Platform
API_VERSION=1.0.0
DEBUG=false

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Logging
LOG_LEVEL=INFO
```

## 📡 API Usage

### Create a Chat Session

```bash
curl -X POST http://localhost:8000/api/v1/chat/sessions
```

Response:
```json
{
  "session_id": "uuid-here",
  "user_id": null,
  "created_at": "2024-01-01T00:00:00",
  "status": "active",
  "config": {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

### Send a Message

```bash
curl -X POST http://localhost:8000/api/v1/chat/messages \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "session_id": "uuid-here"
  }'
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/chat/stream/session-id');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'stream') {
    console.log('Chunk:', data.content);
  }
};

ws.send(JSON.stringify({
  message: "Tell me a story"
}));
```

## 📊 Monitoring

### Access Services
- **API Documentation**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Available Metrics
- Request rate and latency
- Token usage by model
- Cache hit/miss rates
- API costs tracking
- Active sessions count
- Error rates by endpoint

## 🧪 Testing

### Run Unit Tests
```bash
docker-compose exec api pytest tests/ -v
```

### Run Load Tests
```bash
locust -f api/tests/load_test.py --host=http://localhost:8000
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   WebSocket     │     │   REST API      │     │   Metrics       │
│   Endpoints     │     │   Endpoints     │     │   Endpoint      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      FastAPI App        │
                    │  ┌──────────────────┐  │
                    │  │   Middleware     │  │
                    │  │  - Rate Limiter  │  │
                    │  │  - Error Handler │  │
                    │  └──────────────────┘  │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌────────▼────────┐    ┌─────────▼────────┐    ┌────────▼────────┐
│  LLM Services   │    │  Cache Services  │    │   Monitoring    │
│  - OpenAI       │    │  - Redis Cache   │    │  - Prometheus   │
│  - Anthropic    │    │  - Semantic      │    │  - Quality Eval │
│  - Orchestrator │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Redis Store        │
                    │  - Session Data         │
                    │  - Message History      │
                    │  - Cache Entries        │
                    └─────────────────────────┘
```

## 🚀 Performance

- **Concurrent Users**: 100+ WebSocket connections
- **P99 Latency**: < 2 seconds
- **Cache Hit Rate**: 30%+ with semantic matching
- **Availability**: 99.9% with automatic failover

## 🔐 Security

- Rate limiting per IP address
- Input validation and sanitization
- Secure API key management
- CORS configuration for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- FastAPI for the excellent web framework
- Redis for high-performance caching

## 📞 Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/cbratkovics/ai-chatbot-system/issues) page.