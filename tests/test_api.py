"""API test suite."""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint():
    """Test chat completion endpoint."""
    response = client.post(
        "/api/v1/chat/completions",
        json={"model": "default", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_websocket():
    """Test WebSocket connection."""
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"message": "test"})
        data = websocket.receive_json()
        assert "response" in data
