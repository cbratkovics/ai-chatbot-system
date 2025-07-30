import pytest
from app.services.llm.orchestrator import LLMOrchestrator
from app.models.chat import Message

@pytest.mark.asyncio
async def test_llm_orchestrator_model_selection():
    orchestrator = LLMOrchestrator()
    
    # Test simple query
    simple_model = orchestrator.select_model_for_query("Hi", optimize_cost=True)
    assert simple_model == "gpt-3.5-turbo"
    
    # Test complex query
    complex_query = "Explain the theory of relativity in detail " * 10
    complex_model = orchestrator.select_model_for_query(complex_query, optimize_cost=True)
    assert complex_model == "gpt-4"

def test_query_complexity_classification():
    orchestrator = LLMOrchestrator()
    
    # Test simple
    assert orchestrator.classify_query_complexity("Hello") == "simple"
    
    # Test moderate
    moderate_text = "Can you explain how this works? " * 3
    assert orchestrator.classify_query_complexity(moderate_text) == "moderate"
    
    # Test complex
    complex_text = "Detailed explanation " * 50
    assert orchestrator.classify_query_complexity(complex_text) == "complex"