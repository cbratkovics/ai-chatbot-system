from app.services.functions.base import function_registry
from app.services.functions.calculator import CalculatorFunction, DataAnalysisFunction
from app.services.functions.web_search import WebSearchFunction, WebScraperFunction
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def initialize_functions():
    """Initialize and register all available functions"""
    
    # Register calculator functions
    function_registry.register(CalculatorFunction())
    function_registry.register(DataAnalysisFunction())
    
    # Register web functions
    # These would need API keys in production
    search_api_key = getattr(settings, 'google_search_api_key', None)
    search_engine_id = getattr(settings, 'google_search_engine_id', None)
    
    function_registry.register(WebSearchFunction(
        search_api_key=search_api_key,
        search_engine_id=search_engine_id
    ))
    function_registry.register(WebScraperFunction())
    
    logger.info(f"Initialized {len(function_registry._functions)} functions")
    return function_registry