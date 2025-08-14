from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)

class FunctionParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: List[FunctionParameter]
    returns: str = "string"

class FunctionResult(BaseModel):
    name: str
    result: Any
    error: Optional[str] = None
    cached: bool = False

class BaseFunction(ABC):
    """Abstract base class for functions that can be called by LLMs"""
    
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.cache_enabled = True
    
    @abstractmethod
    def get_definition(self) -> FunctionDefinition:
        """Return the function definition for the LLM"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the function with given parameters"""
        pass
    
    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function calling schema"""
        definition = self.get_definition()
        
        properties = {}
        required = []
        
        for param in definition.parameters:
            prop_schema = {"type": param.type, "description": param.description}
            if param.enum:
                prop_schema["enum"] = param.enum
            properties[param.name] = prop_schema
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": definition.name,
            "description": definition.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

class FunctionRegistry:
    """Registry for managing available functions"""
    
    def __init__(self):
        self._functions: Dict[str, BaseFunction] = {}
        self._function_cache: Dict[str, Any] = {}
        
    def register(self, function: BaseFunction):
        """Register a function"""
        self._functions[function.get_definition().name] = function
        logger.info(f"Registered function: {function.get_definition().name}")
    
    def get_function(self, name: str) -> Optional[BaseFunction]:
        """Get a function by name"""
        return self._functions.get(name)
    
    def get_all_schemas(self) -> List[Dict]:
        """Get OpenAI schemas for all registered functions"""
        return [func.to_openai_schema() for func in self._functions.values()]
    
    async def execute_function(self, name: str, arguments: Dict) -> FunctionResult:
        """Execute a function by name with arguments"""
        function = self.get_function(name)
        if not function:
            return FunctionResult(
                name=name,
                result=None,
                error=f"Function '{name}' not found"
            )
        
        # Check cache
        cache_key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        if function.cache_enabled and cache_key in self._function_cache:
            logger.info(f"Cache hit for function: {name}")
            return FunctionResult(
                name=name,
                result=self._function_cache[cache_key],
                cached=True
            )
        
        try:
            result = await function.execute(**arguments)
            
            # Cache result
            if function.cache_enabled:
                self._function_cache[cache_key] = result
            
            return FunctionResult(name=name, result=result)
        except Exception as e:
            logger.error(f"Error executing function {name}: {str(e)}")
            return FunctionResult(
                name=name,
                result=None,
                error=str(e)
            )
    
    def clear_cache(self):
        """Clear function result cache"""
        self._function_cache.clear()

# Global registry instance
function_registry = FunctionRegistry()