import aiohttp
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import json
from app.services.functions.base import BaseFunction, FunctionDefinition, FunctionParameter
import logging

logger = logging.getLogger(__name__)

class WebSearchFunction(BaseFunction):
    """Function to search the web and return relevant results"""
    
    def __init__(self, search_api_key: str = None, search_engine_id: str = None):
        super().__init__()
        self.api_key = search_api_key
        self.engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def get_definition(self) -> FunctionDefinition:
        return FunctionDefinition(
            name="web_search",
            description="Search the web for current information on any topic",
            parameters=[
                FunctionParameter(
                    name="query",
                    type="string",
                    description="The search query"
                ),
                FunctionParameter(
                    name="num_results",
                    type="integer",
                    description="Number of results to return (1-10)",
                    required=False
                )
            ],
            returns="array"
        )
    
    async def execute(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Execute web search and return results"""
        if not self.api_key or not self.engine_id:
            # Fallback to mock results for demo
            return self._get_mock_results(query)
        
        async with aiohttp.ClientSession() as session:
            params = {
                "key": self.api_key,
                "cx": self.engine_id,
                "q": query,
                "num": min(num_results, 10)
            }
            
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_results(data.get("items", []))
                    else:
                        logger.error(f"Search API error: {response.status}")
                        return self._get_mock_results(query)
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return self._get_mock_results(query)
    
    def _format_results(self, items: List[Dict]) -> List[Dict[str, Any]]:
        """Format search results"""
        results = []
        for item in items:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("displayLink", "")
            })
        return results
    
    def _get_mock_results(self, query: str) -> List[Dict[str, Any]]:
        """Return mock results for demo purposes"""
        return [
            {
                "title": f"Latest developments in {query}",
                "link": f"https://example.com/{query.replace(' ', '-')}",
                "snippet": f"Recent information about {query} shows significant progress...",
                "source": "example.com"
            },
            {
                "title": f"Understanding {query}: A comprehensive guide",
                "link": f"https://guide.com/{query.replace(' ', '-')}",
                "snippet": f"Everything you need to know about {query} in 2024...",
                "source": "guide.com"
            }
        ]

class WebScraperFunction(BaseFunction):
    """Function to scrape content from a specific URL"""
    
    def get_definition(self) -> FunctionDefinition:
        return FunctionDefinition(
            name="web_scraper",
            description="Extract and read content from a specific webpage",
            parameters=[
                FunctionParameter(
                    name="url",
                    type="string",
                    description="The URL to scrape"
                ),
                FunctionParameter(
                    name="selector",
                    type="string",
                    description="CSS selector to extract specific content",
                    required=False
                )
            ],
            returns="object"
        )
    
    async def execute(self, url: str, selector: str = None) -> Dict[str, Any]:
        """Scrape content from URL"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        if selector:
                            elements = soup.select(selector)
                            content = '\n'.join([elem.get_text(strip=True) for elem in elements])
                        else:
                            # Extract main content
                            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                                tag.decompose()
                            content = soup.get_text(strip=True)
                        
                        # Limit content length
                        if len(content) > 5000:
                            content = content[:5000] + "..."
                        
                        return {
                            "url": url,
                            "title": soup.title.string if soup.title else "No title",
                            "content": content,
                            "length": len(content)
                        }
                    else:
                        return {
                            "url": url,
                            "error": f"HTTP {response.status}",
                            "content": None
                        }
            except Exception as e:
                logger.error(f"Scraping error: {str(e)}")
                return {
                    "url": url,
                    "error": str(e),
                    "content": None
                }