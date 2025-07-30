from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class QualityEvaluator:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.safety_keywords = [
            "harmful", "dangerous", "illegal", "unethical",
            "violence", "hate", "discrimination"
        ]
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        context: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate response quality on multiple dimensions"""
        
        evaluation = {
            "relevance": self._calculate_relevance(query, response),
            "safety": self._check_safety(response),
            "completeness": self._assess_completeness(response),
            "clarity": self._assess_clarity(response)
        }
        
        # Calculate overall score
        evaluation["overall"] = np.mean(list(evaluation.values()))
        
        logger.info(f"Quality evaluation: {evaluation}")
        return evaluation
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate semantic similarity between query and response"""
        query_embedding = self.encoder.encode(query)
        response_embedding = self.encoder.encode(response)
        
        similarity = np.dot(query_embedding, response_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)
        )
        
        return float(similarity)
    
    def _check_safety(self, response: str) -> float:
        """Check response for safety issues"""
        response_lower = response.lower()
        
        for keyword in self.safety_keywords:
            if keyword in response_lower:
                return 0.0
        
        return 1.0
    
    def _assess_completeness(self, response: str) -> float:
        """Assess if response seems complete"""
        # Simple heuristic based on length and punctuation
        if len(response) < 20:
            return 0.3
        
        if not response.strip().endswith(('.', '!', '?')):
            return 0.7
        
        return 1.0
    
    def _assess_clarity(self, response: str) -> float:
        """Assess response clarity"""
        # Simple heuristic based on sentence structure
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Optimal sentence length is 15-20 words
        if 15 <= avg_sentence_length <= 20:
            return 1.0
        elif 10 <= avg_sentence_length <= 30:
            return 0.8
        else:
            return 0.5