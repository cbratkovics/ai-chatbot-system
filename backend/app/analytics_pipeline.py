"""
Advanced Monitoring and Analytics Pipeline
Real-time analytics, quality scoring, and pattern detection
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from textstat import flesch_reading_ease, flesch_kincaid_grade
import aioredis
from prometheus_client import Counter, Histogram, Gauge, Summary
import logging

logger = logging.getLogger(__name__)

# Metrics
analytics_events = Counter('analytics_events_total', 'Total analytics events', ['event_type'])
quality_scores = Histogram('conversation_quality_score', 'Conversation quality scores', buckets=[0.1, 0.3, 0.5, 0.7, 0.9])
satisfaction_scores = Histogram('user_satisfaction_score', 'User satisfaction scores', buckets=[1, 2, 3, 4, 5])
anomaly_detections = Counter('anomaly_detections_total', 'Anomaly detections', ['anomaly_type'])
pattern_discoveries = Counter('pattern_discoveries_total', 'Pattern discoveries', ['pattern_type'])


class AnalyticsEventType(Enum):
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    ERROR_OCCURRED = "error_occurred"
    QUALITY_SCORED = "quality_scored"
    FEEDBACK_RECEIVED = "feedback_received"
    

class QualityDimension(Enum):
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    SAFETY = "safety"
    ENGAGEMENT = "engagement"
    

class AnomalyType(Enum):
    USAGE_SPIKE = "usage_spike"
    ERROR_PATTERN = "error_pattern"
    QUALITY_DROP = "quality_drop"
    COST_ANOMALY = "cost_anomaly"
    LATENCY_ANOMALY = "latency_anomaly"
    

@dataclass
class ConversationMetrics:
    conversation_id: str
    user_id: str
    tenant_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    message_count: int = 0
    token_count: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    quality_score: float = 0.0
    satisfaction_score: Optional[float] = None
    error_count: int = 0
    model_switches: int = 0
    cache_hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class QualityReport:
    conversation_id: str
    timestamp: datetime
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    

@dataclass
class AnalyticsInsight:
    type: str
    title: str
    description: str
    impact: str  # high, medium, low
    data: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    

class AdvancedAnalytics:
    """Advanced analytics pipeline for conversational AI"""
    
    def __init__(self, redis: aioredis.Redis, config: Dict[str, Any]):
        self.redis = redis
        self.config = config
        
        # ML models for analytics
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.pattern_detector = DBSCAN(
            eps=0.3,
            min_samples=5
        )
        
        # Analytics state
        self.conversation_metrics: Dict[str, ConversationMetrics] = {}
        self.quality_history: List[float] = []
        self.error_patterns: Dict[str, int] = {}
        
        # Background tasks
        self._aggregation_task: Optional[asyncio.Task] = None
        self._anomaly_detection_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize analytics pipeline"""
        # Start background tasks
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self._anomaly_detection_task = asyncio.create_task(self._anomaly_detection_loop())
        
        logger.info("Analytics pipeline initialized")
        
    async def shutdown(self):
        """Shutdown analytics pipeline"""
        if self._aggregation_task:
            self._aggregation_task.cancel()
        if self._anomaly_detection_task:
            self._anomaly_detection_task.cancel()
            
    # Event Tracking
    
    async def track_event(
        self,
        event_type: AnalyticsEventType,
        conversation_id: str,
        user_id: str,
        tenant_id: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Track analytics event"""
        event = {
            "type": event_type.value,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {}
        }
        
        # Store event
        await self.redis.zadd(
            f"analytics:events:{tenant_id}",
            {json.dumps(event): time.time()}
        )
        
        # Update metrics
        analytics_events.labels(event_type=event_type.value).inc()
        
        # Update conversation metrics
        await self._update_conversation_metrics(event)
        
    async def _update_conversation_metrics(self, event: Dict[str, Any]):
        """Update real-time conversation metrics"""
        conv_id = event["conversation_id"]
        
        if conv_id not in self.conversation_metrics:
            self.conversation_metrics[conv_id] = ConversationMetrics(
                conversation_id=conv_id,
                user_id=event["user_id"],
                tenant_id=event["tenant_id"],
                start_time=datetime.fromisoformat(event["timestamp"])
            )
            
        metrics = self.conversation_metrics[conv_id]
        
        if event["type"] == AnalyticsEventType.MESSAGE_SENT.value:
            metrics.message_count += 1
            metrics.token_count += event["data"].get("tokens", 0)
            metrics.total_cost += event["data"].get("cost", 0)
            
        elif event["type"] == AnalyticsEventType.ERROR_OCCURRED.value:
            metrics.error_count += 1
            
        elif event["type"] == AnalyticsEventType.CONVERSATION_END.value:
            metrics.end_time = datetime.fromisoformat(event["timestamp"])
            # Calculate final metrics
            await self._finalize_conversation_metrics(metrics)
            
    async def _finalize_conversation_metrics(self, metrics: ConversationMetrics):
        """Finalize and store conversation metrics"""
        # Calculate quality score
        quality_report = await self.score_conversation_quality(
            metrics.conversation_id,
            await self._get_conversation_messages(metrics.conversation_id)
        )
        metrics.quality_score = quality_report.overall_score
        
        # Store metrics
        await self.redis.setex(
            f"analytics:conversation:{metrics.conversation_id}",
            86400 * 30,  # 30 day retention
            json.dumps({
                "conversation_id": metrics.conversation_id,
                "user_id": metrics.user_id,
                "tenant_id": metrics.tenant_id,
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
                "duration_seconds": (metrics.end_time - metrics.start_time).total_seconds() if metrics.end_time else 0,
                "message_count": metrics.message_count,
                "token_count": metrics.token_count,
                "total_cost": metrics.total_cost,
                "quality_score": metrics.quality_score,
                "satisfaction_score": metrics.satisfaction_score,
                "error_count": metrics.error_count,
                "cache_hit_rate": metrics.cache_hits / max(metrics.message_count, 1)
            })
        )
        
        # Update aggregates
        await self._update_aggregates(metrics)
        
    # Quality Scoring
    
    async def score_conversation_quality(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> QualityReport:
        """Score conversation quality using ML model"""
        dimension_scores = {}
        issues = []
        
        # Relevance scoring
        relevance_score = await self._score_relevance(messages)
        dimension_scores[QualityDimension.RELEVANCE] = relevance_score
        
        # Coherence scoring
        coherence_score = await self._score_coherence(messages)
        dimension_scores[QualityDimension.COHERENCE] = coherence_score
        
        # Completeness scoring
        completeness_score = await self._score_completeness(messages)
        dimension_scores[QualityDimension.COMPLETENESS] = completeness_score
        
        # Safety scoring
        safety_score = await self._score_safety(messages)
        dimension_scores[QualityDimension.SAFETY] = safety_score
        
        # Engagement scoring
        engagement_score = await self._score_engagement(messages)
        dimension_scores[QualityDimension.ENGAGEMENT] = engagement_score
        
        # Calculate overall score (weighted average)
        weights = {
            QualityDimension.RELEVANCE: 0.3,
            QualityDimension.COHERENCE: 0.2,
            QualityDimension.COMPLETENESS: 0.2,
            QualityDimension.SAFETY: 0.2,
            QualityDimension.ENGAGEMENT: 0.1
        }
        
        overall_score = sum(
            dimension_scores.get(dim, 0) * weight
            for dim, weight in weights.items()
        )
        
        # Identify issues
        for dim, score in dimension_scores.items():
            if score < 0.5:
                issues.append({
                    "dimension": dim.value,
                    "score": score,
                    "severity": "high" if score < 0.3 else "medium"
                })
                
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(
            dimension_scores,
            issues
        )
        
        report = QualityReport(
            conversation_id=conversation_id,
            timestamp=datetime.now(timezone.utc),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues=issues,
            recommendations=recommendations
        )
        
        # Record metrics
        quality_scores.observe(overall_score)
        analytics_events.labels(event_type="quality_scored").inc()
        
        return report
        
    async def _score_relevance(self, messages: List[Dict[str, Any]]) -> float:
        """Score relevance of responses to queries"""
        if len(messages) < 2:
            return 0.5
            
        scores = []
        
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                query = messages[i].get("content", "")
                response = messages[i + 1].get("content", "")
                
                # Simple relevance scoring (in production, use embeddings)
                query_tokens = set(query.lower().split())
                response_tokens = set(response.lower().split())
                
                if query_tokens:
                    overlap = len(query_tokens.intersection(response_tokens))
                    score = overlap / len(query_tokens)
                    scores.append(min(score * 2, 1.0))  # Scale up
                    
        return statistics.mean(scores) if scores else 0.5
        
    async def _score_coherence(self, messages: List[Dict[str, Any]]) -> float:
        """Score coherence and flow of conversation"""
        if len(messages) < 2:
            return 0.5
            
        # Check readability
        readability_scores = []
        
        for msg in messages:
            if msg.get("role") == "assistant":
                text = msg.get("content", "")
                if text:
                    # Flesch Reading Ease (0-100, higher is easier)
                    try:
                        score = flesch_reading_ease(text)
                        # Normalize to 0-1 (target range 60-80)
                        normalized = max(0, min(1, (score - 30) / 50))
                        readability_scores.append(normalized)
                    except:
                        readability_scores.append(0.7)
                        
        return statistics.mean(readability_scores) if readability_scores else 0.5
        
    async def _score_completeness(self, messages: List[Dict[str, Any]]) -> float:
        """Score completeness of responses"""
        scores = []
        
        for msg in messages:
            if msg.get("role") == "assistant":
                response = msg.get("content", "")
                
                # Check for incomplete responses
                incomplete_indicators = [
                    "...",
                    "I'm not sure",
                    "I don't know",
                    "unclear",
                    "more information needed"
                ]
                
                penalties = sum(
                    0.1 for indicator in incomplete_indicators
                    if indicator.lower() in response.lower()
                )
                
                # Check response length (too short might be incomplete)
                word_count = len(response.split())
                if word_count < 10:
                    penalties += 0.2
                    
                scores.append(max(0, 1 - penalties))
                
        return statistics.mean(scores) if scores else 0.5
        
    async def _score_safety(self, messages: List[Dict[str, Any]]) -> float:
        """Score safety of conversation"""
        # In production, integrate with safety pipeline
        # For now, simple keyword check
        unsafe_keywords = [
            "confidential", "password", "ssn", "credit card",
            "hack", "exploit", "vulnerability"
        ]
        
        violations = 0
        total_messages = 0
        
        for msg in messages:
            content = msg.get("content", "").lower()
            total_messages += 1
            
            for keyword in unsafe_keywords:
                if keyword in content:
                    violations += 1
                    break
                    
        if total_messages == 0:
            return 1.0
            
        return max(0, 1 - (violations / total_messages))
        
    async def _score_engagement(self, messages: List[Dict[str, Any]]) -> float:
        """Score user engagement level"""
        if len(messages) < 4:
            return 0.5
            
        # Analyze conversation dynamics
        user_msg_lengths = []
        response_times = []
        
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                user_msg_lengths.append(len(msg.get("content", "").split()))
                
                # Check if user asks follow-up questions
                if "?" in msg.get("content", ""):
                    response_times.append(1)  # Engaged
                else:
                    response_times.append(0.5)  # Neutral
                    
        # Increasing message length indicates engagement
        if len(user_msg_lengths) > 1:
            length_trend = np.polyfit(range(len(user_msg_lengths)), user_msg_lengths, 1)[0]
            engagement_score = min(1, max(0, 0.5 + length_trend / 10))
        else:
            engagement_score = 0.5
            
        return engagement_score
        
    def _generate_quality_recommendations(
        self,
        dimension_scores: Dict[QualityDimension, float],
        issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on quality analysis"""
        recommendations = []
        
        if dimension_scores.get(QualityDimension.RELEVANCE, 1) < 0.5:
            recommendations.append(
                "Consider using more advanced models for complex queries to improve relevance"
            )
            
        if dimension_scores.get(QualityDimension.COHERENCE, 1) < 0.5:
            recommendations.append(
                "Adjust temperature settings to improve response coherence"
            )
            
        if dimension_scores.get(QualityDimension.COMPLETENESS, 1) < 0.5:
            recommendations.append(
                "Increase max_tokens or use follow-up prompts for more complete responses"
            )
            
        if dimension_scores.get(QualityDimension.SAFETY, 1) < 0.7:
            recommendations.append(
                "Enable stricter content filtering to improve safety scores"
            )
            
        return recommendations
        
    # User Satisfaction Analysis
    
    async def record_user_feedback(
        self,
        conversation_id: str,
        rating: int,  # 1-5
        feedback_text: Optional[str] = None
    ):
        """Record user satisfaction feedback"""
        # Update conversation metrics
        if conversation_id in self.conversation_metrics:
            self.conversation_metrics[conversation_id].satisfaction_score = rating
            
        # Store feedback
        feedback_data = {
            "conversation_id": conversation_id,
            "rating": rating,
            "feedback_text": feedback_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.redis.zadd(
            "analytics:feedback",
            {json.dumps(feedback_data): time.time()}
        )
        
        # Update metrics
        satisfaction_scores.observe(rating)
        analytics_events.labels(event_type="feedback_received").inc()
        
        # Analyze sentiment if text provided
        if feedback_text:
            sentiment = await self._analyze_feedback_sentiment(feedback_text)
            if sentiment < 0.3 and rating >= 4:
                # Mismatch between rating and sentiment
                logger.warning(f"Feedback mismatch: rating={rating}, sentiment={sentiment}")
                
    async def _analyze_feedback_sentiment(self, text: str) -> float:
        """Analyze sentiment of feedback text"""
        # Simple sentiment analysis (in production, use proper NLP)
        positive_words = ["good", "great", "excellent", "helpful", "useful", "amazing"]
        negative_words = ["bad", "poor", "terrible", "useless", "awful", "horrible"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5
            
        return positive_count / (positive_count + negative_count)
        
    # Pattern Detection and Insights
    
    async def detect_conversation_patterns(
        self,
        tenant_id: str,
        time_window: timedelta = timedelta(days=7)
    ) -> List[AnalyticsInsight]:
        """Detect patterns in conversation data"""
        insights = []
        
        # Get recent conversations
        end_time = datetime.now(timezone.utc)
        start_time = end_time - time_window
        
        conversations = await self._get_conversations_in_range(
            tenant_id,
            start_time,
            end_time
        )
        
        if len(conversations) < 10:
            return insights
            
        # Pattern 1: Peak usage times
        usage_pattern = self._analyze_usage_patterns(conversations)
        if usage_pattern:
            insights.append(usage_pattern)
            
        # Pattern 2: Common error patterns
        error_pattern = self._analyze_error_patterns(conversations)
        if error_pattern:
            insights.append(error_pattern)
            
        # Pattern 3: Cost optimization opportunities
        cost_pattern = self._analyze_cost_patterns(conversations)
        if cost_pattern:
            insights.append(cost_pattern)
            
        # Pattern 4: Quality trends
        quality_pattern = self._analyze_quality_trends(conversations)
        if quality_pattern:
            insights.append(quality_pattern)
            
        return insights
        
    def _analyze_usage_patterns(self, conversations: List[Dict[str, Any]]) -> Optional[AnalyticsInsight]:
        """Analyze usage patterns"""
        # Group by hour of day
        hourly_usage = {}
        
        for conv in conversations:
            start_time = datetime.fromisoformat(conv["start_time"])
            hour = start_time.hour
            hourly_usage[hour] = hourly_usage.get(hour, 0) + 1
            
        # Find peak hours
        peak_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if peak_hours and peak_hours[0][1] > len(conversations) * 0.2:
            return AnalyticsInsight(
                type="usage_pattern",
                title="Peak Usage Hours Detected",
                description=f"Highest usage at {peak_hours[0][0]}:00 with {peak_hours[0][1]} conversations",
                impact="medium",
                data={"hourly_usage": hourly_usage, "peak_hours": peak_hours},
                recommendations=[
                    "Scale up resources during peak hours",
                    "Implement request queuing for better load distribution",
                    "Consider geographic distribution of load"
                ]
            )
            
        return None
        
    def _analyze_error_patterns(self, conversations: List[Dict[str, Any]]) -> Optional[AnalyticsInsight]:
        """Analyze error patterns"""
        error_types = {}
        error_conversations = []
        
        for conv in conversations:
            if conv.get("error_count", 0) > 0:
                error_conversations.append(conv)
                # In production, analyze actual error types
                error_type = "timeout" if conv.get("duration_seconds", 0) > 30 else "api_error"
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
        if len(error_conversations) > len(conversations) * 0.1:
            return AnalyticsInsight(
                type="error_pattern",
                title="High Error Rate Detected",
                description=f"{len(error_conversations)} conversations with errors ({len(error_conversations)/len(conversations)*100:.1f}%)",
                impact="high",
                data={"error_types": error_types, "error_rate": len(error_conversations)/len(conversations)},
                recommendations=[
                    "Investigate root cause of errors",
                    "Implement retry logic for transient failures",
                    "Add circuit breakers for external services"
                ]
            )
            
        return None
        
    def _analyze_cost_patterns(self, conversations: List[Dict[str, Any]]) -> Optional[AnalyticsInsight]:
        """Analyze cost optimization opportunities"""
        total_cost = sum(conv.get("total_cost", 0) for conv in conversations)
        avg_cost = total_cost / len(conversations) if conversations else 0
        
        high_cost_convs = [
            conv for conv in conversations
            if conv.get("total_cost", 0) > avg_cost * 2
        ]
        
        cache_hit_rates = [
            conv.get("cache_hit_rate", 0) for conv in conversations
        ]
        avg_cache_hit = statistics.mean(cache_hit_rates) if cache_hit_rates else 0
        
        if avg_cache_hit < 0.3 or len(high_cost_convs) > len(conversations) * 0.2:
            potential_savings = total_cost * 0.3  # Estimate 30% savings
            
            return AnalyticsInsight(
                type="cost_pattern",
                title="Cost Optimization Opportunity",
                description=f"Low cache hit rate ({avg_cache_hit:.1%}) and {len(high_cost_convs)} high-cost conversations",
                impact="high",
                data={
                    "total_cost": total_cost,
                    "potential_savings": potential_savings,
                    "cache_hit_rate": avg_cache_hit
                },
                recommendations=[
                    "Increase semantic cache similarity threshold",
                    "Implement response compression for long conversations",
                    "Use cheaper models for simple queries"
                ]
            )
            
        return None
        
    def _analyze_quality_trends(self, conversations: List[Dict[str, Any]]) -> Optional[AnalyticsInsight]:
        """Analyze quality trends"""
        quality_scores = [
            conv.get("quality_score", 0) for conv in conversations
            if conv.get("quality_score") is not None
        ]
        
        if len(quality_scores) < 5:
            return None
            
        avg_quality = statistics.mean(quality_scores)
        quality_trend = np.polyfit(range(len(quality_scores)), quality_scores, 1)[0]
        
        if quality_trend < -0.01 or avg_quality < 0.6:
            return AnalyticsInsight(
                type="quality_pattern",
                title="Quality Degradation Detected",
                description=f"Average quality score {avg_quality:.2f} with {'declining' if quality_trend < 0 else 'stable'} trend",
                impact="high",
                data={
                    "average_quality": avg_quality,
                    "trend": quality_trend,
                    "scores": quality_scores[-10:]  # Last 10 scores
                },
                recommendations=[
                    "Review recent model or prompt changes",
                    "Increase quality monitoring frequency",
                    "Consider A/B testing different configurations"
                ]
            )
            
        return None
        
    # Anomaly Detection
    
    async def detect_anomalies(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Detect anomalies in real-time metrics"""
        anomalies = []
        
        # Get recent metrics
        metrics = await self._get_recent_metrics(tenant_id, hours=24)
        
        if len(metrics) < 10:
            return anomalies
            
        # Extract features for anomaly detection
        features = []
        for m in metrics:
            features.append([
                m.get("message_count", 0),
                m.get("total_cost", 0),
                m.get("error_count", 0),
                m.get("duration_seconds", 0),
                m.get("quality_score", 0.5)
            ])
            
        features_array = np.array(features)
        
        # Detect anomalies
        if len(features) > 20:
            self.anomaly_detector.fit(features_array[:-10])  # Train on older data
            predictions = self.anomaly_detector.predict(features_array[-10:])  # Predict on recent
            
            for i, pred in enumerate(predictions):
                if pred == -1:  # Anomaly
                    idx = len(features) - 10 + i
                    anomaly_data = metrics[idx]
                    
                    # Determine anomaly type
                    if anomaly_data.get("total_cost", 0) > statistics.mean([m.get("total_cost", 0) for m in metrics]) * 3:
                        anomaly_type = AnomalyType.COST_ANOMALY
                    elif anomaly_data.get("error_count", 0) > 5:
                        anomaly_type = AnomalyType.ERROR_PATTERN
                    else:
                        anomaly_type = AnomalyType.USAGE_SPIKE
                        
                    anomalies.append({
                        "type": anomaly_type.value,
                        "timestamp": anomaly_data.get("timestamp"),
                        "data": anomaly_data,
                        "severity": "high"
                    })
                    
                    anomaly_detections.labels(anomaly_type=anomaly_type.value).inc()
                    
        return anomalies
        
    # Aggregation and Reporting
    
    async def generate_analytics_report(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        # Get conversations in range
        conversations = await self._get_conversations_in_range(
            tenant_id,
            start_date,
            end_date
        )
        
        if not conversations:
            return {"error": "No data available for the specified period"}
            
        # Calculate aggregate metrics
        total_conversations = len(conversations)
        total_messages = sum(c.get("message_count", 0) for c in conversations)
        total_tokens = sum(c.get("token_count", 0) for c in conversations)
        total_cost = sum(c.get("total_cost", 0) for c in conversations)
        
        quality_scores = [c.get("quality_score", 0) for c in conversations if c.get("quality_score")]
        satisfaction_scores = [c.get("satisfaction_score", 0) for c in conversations if c.get("satisfaction_score")]
        
        # Model usage breakdown
        model_usage = {}
        for conv in conversations:
            model = conv.get("metadata", {}).get("primary_model", "unknown")
            model_usage[model] = model_usage.get(model, 0) + 1
            
        # Error analysis
        error_rate = sum(1 for c in conversations if c.get("error_count", 0) > 0) / total_conversations
        
        # Cost analysis
        daily_costs = {}
        for conv in conversations:
            date = datetime.fromisoformat(conv["start_time"]).date()
            daily_costs[str(date)] = daily_costs.get(str(date), 0) + conv.get("total_cost", 0)
            
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "summary": {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 2),
                "avg_cost_per_conversation": round(total_cost / total_conversations, 2) if total_conversations else 0,
                "avg_messages_per_conversation": round(total_messages / total_conversations, 1) if total_conversations else 0
            },
            "quality": {
                "average_score": round(statistics.mean(quality_scores), 2) if quality_scores else None,
                "min_score": round(min(quality_scores), 2) if quality_scores else None,
                "max_score": round(max(quality_scores), 2) if quality_scores else None,
                "satisfaction_average": round(statistics.mean(satisfaction_scores), 1) if satisfaction_scores else None
            },
            "reliability": {
                "error_rate": round(error_rate, 3),
                "success_rate": round(1 - error_rate, 3),
                "availability": 0.999  # Would be calculated from uptime monitoring
            },
            "model_usage": model_usage,
            "daily_costs": daily_costs,
            "insights": await self.detect_conversation_patterns(tenant_id, end_date - start_date),
            "anomalies": await self.detect_anomalies(tenant_id)
        }
        
        return report
        
    # Background Tasks
    
    async def _aggregation_loop(self):
        """Periodic aggregation of metrics"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Aggregate metrics for all active tenants
                tenants = await self._get_active_tenants()
                
                for tenant_id in tenants:
                    await self._aggregate_tenant_metrics(tenant_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
                
    async def _anomaly_detection_loop(self):
        """Periodic anomaly detection"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Check for anomalies in real-time metrics
                tenants = await self._get_active_tenants()
                
                for tenant_id in tenants:
                    anomalies = await self.detect_anomalies(tenant_id)
                    
                    if anomalies:
                        # Send alerts
                        await self._send_anomaly_alerts(tenant_id, anomalies)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                
    # Helper Methods
    
    async def _get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get messages for a conversation"""
        # In production, retrieve from message store
        messages_data = await self.redis.get(f"conversation:messages:{conversation_id}")
        if messages_data:
            return json.loads(messages_data)
        return []
        
    async def _get_conversations_in_range(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get conversations within time range"""
        conversations = []
        
        # Scan for conversation metrics
        cursor = 0
        pattern = f"analytics:conversation:*"
        
        while True:
            cursor, keys = await self.redis.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            if keys:
                values = await self.redis.mget(keys)
                for value in values:
                    if value:
                        conv = json.loads(value)
                        if conv.get("tenant_id") == tenant_id:
                            conv_time = datetime.fromisoformat(conv["start_time"])
                            if start_time <= conv_time <= end_time:
                                conversations.append(conv)
                                
            if cursor == 0:
                break
                
        return sorted(conversations, key=lambda x: x["start_time"])
        
    async def _get_recent_metrics(self, tenant_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent metrics for anomaly detection"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        return await self._get_conversations_in_range(tenant_id, start_time, end_time)
        
    async def _update_aggregates(self, metrics: ConversationMetrics):
        """Update aggregate metrics"""
        # Update hourly aggregates
        hour_key = metrics.start_time.strftime("%Y-%m-%d-%H")
        
        await self.redis.hincrby(
            f"analytics:hourly:{metrics.tenant_id}:{hour_key}",
            "conversations",
            1
        )
        await self.redis.hincrbyfloat(
            f"analytics:hourly:{metrics.tenant_id}:{hour_key}",
            "total_cost",
            metrics.total_cost
        )
        await self.redis.hincrby(
            f"analytics:hourly:{metrics.tenant_id}:{hour_key}",
            "messages",
            metrics.message_count
        )
        
        # Set expiry
        await self.redis.expire(
            f"analytics:hourly:{metrics.tenant_id}:{hour_key}",
            86400 * 90  # 90 days
        )
        
    async def _aggregate_tenant_metrics(self, tenant_id: str):
        """Aggregate metrics for a tenant"""
        # This would calculate daily/weekly/monthly rollups
        pass
        
    async def _get_active_tenants(self) -> List[str]:
        """Get list of active tenants"""
        # In production, retrieve from tenant registry
        return ["default"]
        
    async def _send_anomaly_alerts(self, tenant_id: str, anomalies: List[Dict[str, Any]]):
        """Send anomaly alerts"""
        for anomaly in anomalies:
            alert = {
                "tenant_id": tenant_id,
                "anomaly": anomaly,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await self.redis.publish("analytics:alerts", json.dumps(alert))
            logger.warning(f"Anomaly detected for {tenant_id}: {anomaly['type']}")