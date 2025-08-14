"""
AI Safety and Compliance Framework
Content filtering, bias detection, PII detection, and audit trails
"""

import asyncio
import json
import logging
import time
import uuid
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import sqlite3
import aiofiles
import asyncpg
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import spacy
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import openai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import requests
import base64
import os

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"

class ContentCategory(Enum):
    SAFE = "safe"
    INAPPROPRIATE = "inappropriate"
    HARMFUL = "harmful"
    ILLEGAL = "illegal"
    SPAM = "spam"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    ADULT_CONTENT = "adult_content"
    PERSONAL_INFO = "personal_info"

class BiasType(Enum):
    GENDER = "gender"
    RACE = "race"
    AGE = "age"
    RELIGION = "religion"
    POLITICAL = "political"
    SOCIOECONOMIC = "socioeconomic"
    CULTURAL = "cultural"

@dataclass
class SafetyResult:
    is_safe: bool
    confidence: float
    categories: List[ContentCategory]
    reasons: List[str]
    severity: SafetyLevel
    recommendations: List[str] = field(default_factory=list)

@dataclass
class BiasAnalysisResult:
    has_bias: bool
    bias_types: List[BiasType]
    confidence: float
    examples: List[str]
    mitigation_suggestions: List[str]

@dataclass
class PIIAnalysisResult:
    has_pii: bool
    pii_entities: List[Dict[str, Any]]
    anonymized_text: str
    confidence: float
    risk_level: SafetyLevel

@dataclass
class ComplianceResult:
    is_compliant: bool
    standards: List[ComplianceStandard]
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_score: float

@dataclass
class AuditLogEntry:
    id: str
    timestamp: datetime
    user_id: str
    session_id: str
    action: str
    resource: str
    result: str
    metadata: Dict[str, Any]
    ip_address: str
    user_agent: str

class AISafetyComplianceFramework:
    """
    Comprehensive AI safety and compliance framework with multiple detection engines
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Database connections
        self.db_pool = None
        
        # Encryption key for sensitive data
        self.encryption_key = self._derive_encryption_key(config['encryption_password'])
        self.cipher = Fernet(self.encryption_key)
        
        # Load ML models
        self.content_classifier = None
        self.bias_detector = None
        self.toxicity_detector = None
        self.prompt_injection_detector = None
        
        # NLP models
        self.nlp = None
        
        # PII detection engines
        self.pii_analyzer = AnalyzerEngine()
        self.pii_anonymizer = AnonymizerEngine()
        
        # Compliance rules
        self.compliance_rules = self._load_compliance_rules()
        
        # Content filters
        self.content_filters = self._load_content_filters()
        
        # Audit log settings
        self.audit_enabled = config.get('audit_enabled', True)
        self.audit_retention_days = config.get('audit_retention_days', 2555)  # 7 years
        
        # Rate limiting for safety checks
        self.rate_limits = {}
        
    async def initialize(self):
        """Initialize the safety and compliance framework"""
        
        # Initialize database connection
        await self._initialize_database()
        
        # Load ML models
        await self._load_ml_models()
        
        # Load NLP models
        await self._load_nlp_models()
        
        # Start background tasks
        asyncio.create_task(self._audit_log_cleanup())
        asyncio.create_task(self._model_performance_monitoring())
        
        logger.info("AI Safety and Compliance Framework initialized")
        
    async def _initialize_database(self):
        """Initialize database connections and tables"""
        try:
            # PostgreSQL connection for audit logs
            self.db_pool = await asyncpg.create_pool(
                self.config['database_url'],
                min_size=5,
                max_size=20
            )
            
            # Create audit log table
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id UUID PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        user_id VARCHAR(255),
                        session_id VARCHAR(255),
                        action VARCHAR(255) NOT NULL,
                        resource VARCHAR(255),
                        result VARCHAR(255),
                        metadata JSONB,
                        ip_address INET,
                        user_agent TEXT,
                        encrypted_data BYTEA
                    )
                ''')
                
                # Create indexes
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)')
                
                # Create safety analysis results table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS safety_analysis (
                        id UUID PRIMARY KEY,
                        content_hash VARCHAR(64) NOT NULL,
                        analysis_type VARCHAR(50) NOT NULL,
                        result JSONB NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        model_version VARCHAR(50)
                    )
                ''')
                
                # Create compliance violations table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS compliance_violations (
                        id UUID PRIMARY KEY,
                        violation_type VARCHAR(100) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        description TEXT,
                        user_id VARCHAR(255),
                        session_id VARCHAR(255),
                        timestamp TIMESTAMPTZ NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolution_notes TEXT
                    )
                ''')
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
            
    async def _load_ml_models(self):
        """Load machine learning models for safety analysis"""
        try:
            # Content classification model
            self.content_classifier = pipeline(
                "text-classification",
                model=self.config.get('content_model', 'unitary/toxic-bert'),
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Bias detection model
            self.bias_detector = pipeline(
                "text-classification",
                model=self.config.get('bias_model', 'unitary/unbiased-toxic-roberta'),
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Toxicity detection
            self.toxicity_detector = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Prompt injection detection
            self.prompt_injection_detector = pipeline(
                "text-classification",
                model="deepset/deberta-v3-base-injection",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"ML model loading error: {e}")
            # Fall back to rule-based detection
            
    async def _load_nlp_models(self):
        """Load NLP models for text analysis"""
        try:
            # Load spaCy model for NER and linguistic analysis
            self.nlp = spacy.load("en_core_web_sm")
            
        except Exception as e:
            logger.error(f"NLP model loading error: {e}")
            
    def _derive_encryption_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        password_bytes = password.encode()
        salt = b'safety_compliance_salt'  # In production, use random salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
        
    def _load_compliance_rules(self) -> Dict[ComplianceStandard, Dict[str, Any]]:
        """Load compliance rules for different standards"""
        return {
            ComplianceStandard.GDPR: {
                'pii_detection_required': True,
                'data_retention_days': 1095,  # 3 years
                'consent_required': True,
                'right_to_erasure': True,
                'data_portability': True,
                'breach_notification_hours': 72
            },
            ComplianceStandard.HIPAA: {
                'phi_detection_required': True,
                'encryption_required': True,
                'access_logging_required': True,
                'minimum_necessary_standard': True,
                'breach_notification_days': 60
            },
            ComplianceStandard.CCPA: {
                'personal_info_detection': True,
                'opt_out_required': True,
                'data_disclosure_tracking': True,
                'data_retention_limits': True
            },
            ComplianceStandard.SOX: {
                'financial_data_protection': True,
                'audit_trail_required': True,
                'data_integrity_controls': True,
                'whistleblower_protection': True
            }
        }
        
    def _load_content_filters(self) -> Dict[str, Any]:
        """Load content filtering rules"""
        return {
            'profanity_words': self._load_profanity_list(),
            'hate_speech_patterns': self._load_hate_speech_patterns(),
            'violence_keywords': self._load_violence_keywords(),
            'adult_content_patterns': self._load_adult_content_patterns(),
            'spam_indicators': self._load_spam_indicators()
        }
        
    def _load_profanity_list(self) -> Set[str]:
        """Load profanity word list"""
        # This would typically load from a comprehensive database
        return {
            'damn', 'hell', 'crap', 'shit', 'fuck', 'bitch', 'asshole',
            'bastard', 'whore', 'slut', 'nigger', 'faggot', 'retard'
        }
        
    def _load_hate_speech_patterns(self) -> List[str]:
        """Load hate speech detection patterns"""
        return [
            r'\b(kill|murder|die)\s+(all\s+)?(jews|muslims|christians|blacks|whites)\b',
            r'\b(hate|despise)\s+(all\s+)?(jews|muslims|christians|blacks|whites)\b',
            r'\b(inferior|superior)\s+race\b',
            r'\bgas\s+chamber\b',
            r'\bfinal\s+solution\b'
        ]
        
    def _load_violence_keywords(self) -> Set[str]:
        """Load violence-related keywords"""
        return {
            'kill', 'murder', 'assassinate', 'torture', 'bomb', 'explosion',
            'weapon', 'gun', 'knife', 'violence', 'harm', 'hurt', 'attack'
        }
        
    def _load_adult_content_patterns(self) -> List[str]:
        """Load adult content detection patterns"""
        return [
            r'\b(sex|sexual|porn|pornography|nude|naked)\b',
            r'\b(breast|penis|vagina|anal|oral)\b',
            r'\b(masturbat|orgasm|climax)\b'
        ]
        
    def _load_spam_indicators(self) -> List[str]:
        """Load spam detection indicators"""
        return [
            r'click here now',
            r'limited time offer',
            r'make money fast',
            r'free money',
            r'guaranteed results',
            r'act now',
            r'urgent action required'
        ]
        
    async def analyze_content_safety(self, content: str, context: Dict[str, Any] = None) -> SafetyResult:
        """
        Comprehensive content safety analysis
        """
        try:
            # Generate content hash for caching
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check cache first
            cached_result = await self._get_cached_analysis(content_hash, 'safety')
            if cached_result:
                return SafetyResult(**cached_result)
                
            categories = []
            reasons = []
            confidence_scores = []
            recommendations = []
            
            # Rule-based filtering
            rule_results = await self._rule_based_content_filter(content)
            if rule_results['violations']:
                categories.extend(rule_results['categories'])
                reasons.extend(rule_results['reasons'])
                confidence_scores.append(rule_results['confidence'])
                
            # ML-based classification
            if self.content_classifier:
                ml_results = await self._ml_content_classification(content)
                categories.extend(ml_results['categories'])
                reasons.extend(ml_results['reasons'])
                confidence_scores.append(ml_results['confidence'])
                
            # Toxicity detection
            if self.toxicity_detector:
                toxicity_results = await self._detect_toxicity(content)
                if toxicity_results['is_toxic']:
                    categories.append(ContentCategory.HARMFUL)
                    reasons.append(f"Toxic content detected: {toxicity_results['reason']}")
                    confidence_scores.append(toxicity_results['confidence'])
                    
            # Prompt injection detection
            injection_results = await self._detect_prompt_injection(content)
            if injection_results['is_injection']:
                categories.append(ContentCategory.HARMFUL)
                reasons.append("Potential prompt injection detected")
                confidence_scores.append(injection_results['confidence'])
                recommendations.append("Review for manipulation attempts")
                
            # Determine overall safety
            is_safe = len(categories) == 0 or all(cat == ContentCategory.SAFE for cat in categories)
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 1.0
            
            # Determine severity level
            severity = self._determine_severity_level(categories)
            
            # Create result
            result = SafetyResult(
                is_safe=is_safe,
                confidence=float(overall_confidence),
                categories=list(set(categories)),
                reasons=reasons,
                severity=severity,
                recommendations=recommendations
            )
            
            # Cache result
            await self._cache_analysis_result(content_hash, 'safety', result.__dict__)
            
            # Log analysis
            await self._log_safety_analysis(content_hash, result, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Content safety analysis error: {e}")
            return SafetyResult(
                is_safe=False,
                confidence=0.0,
                categories=[ContentCategory.HARMFUL],
                reasons=[f"Analysis error: {str(e)}"],
                severity=SafetyLevel.HIGH
            )
            
    async def analyze_bias(self, content: str, context: Dict[str, Any] = None) -> BiasAnalysisResult:
        """
        Analyze content for potential bias
        """
        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check cache
            cached_result = await self._get_cached_analysis(content_hash, 'bias')
            if cached_result:
                return BiasAnalysisResult(**cached_result)
                
            bias_types = []
            examples = []
            mitigation_suggestions = []
            confidence_scores = []
            
            # Gender bias detection
            gender_bias = await self._detect_gender_bias(content)
            if gender_bias['has_bias']:
                bias_types.append(BiasType.GENDER)
                examples.extend(gender_bias['examples'])
                confidence_scores.append(gender_bias['confidence'])
                mitigation_suggestions.append("Use gender-neutral language")
                
            # Racial bias detection
            racial_bias = await self._detect_racial_bias(content)
            if racial_bias['has_bias']:
                bias_types.append(BiasType.RACE)
                examples.extend(racial_bias['examples'])
                confidence_scores.append(racial_bias['confidence'])
                mitigation_suggestions.append("Review for racial stereotypes")
                
            # Age bias detection
            age_bias = await self._detect_age_bias(content)
            if age_bias['has_bias']:
                bias_types.append(BiasType.AGE)
                examples.extend(age_bias['examples'])
                confidence_scores.append(age_bias['confidence'])
                mitigation_suggestions.append("Avoid age-related assumptions")
                
            # ML-based bias detection
            if self.bias_detector:
                ml_bias = await self._ml_bias_detection(content)
                bias_types.extend(ml_bias['bias_types'])
                examples.extend(ml_bias['examples'])
                confidence_scores.append(ml_bias['confidence'])
                
            # Overall assessment
            has_bias = len(bias_types) > 0
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            result = BiasAnalysisResult(
                has_bias=has_bias,
                bias_types=list(set(bias_types)),
                confidence=float(overall_confidence),
                examples=examples,
                mitigation_suggestions=list(set(mitigation_suggestions))
            )
            
            # Cache result
            await self._cache_analysis_result(content_hash, 'bias', result.__dict__)
            
            return result
            
        except Exception as e:
            logger.error(f"Bias analysis error: {e}")
            return BiasAnalysisResult(
                has_bias=True,
                bias_types=[],
                confidence=0.0,
                examples=[],
                mitigation_suggestions=["Manual review recommended due to analysis error"]
            )
            
    async def detect_pii(self, content: str, anonymize: bool = True) -> PIIAnalysisResult:
        """
        Detect and optionally anonymize PII in content
        """
        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Use Presidio for PII detection
            analysis_results = self.pii_analyzer.analyze(
                text=content,
                language='en',
                entities=[
                    'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'SSN',
                    'CREDIT_CARD', 'IP_ADDRESS', 'DATE_TIME', 'LOCATION',
                    'ORGANIZATION', 'MEDICAL_LICENSE', 'US_DRIVER_LICENSE'
                ]
            )
            
            has_pii = len(analysis_results) > 0
            
            # Extract PII entities
            pii_entities = []
            for result in analysis_results:
                pii_entities.append({
                    'entity_type': result.entity_type,
                    'start': result.start,
                    'end': result.end,
                    'score': result.score,
                    'text': content[result.start:result.end]
                })
                
            # Anonymize if requested
            anonymized_text = content
            if anonymize and has_pii:
                anonymized_result = self.pii_anonymizer.anonymize(
                    text=content,
                    analyzer_results=analysis_results
                )
                anonymized_text = anonymized_result.text
                
            # Calculate confidence and risk level
            if pii_entities:
                avg_confidence = np.mean([entity['score'] for entity in pii_entities])
                risk_level = self._calculate_pii_risk_level(pii_entities)
            else:
                avg_confidence = 1.0
                risk_level = SafetyLevel.LOW
                
            result = PIIAnalysisResult(
                has_pii=has_pii,
                pii_entities=pii_entities,
                anonymized_text=anonymized_text,
                confidence=float(avg_confidence),
                risk_level=risk_level
            )
            
            # Log PII detection
            if has_pii:
                await self._log_pii_detection(content_hash, result)
                
            return result
            
        except Exception as e:
            logger.error(f"PII detection error: {e}")
            return PIIAnalysisResult(
                has_pii=True,  # Assume PII present on error for safety
                pii_entities=[],
                anonymized_text="[REDACTED DUE TO ANALYSIS ERROR]",
                confidence=0.0,
                risk_level=SafetyLevel.HIGH
            )
            
    async def check_compliance(self, content: str, standards: List[ComplianceStandard], context: Dict[str, Any] = None) -> ComplianceResult:
        """
        Check content compliance against specified standards
        """
        try:
            violations = []
            recommendations = []
            risk_scores = []
            
            for standard in standards:
                standard_violations, standard_recommendations, risk_score = await self._check_standard_compliance(
                    content, standard, context
                )
                
                violations.extend(standard_violations)
                recommendations.extend(standard_recommendations)
                risk_scores.append(risk_score)
                
            is_compliant = len(violations) == 0
            overall_risk_score = np.mean(risk_scores) if risk_scores else 0.0
            
            result = ComplianceResult(
                is_compliant=is_compliant,
                standards=standards,
                violations=violations,
                recommendations=list(set(recommendations)),
                risk_score=float(overall_risk_score)
            )
            
            # Log compliance check
            await self._log_compliance_check(result, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Compliance check error: {e}")
            return ComplianceResult(
                is_compliant=False,
                standards=standards,
                violations=[{'type': 'analysis_error', 'description': str(e)}],
                recommendations=["Manual compliance review required"],
                risk_score=1.0
            )
            
    async def create_audit_log(self, user_id: str, session_id: str, action: str, resource: str, result: str, metadata: Dict[str, Any] = None, ip_address: str = None, user_agent: str = None):
        """
        Create audit log entry
        """
        try:
            if not self.audit_enabled:
                return
                
            log_entry = AuditLogEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                session_id=session_id,
                action=action,
                resource=resource,
                result=result,
                metadata=metadata or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Encrypt sensitive data
            encrypted_data = self.cipher.encrypt(json.dumps({
                'metadata': metadata,
                'user_agent': user_agent
            }).encode())
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO audit_logs (
                        id, timestamp, user_id, session_id, action, resource,
                        result, metadata, ip_address, user_agent, encrypted_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ''', log_entry.id, log_entry.timestamp, log_entry.user_id,
                     log_entry.session_id, log_entry.action, log_entry.resource,
                     log_entry.result, json.dumps(log_entry.metadata),
                     log_entry.ip_address, log_entry.user_agent, encrypted_data)
                     
        except Exception as e:
            logger.error(f"Audit logging error: {e}")
            
    async def _rule_based_content_filter(self, content: str) -> Dict[str, Any]:
        """Rule-based content filtering"""
        violations = []
        categories = []
        reasons = []
        
        content_lower = content.lower()
        
        # Profanity check
        profanity_found = any(word in content_lower for word in self.content_filters['profanity_words'])
        if profanity_found:
            violations.append('profanity')
            categories.append(ContentCategory.INAPPROPRIATE)
            reasons.append("Profanity detected")
            
        # Hate speech patterns
        for pattern in self.content_filters['hate_speech_patterns']:
            if re.search(pattern, content_lower, re.IGNORECASE):
                violations.append('hate_speech')
                categories.append(ContentCategory.HATE_SPEECH)
                reasons.append("Hate speech pattern detected")
                break
                
        # Violence keywords
        violence_count = sum(1 for word in self.content_filters['violence_keywords'] if word in content_lower)
        if violence_count >= 2:  # Threshold for violence detection
            violations.append('violence')
            categories.append(ContentCategory.VIOLENCE)
            reasons.append(f"Violence-related content detected ({violence_count} indicators)")
            
        # Adult content patterns
        for pattern in self.content_filters['adult_content_patterns']:
            if re.search(pattern, content_lower, re.IGNORECASE):
                violations.append('adult_content')
                categories.append(ContentCategory.ADULT_CONTENT)
                reasons.append("Adult content detected")
                break
                
        # Spam indicators
        spam_score = sum(1 for pattern in self.content_filters['spam_indicators'] 
                        if re.search(pattern, content_lower, re.IGNORECASE))
        if spam_score >= 2:
            violations.append('spam')
            categories.append(ContentCategory.SPAM)
            reasons.append(f"Spam indicators detected (score: {spam_score})")
            
        confidence = 0.8 if violations else 0.9
        
        return {
            'violations': violations,
            'categories': categories,
            'reasons': reasons,
            'confidence': confidence
        }
        
    async def _ml_content_classification(self, content: str) -> Dict[str, Any]:
        """ML-based content classification"""
        try:
            # Use content classifier
            results = self.content_classifier(content)
            
            categories = []
            reasons = []
            confidence = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if score > 0.7:  # High confidence threshold
                    if 'toxic' in label or 'harmful' in label:
                        categories.append(ContentCategory.HARMFUL)
                        reasons.append(f"ML classifier: {label} (confidence: {score:.2f})")
                        confidence = max(confidence, score)
                    elif 'inappropriate' in label:
                        categories.append(ContentCategory.INAPPROPRIATE)
                        reasons.append(f"ML classifier: {label} (confidence: {score:.2f})")
                        confidence = max(confidence, score)
                        
            return {
                'categories': categories,
                'reasons': reasons,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"ML content classification error: {e}")
            return {'categories': [], 'reasons': [], 'confidence': 0.0}
            
    async def _detect_toxicity(self, content: str) -> Dict[str, Any]:
        """Detect toxic content using specialized model"""
        try:
            results = self.toxicity_detector(content)
            
            for result in results:
                if result['label'] == 'TOXIC' and result['score'] > 0.6:
                    return {
                        'is_toxic': True,
                        'confidence': result['score'],
                        'reason': f"Toxicity score: {result['score']:.2f}"
                    }
                    
            return {'is_toxic': False, 'confidence': 0.0, 'reason': 'No toxicity detected'}
            
        except Exception:
            return {'is_toxic': False, 'confidence': 0.0, 'reason': 'Analysis unavailable'}
            
    async def _detect_prompt_injection(self, content: str) -> Dict[str, Any]:
        """Detect prompt injection attempts"""
        try:
            # Check for common injection patterns
            injection_patterns = [
                r'ignore (previous|above) instructions',
                r'disregard (previous|above|prior) instructions',
                r'forget (everything|all previous)',
                r'new instructions:',
                r'system: ',
                r'override (security|safety)',
                r'jailbreak',
                r'act as.*different',
                r'pretend (you are|to be)',
                r'roleplay as'
            ]
            
            content_lower = content.lower()
            injection_score = 0
            
            for pattern in injection_patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    injection_score += 1
                    
            # Use ML model if available
            ml_score = 0.0
            if self.prompt_injection_detector:
                try:
                    results = self.prompt_injection_detector(content)
                    for result in results:
                        if result['label'] == 'INJECTION' and result['score'] > 0.7:
                            ml_score = result['score']
                            break
                except:
                    pass
                    
            # Combine rule-based and ML scores
            is_injection = injection_score >= 2 or ml_score > 0.7
            confidence = max(injection_score / 5.0, ml_score)
            
            return {
                'is_injection': is_injection,
                'confidence': min(confidence, 1.0)
            }
            
        except Exception:
            return {'is_injection': False, 'confidence': 0.0}
            
    async def _detect_gender_bias(self, content: str) -> Dict[str, Any]:
        """Detect gender bias in content"""
        try:
            bias_indicators = []
            
            # Gender stereotypes
            gender_stereotypes = [
                (r'\bmen are (better|stronger|smarter)', 'male superiority stereotype'),
                (r'\bwomen are (emotional|weak|irrational)', 'female stereotype'),
                (r'\bgirls (can\'t|cannot) (do|play|understand)', 'female limitation stereotype'),
                (r'\bboys (don\'t|do not) (cry|show emotion)', 'male stereotype'),
                (r'\b(he|his) (leads|manages|decides)', 'male leadership assumption'),
                (r'\b(she|her) (assists|supports|helps)', 'female support role assumption')
            ]
            
            for pattern, description in gender_stereotypes:
                if re.search(pattern, content, re.IGNORECASE):
                    bias_indicators.append(description)
                    
            # Pronoun analysis
            if self.nlp:
                doc = self.nlp(content)
                male_pronouns = sum(1 for token in doc if token.text.lower() in ['he', 'his', 'him'])
                female_pronouns = sum(1 for token in doc if token.text.lower() in ['she', 'her', 'hers'])
                
                total_pronouns = male_pronouns + female_pronouns
                if total_pronouns > 0:
                    ratio = abs(male_pronouns - female_pronouns) / total_pronouns
                    if ratio > 0.8:  # Significant imbalance
                        bias_indicators.append(f"Pronoun imbalance: {male_pronouns} male, {female_pronouns} female")
                        
            has_bias = len(bias_indicators) > 0
            confidence = min(len(bias_indicators) / 3.0, 1.0)
            
            return {
                'has_bias': has_bias,
                'confidence': confidence,
                'examples': bias_indicators
            }
            
        except Exception:
            return {'has_bias': False, 'confidence': 0.0, 'examples': []}
            
    async def _detect_racial_bias(self, content: str) -> Dict[str, Any]:
        """Detect racial bias in content"""
        try:
            bias_indicators = []
            
            # Racial stereotypes and slurs
            racial_bias_patterns = [
                (r'\b(all|most) (blacks|whites|asians|hispanics) (are|do)', 'racial generalization'),
                (r'\b(typical|stereotypical) (black|white|asian|hispanic)', 'racial stereotype'),
                (r'\b(race|racial) (superiority|inferiority)', 'racial hierarchy'),
                (r'\b(good|bad) at .* because .* (race|ethnicity)', 'racial attribution')
            ]
            
            for pattern, description in racial_bias_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    bias_indicators.append(description)
                    
            has_bias = len(bias_indicators) > 0
            confidence = min(len(bias_indicators) / 2.0, 1.0)
            
            return {
                'has_bias': has_bias,
                'confidence': confidence,
                'examples': bias_indicators
            }
            
        except Exception:
            return {'has_bias': False, 'confidence': 0.0, 'examples': []}
            
    async def _detect_age_bias(self, content: str) -> Dict[str, Any]:
        """Detect age bias in content"""
        try:
            bias_indicators = []
            
            # Age-related stereotypes
            age_bias_patterns = [
                (r'\b(old|elderly) people (can\'t|cannot|don\'t understand)', 'elderly stereotype'),
                (r'\b(young|millennials|gen z) (are|don\'t)', 'youth stereotype'),
                (r'\btoo (old|young) (for|to)', 'age discrimination'),
                (r'\b(past|beyond) (his|her|their) prime', 'age decline assumption')
            ]
            
            for pattern, description in age_bias_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    bias_indicators.append(description)
                    
            has_bias = len(bias_indicators) > 0
            confidence = min(len(bias_indicators) / 2.0, 1.0)
            
            return {
                'has_bias': has_bias,
                'confidence': confidence,
                'examples': bias_indicators
            }
            
        except Exception:
            return {'has_bias': False, 'confidence': 0.0, 'examples': []}
            
    async def _ml_bias_detection(self, content: str) -> Dict[str, Any]:
        """ML-based bias detection"""
        try:
            results = self.bias_detector(content)
            
            bias_types = []
            examples = []
            confidence = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if score > 0.6:  # Bias detection threshold
                    if 'gender' in label:
                        bias_types.append(BiasType.GENDER)
                    elif 'race' in label or 'racial' in label:
                        bias_types.append(BiasType.RACE)
                    elif 'age' in label:
                        bias_types.append(BiasType.AGE)
                    else:
                        bias_types.append(BiasType.CULTURAL)  # Generic bias type
                        
                    examples.append(f"ML detected: {label} (confidence: {score:.2f})")
                    confidence = max(confidence, score)
                    
            return {
                'bias_types': bias_types,
                'examples': examples,
                'confidence': confidence
            }
            
        except Exception:
            return {'bias_types': [], 'examples': [], 'confidence': 0.0}
            
    def _calculate_pii_risk_level(self, pii_entities: List[Dict[str, Any]]) -> SafetyLevel:
        """Calculate risk level based on PII entities"""
        high_risk_entities = {'SSN', 'CREDIT_CARD', 'MEDICAL_LICENSE', 'US_DRIVER_LICENSE'}
        medium_risk_entities = {'EMAIL_ADDRESS', 'PHONE_NUMBER', 'PERSON'}
        
        high_risk_count = sum(1 for entity in pii_entities if entity['entity_type'] in high_risk_entities)
        medium_risk_count = sum(1 for entity in pii_entities if entity['entity_type'] in medium_risk_entities)
        
        if high_risk_count > 0:
            return SafetyLevel.CRITICAL
        elif medium_risk_count > 2:
            return SafetyLevel.HIGH
        elif medium_risk_count > 0:
            return SafetyLevel.MEDIUM
        else:
            return SafetyLevel.LOW
            
    async def _check_standard_compliance(self, content: str, standard: ComplianceStandard, context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], float]:
        """Check compliance against a specific standard"""
        violations = []
        recommendations = []
        risk_score = 0.0
        
        rules = self.compliance_rules.get(standard, {})
        
        if standard == ComplianceStandard.GDPR:
            # GDPR compliance checks
            pii_result = await self.detect_pii(content, anonymize=False)
            
            if pii_result.has_pii and not context.get('consent_given', False):
                violations.append({
                    'type': 'gdpr_consent_required',
                    'description': 'PII processing without explicit consent',
                    'severity': 'high'
                })
                risk_score += 0.4
                
            if pii_result.has_pii:
                recommendations.append("Implement data minimization principles")
                recommendations.append("Ensure right to erasure capability")
                
        elif standard == ComplianceStandard.HIPAA:
            # HIPAA compliance checks
            phi_detected = await self._detect_phi(content)
            
            if phi_detected and not context.get('encrypted', False):
                violations.append({
                    'type': 'hipaa_encryption_required',
                    'description': 'PHI not encrypted',
                    'severity': 'critical'
                })
                risk_score += 0.6
                
        elif standard == ComplianceStandard.CCPA:
            # CCPA compliance checks
            personal_info = await self.detect_pii(content, anonymize=False)
            
            if personal_info.has_pii and not context.get('opt_out_provided', False):
                violations.append({
                    'type': 'ccpa_opt_out_required',
                    'description': 'Personal information processing without opt-out option',
                    'severity': 'medium'
                })
                risk_score += 0.3
                
        return violations, recommendations, risk_score
        
    async def _detect_phi(self, content: str) -> bool:
        """Detect Protected Health Information"""
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\bpatient (id|number|record)',
            r'\bmedical (record|history)',
            r'\bdiagnosis:',
            r'\bprescription:',
            r'\b(diabetes|hypertension|cancer|hiv|aids)\b'
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower, re.IGNORECASE) for pattern in phi_patterns)
        
    def _determine_severity_level(self, categories: List[ContentCategory]) -> SafetyLevel:
        """Determine severity level based on content categories"""
        if ContentCategory.ILLEGAL in categories:
            return SafetyLevel.CRITICAL
        elif ContentCategory.HARMFUL in categories or ContentCategory.HATE_SPEECH in categories:
            return SafetyLevel.HIGH
        elif ContentCategory.INAPPROPRIATE in categories or ContentCategory.VIOLENCE in categories:
            return SafetyLevel.MEDIUM
        elif ContentCategory.SPAM in categories:
            return SafetyLevel.LOW
        else:
            return SafetyLevel.LOW
            
    async def _get_cached_analysis(self, content_hash: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow('''
                    SELECT result FROM safety_analysis 
                    WHERE content_hash = $1 AND analysis_type = $2
                    AND timestamp > NOW() - INTERVAL '1 hour'
                    ORDER BY timestamp DESC LIMIT 1
                ''', content_hash, analysis_type)
                
                return dict(result['result']) if result else None
                
        except Exception:
            return None
            
    async def _cache_analysis_result(self, content_hash: str, analysis_type: str, result: Dict[str, Any]):
        """Cache analysis result"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO safety_analysis (id, content_hash, analysis_type, result, timestamp, model_version)
                    VALUES ($1, $2, $3, $4, $5, $6)
                ''', str(uuid.uuid4()), content_hash, analysis_type, json.dumps(result),
                     datetime.now(timezone.utc), "v1.0")
                     
        except Exception as e:
            logger.error(f"Cache error: {e}")
            
    async def _log_safety_analysis(self, content_hash: str, result: SafetyResult, context: Dict[str, Any]):
        """Log safety analysis result"""
        await self.create_audit_log(
            user_id=context.get('user_id', 'system'),
            session_id=context.get('session_id', 'system'),
            action='safety_analysis',
            resource=f'content:{content_hash[:16]}',
            result='completed',
            metadata={
                'is_safe': result.is_safe,
                'categories': [cat.value for cat in result.categories],
                'severity': result.severity.value,
                'confidence': result.confidence
            }
        )
        
    async def _log_pii_detection(self, content_hash: str, result: PIIAnalysisResult):
        """Log PII detection event"""
        await self.create_audit_log(
            user_id='system',
            session_id='system',
            action='pii_detection',
            resource=f'content:{content_hash[:16]}',
            result='pii_detected' if result.has_pii else 'no_pii',
            metadata={
                'entity_count': len(result.pii_entities),
                'entity_types': [entity['entity_type'] for entity in result.pii_entities],
                'risk_level': result.risk_level.value
            }
        )
        
    async def _log_compliance_check(self, result: ComplianceResult, context: Dict[str, Any]):
        """Log compliance check"""
        await self.create_audit_log(
            user_id=context.get('user_id', 'system'),
            session_id=context.get('session_id', 'system'),
            action='compliance_check',
            resource='content',
            result='compliant' if result.is_compliant else 'violations_found',
            metadata={
                'standards': [std.value for std in result.standards],
                'violation_count': len(result.violations),
                'risk_score': result.risk_score
            }
        )
        
    async def _audit_log_cleanup(self):
        """Background task to clean up old audit logs"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Delete logs older than retention period
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.audit_retention_days)
                
                async with self.db_pool.acquire() as conn:
                    deleted_count = await conn.fetchval('''
                        DELETE FROM audit_logs WHERE timestamp < $1
                    ''', cutoff_date)
                    
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old audit log entries")
                        
            except Exception as e:
                logger.error(f"Audit log cleanup error: {e}")
                
    async def _model_performance_monitoring(self):
        """Monitor ML model performance and retrain if needed"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                # This would implement model performance monitoring
                # and trigger retraining if performance degrades
                
                logger.debug("Model performance monitoring completed")
                
            except Exception as e:
                logger.error(f"Model monitoring error: {e}")
                
    async def generate_compliance_report(self, start_date: datetime, end_date: datetime, standards: List[ComplianceStandard]) -> Dict[str, Any]:
        """Generate compliance report for specified period"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get audit log statistics
                audit_stats = await conn.fetchrow('''
                    SELECT 
                        COUNT(*) as total_events,
                        COUNT(DISTINCT user_id) as unique_users,
                        COUNT(DISTINCT session_id) as unique_sessions
                    FROM audit_logs 
                    WHERE timestamp BETWEEN $1 AND $2
                ''', start_date, end_date)
                
                # Get compliance violations
                violations = await conn.fetch('''
                    SELECT violation_type, severity, COUNT(*) as count
                    FROM compliance_violations 
                    WHERE timestamp BETWEEN $1 AND $2
                    GROUP BY violation_type, severity
                ''', start_date, end_date)
                
                # Get safety analysis results
                safety_stats = await conn.fetchrow('''
                    SELECT 
                        COUNT(*) as total_analyses,
                        AVG((result->>'confidence')::float) as avg_confidence
                    FROM safety_analysis 
                    WHERE timestamp BETWEEN $1 AND $2 
                    AND analysis_type = 'safety'
                ''', start_date, end_date)
                
                report = {
                    'period': {
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    },
                    'standards': [std.value for std in standards],
                    'audit_statistics': dict(audit_stats) if audit_stats else {},
                    'violations': [dict(v) for v in violations],
                    'safety_statistics': dict(safety_stats) if safety_stats else {},
                    'generated_at': datetime.now(timezone.utc).isoformat()
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Compliance report generation error: {e}")
            return {'error': str(e)}
            
    async def shutdown(self):
        """Clean shutdown of the framework"""
        if self.db_pool:
            await self.db_pool.close()
            
        logger.info("AI Safety and Compliance Framework shut down")