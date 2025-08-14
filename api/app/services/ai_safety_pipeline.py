"""
AI Safety and Compliance Pipeline
Content filtering, PII detection, prompt injection prevention, and audit trails
"""

import asyncio
import json
import hashlib
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import aioredis
from cryptography.fernet import Fernet
import spacy
import nltk
from transformers import pipeline
from prometheus_client import Counter, Histogram, Gauge
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Metrics
safety_checks = Counter('safety_checks_total', 'Total safety checks', ['check_type', 'result'])
pii_detections = Counter('pii_detections_total', 'PII detections', ['pii_type'])
content_filtered = Counter('content_filtered_total', 'Content filtered', ['filter_type'])
compliance_violations = Counter('compliance_violations_total', 'Compliance violations', ['type'])
safety_latency = Histogram('safety_check_latency_seconds', 'Safety check latency', ['check_type'])


class SafetyCheckResult(Enum):
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"
    

class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "dob"
    MEDICAL_ID = "medical_id"
    

class ContentFilterType(Enum):
    TOXIC = "toxic"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    ILLEGAL = "illegal"
    PROMPT_INJECTION = "prompt_injection"
    

class ComplianceStandard(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    COPPA = "coppa"
    

@dataclass
class SafetyCheckReport:
    result: SafetyCheckResult
    filtered_content: Optional[str] = None
    detections: Dict[str, List[Any]] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    compliance_flags: Set[ComplianceStandard] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class AuditEntry:
    id: str
    timestamp: datetime
    user_id: str
    tenant_id: str
    conversation_id: str
    input_text: str
    output_text: Optional[str]
    safety_report: SafetyCheckReport
    model_used: str
    processing_time_ms: float
    encrypted: bool = False
    

class AISafetyPipeline:
    """Comprehensive AI safety and compliance pipeline"""
    
    def __init__(self, redis: aioredis.Redis, config: Dict[str, Any]):
        self.redis = redis
        self.config = config
        
        # Initialize NLP models
        self.nlp = spacy.load("en_core_web_sm")
        
        # Content filtering models (in production, use actual models)
        self.content_filters = {
            ContentFilterType.TOXIC: None,  # Initialize with actual toxicity model
            ContentFilterType.PROMPT_INJECTION: None  # Initialize with injection detector
        }
        
        # PII patterns
        self.pii_patterns = {
            PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE: re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            PIIType.SSN: re.compile(r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'),
            PIIType.CREDIT_CARD: re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b'),
            PIIType.IP_ADDRESS: re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
        }
        
        # Prompt injection patterns
        self.injection_patterns = [
            r'ignore (?:previous|all|above) (?:instructions|prompts|commands)',
            r'disregard (?:the|all|any) (?:rules|instructions|constraints)',
            r'pretend (?:you are|to be|that you)',
            r'act as (?:if|though) you (?:are|were)',
            r'bypass (?:the|any|all) (?:filters|safety|restrictions)',
            r'reveal (?:your|the) (?:instructions|prompt|system)',
            r'what (?:are|were) (?:your|the) (?:instructions|original prompt)',
            r'repeat (?:back|verbatim) (?:the|your) (?:prompt|instructions)',
            r'</?(script|img|iframe|object|embed|link)[^>]*>',
            r'javascript:|onerror=|onclick=|onload=',
        ]
        
        # Encryption for audit logs
        self.encryption_key = config.get("audit_encryption_key")
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)
        else:
            self.cipher = None
            
        # Compliance configurations
        self.retention_policies = {
            ComplianceStandard.GDPR: timedelta(days=30),
            ComplianceStandard.CCPA: timedelta(days=45),
            ComplianceStandard.HIPAA: timedelta(days=2190),  # 6 years
            ComplianceStandard.PCI_DSS: timedelta(days=365),
        }
        
    async def check_safety(
        self,
        input_text: str,
        user_id: str,
        tenant_id: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyCheckReport:
        """Perform comprehensive safety checks on input"""
        start_time = time.time()
        
        report = SafetyCheckReport(result=SafetyCheckResult.SAFE)
        
        # Run all checks in parallel
        check_tasks = [
            self._check_pii(input_text),
            self._check_content_filters(input_text),
            self._check_prompt_injection(input_text),
            self._check_compliance_requirements(input_text, context)
        ]
        
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process PII check
        pii_result = results[0]
        if not isinstance(pii_result, Exception):
            report.detections["pii"] = pii_result["detections"]
            if pii_result["found"]:
                report.result = SafetyCheckResult.WARNING
                
        # Process content filters
        content_result = results[1]
        if not isinstance(content_result, Exception):
            report.detections["content_filters"] = content_result["violations"]
            report.confidence_scores.update(content_result["scores"])
            if content_result["blocked"]:
                report.result = SafetyCheckResult.BLOCKED
                
        # Process prompt injection
        injection_result = results[2]
        if not isinstance(injection_result, Exception):
            if injection_result["detected"]:
                report.result = SafetyCheckResult.BLOCKED
                report.detections["prompt_injection"] = injection_result["patterns"]
                
        # Process compliance
        compliance_result = results[3]
        if not isinstance(compliance_result, Exception):
            report.compliance_flags = compliance_result["flags"]
            
        # Apply content sanitization if needed
        if report.result != SafetyCheckResult.BLOCKED:
            report.filtered_content = await self._sanitize_content(
                input_text,
                report.detections
            )
        else:
            report.filtered_content = None
            
        # Record metrics
        safety_checks.labels(
            check_type="comprehensive",
            result=report.result.value
        ).inc()
        
        safety_latency.labels(check_type="comprehensive").observe(
            time.time() - start_time
        )
        
        return report
        
    async def _check_pii(self, text: str) -> Dict[str, Any]:
        """Detect PII in text"""
        detections = []
        found = False
        
        # Pattern-based detection
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                found = True
                for match in matches:
                    detections.append({
                        "type": pii_type.value,
                        "value": match if isinstance(match, str) else match[0],
                        "confidence": 0.95
                    })
                    pii_detections.labels(pii_type=pii_type.value).inc()
                    
        # NLP-based detection for names and addresses
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                detections.append({
                    "type": PIIType.NAME.value,
                    "value": ent.text,
                    "confidence": 0.8
                })
                pii_detections.labels(pii_type=PIIType.NAME.value).inc()
                found = True
                
            elif ent.label_ in ["GPE", "LOC", "FAC"]:
                # Could be address component
                detections.append({
                    "type": PIIType.ADDRESS.value,
                    "value": ent.text,
                    "confidence": 0.6
                })
                
        return {"found": found, "detections": detections}
        
    async def _check_content_filters(self, text: str) -> Dict[str, Any]:
        """Check content against safety filters"""
        violations = []
        scores = {}
        blocked = False
        
        # Keyword-based filtering (simple example)
        toxic_keywords = [
            "hate", "kill", "violence", "illegal",
            # Add more comprehensive lists
        ]
        
        text_lower = text.lower()
        for keyword in toxic_keywords:
            if keyword in text_lower:
                violations.append({
                    "type": ContentFilterType.TOXIC.value,
                    "keyword": keyword,
                    "confidence": 0.7
                })
                scores[ContentFilterType.TOXIC.value] = 0.7
                
        # Model-based filtering
        # In production, use actual toxicity/safety models
        # Example: scores["toxicity"] = await self.toxicity_model.predict(text)
        
        # Determine if content should be blocked
        if any(score > 0.8 for score in scores.values()):
            blocked = True
            content_filtered.labels(filter_type="blocked").inc()
            
        return {
            "violations": violations,
            "scores": scores,
            "blocked": blocked
        }
        
    async def _check_prompt_injection(self, text: str) -> Dict[str, Any]:
        """Detect potential prompt injection attempts"""
        detected = False
        patterns = []
        
        text_lower = text.lower()
        
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected = True
                patterns.append(pattern)
                safety_checks.labels(
                    check_type="prompt_injection",
                    result="detected"
                ).inc()
                
        # Additional heuristics
        # Check for unusual control characters
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > 5:
            detected = True
            patterns.append("excessive_control_characters")
            
        # Check for script-like content
        if any(tag in text_lower for tag in ['<script', 'javascript:', 'eval(']):
            detected = True
            patterns.append("script_content")
            
        return {"detected": detected, "patterns": patterns}
        
    async def _check_compliance_requirements(
        self,
        text: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check compliance requirements based on context"""
        flags = set()
        
        if not context:
            return {"flags": flags}
            
        # GDPR compliance
        if context.get("region") == "EU":
            flags.add(ComplianceStandard.GDPR)
            
        # HIPAA compliance
        if context.get("industry") == "healthcare":
            flags.add(ComplianceStandard.HIPAA)
            
        # COPPA compliance
        if context.get("user_age") and context["user_age"] < 13:
            flags.add(ComplianceStandard.COPPA)
            compliance_violations.labels(type="coppa_age").inc()
            
        # PCI DSS compliance
        credit_card_pattern = self.pii_patterns[PIIType.CREDIT_CARD]
        if credit_card_pattern.search(text):
            flags.add(ComplianceStandard.PCI_DSS)
            
        return {"flags": flags}
        
    async def _sanitize_content(
        self,
        text: str,
        detections: Dict[str, List[Any]]
    ) -> str:
        """Sanitize content based on detections"""
        sanitized = text
        
        # Mask PII
        for pii in detections.get("pii", []):
            if pii["type"] == PIIType.EMAIL.value:
                sanitized = sanitized.replace(
                    pii["value"],
                    "***@***.***"
                )
            elif pii["type"] == PIIType.PHONE.value:
                sanitized = sanitized.replace(
                    pii["value"],
                    "***-***-****"
                )
            elif pii["type"] == PIIType.SSN.value:
                sanitized = sanitized.replace(
                    pii["value"],
                    "***-**-****"
                )
            elif pii["type"] == PIIType.CREDIT_CARD.value:
                # Keep last 4 digits
                masked = "*" * (len(pii["value"]) - 4) + pii["value"][-4:]
                sanitized = sanitized.replace(pii["value"], masked)
                
        return sanitized
        
    async def validate_output(
        self,
        output_text: str,
        safety_report: SafetyCheckReport,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyCheckReport:
        """Validate AI output for safety"""
        # Reuse input validation logic
        output_report = await self.check_safety(
            output_text,
            context.get("user_id", "system"),
            context.get("tenant_id", "system"),
            context.get("conversation_id", "system"),
            context
        )
        
        # Additional output-specific checks
        # Check for hallucinated PII
        if output_report.detections.get("pii") and not safety_report.detections.get("pii"):
            # Model generated PII that wasn't in input
            output_report.result = SafetyCheckResult.BLOCKED
            output_report.metadata["hallucinated_pii"] = True
            
        return output_report
        
    # Audit Trail Management
    
    async def create_audit_entry(
        self,
        user_id: str,
        tenant_id: str,
        conversation_id: str,
        input_text: str,
        output_text: Optional[str],
        safety_report: SafetyCheckReport,
        model_used: str,
        processing_time_ms: float
    ) -> str:
        """Create encrypted audit trail entry"""
        entry = AuditEntry(
            id=f"audit_{int(time.time() * 1000000)}_{user_id[:8]}",
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            input_text=input_text,
            output_text=output_text,
            safety_report=safety_report,
            model_used=model_used,
            processing_time_ms=processing_time_ms
        )
        
        # Serialize entry
        entry_data = {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
            "user_id": entry.user_id,
            "tenant_id": entry.tenant_id,
            "conversation_id": entry.conversation_id,
            "input_text": entry.input_text,
            "output_text": entry.output_text,
            "safety_report": {
                "result": entry.safety_report.result.value,
                "detections": entry.safety_report.detections,
                "confidence_scores": entry.safety_report.confidence_scores,
                "compliance_flags": [f.value for f in entry.safety_report.compliance_flags]
            },
            "model_used": entry.model_used,
            "processing_time_ms": entry.processing_time_ms
        }
        
        # Encrypt if configured
        if self.cipher:
            encrypted_data = self.cipher.encrypt(
                json.dumps(entry_data).encode()
            )
            entry.encrypted = True
            
            await self.redis.setex(
                f"audit:{entry.id}",
                int(self._get_retention_period(entry.safety_report.compliance_flags).total_seconds()),
                encrypted_data
            )
        else:
            await self.redis.setex(
                f"audit:{entry.id}",
                int(self._get_retention_period(entry.safety_report.compliance_flags).total_seconds()),
                json.dumps(entry_data)
            )
            
        # Index by various keys for retrieval
        await self._index_audit_entry(entry)
        
        return entry.id
        
    async def _index_audit_entry(self, entry: AuditEntry):
        """Index audit entry for efficient retrieval"""
        # Index by user
        await self.redis.zadd(
            f"audit:user:{entry.user_id}",
            {entry.id: entry.timestamp.timestamp()}
        )
        
        # Index by tenant
        await self.redis.zadd(
            f"audit:tenant:{entry.tenant_id}",
            {entry.id: entry.timestamp.timestamp()}
        )
        
        # Index by conversation
        await self.redis.zadd(
            f"audit:conversation:{entry.conversation_id}",
            {entry.id: entry.timestamp.timestamp()}
        )
        
        # Index by compliance standards
        for standard in entry.safety_report.compliance_flags:
            await self.redis.zadd(
                f"audit:compliance:{standard.value}",
                {entry.id: entry.timestamp.timestamp()}
            )
            
    def _get_retention_period(self, compliance_flags: Set[ComplianceStandard]) -> timedelta:
        """Get maximum retention period based on compliance requirements"""
        if not compliance_flags:
            return timedelta(days=90)  # Default retention
            
        # Return the longest retention period required
        periods = [
            self.retention_policies.get(flag, timedelta(days=90))
            for flag in compliance_flags
        ]
        
        return max(periods)
        
    async def get_audit_entries(
        self,
        filters: Dict[str, Any],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve audit entries with filters"""
        # Determine index to use
        if "user_id" in filters:
            index_key = f"audit:user:{filters['user_id']}"
        elif "tenant_id" in filters:
            index_key = f"audit:tenant:{filters['tenant_id']}"
        elif "conversation_id" in filters:
            index_key = f"audit:conversation:{filters['conversation_id']}"
        else:
            raise ValueError("At least one filter required")
            
        # Get entry IDs within time range
        min_score = start_time.timestamp() if start_time else 0
        max_score = end_time.timestamp() if end_time else float('inf')
        
        entry_ids = await self.redis.zrangebyscore(
            index_key,
            min_score,
            max_score,
            start=0,
            num=limit
        )
        
        # Retrieve entries
        entries = []
        for entry_id in entry_ids:
            entry_data = await self.redis.get(f"audit:{entry_id}")
            if entry_data:
                if self.cipher and isinstance(entry_data, bytes):
                    # Decrypt
                    decrypted = self.cipher.decrypt(entry_data)
                    entry = json.loads(decrypted)
                else:
                    entry = json.loads(entry_data)
                    
                entries.append(entry)
                
        return entries
        
    async def export_audit_trail(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> bytes:
        """Export audit trail for compliance reporting"""
        entries = await self.get_audit_entries(
            {"tenant_id": tenant_id},
            start_time=start_date,
            end_time=end_date,
            limit=10000  # Adjust based on requirements
        )
        
        if format == "json":
            return json.dumps(entries, indent=2).encode()
        elif format == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "id", "timestamp", "user_id", "conversation_id",
                    "safety_result", "model_used", "processing_time_ms"
                ]
            )
            
            writer.writeheader()
            for entry in entries:
                writer.writerow({
                    "id": entry["id"],
                    "timestamp": entry["timestamp"],
                    "user_id": entry["user_id"],
                    "conversation_id": entry["conversation_id"],
                    "safety_result": entry["safety_report"]["result"],
                    "model_used": entry["model_used"],
                    "processing_time_ms": entry["processing_time_ms"]
                })
                
            return output.getvalue().encode()
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    # Data Retention and GDPR Compliance
    
    async def delete_user_data(self, user_id: str):
        """Delete user data for GDPR compliance"""
        # Get all audit entries for user
        entry_ids = await self.redis.zrange(
            f"audit:user:{user_id}",
            0,
            -1
        )
        
        # Delete entries
        for entry_id in entry_ids:
            await self.redis.delete(f"audit:{entry_id}")
            
        # Delete indexes
        await self.redis.delete(f"audit:user:{user_id}")
        
        # Log deletion for compliance
        logger.info(f"Deleted all data for user {user_id}")
        
    async def anonymize_old_data(self):
        """Anonymize data past retention period"""
        # This would run as a scheduled job
        for standard, retention_period in self.retention_policies.items():
            cutoff_time = datetime.now(timezone.utc) - retention_period
            
            # Get entries past retention
            old_entries = await self.redis.zrangebyscore(
                f"audit:compliance:{standard.value}",
                0,
                cutoff_time.timestamp()
            )
            
            for entry_id in old_entries:
                entry_data = await self.redis.get(f"audit:{entry_id}")
                if entry_data:
                    # Anonymize instead of delete
                    if self.cipher:
                        entry = json.loads(self.cipher.decrypt(entry_data))
                    else:
                        entry = json.loads(entry_data)
                        
                    # Remove PII
                    entry["user_id"] = hashlib.sha256(entry["user_id"].encode()).hexdigest()[:16]
                    entry["input_text"] = "[REDACTED]"
                    entry["output_text"] = "[REDACTED]"
                    
                    # Re-save
                    if self.cipher:
                        encrypted = self.cipher.encrypt(json.dumps(entry).encode())
                        await self.redis.set(f"audit:{entry_id}", encrypted)
                    else:
                        await self.redis.set(f"audit:{entry_id}", json.dumps(entry))
                        
    # Bias Detection
    
    async def check_bias(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check for potential bias in text"""
        bias_indicators = {
            "gender": [
                r'\b(he|him|his|man|men|boy|male)\b',
                r'\b(she|her|hers|woman|women|girl|female)\b'
            ],
            "age": [
                r'\b(young|old|elderly|millennial|boomer|gen[- ]?z)\b'
            ],
            "race": [
                # Be very careful with these patterns
                # In production, use proper bias detection models
            ]
        }
        
        bias_scores = {}
        
        for bias_type, patterns in bias_indicators.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches)
                
            bias_scores[bias_type] = min(score / len(text.split()), 1.0)
            
        return {
            "bias_scores": bias_scores,
            "high_bias": any(score > 0.3 for score in bias_scores.values())
        }