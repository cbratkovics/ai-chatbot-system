#!/usr/bin/env python3
"""
Enterprise FinOps Cost Management Platform
Cloud cost allocation, AI API optimization, and financial operations automation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import boto3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis
from prometheus_client import Counter, Histogram, Gauge
import aiohttp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
cost_allocation_operations = Counter('finops_cost_allocation_total', 'Total cost allocation operations', ['service', 'status'])
api_usage_optimizations = Counter('finops_api_optimizations_total', 'API usage optimizations performed', ['provider', 'model'])
anomaly_detections = Counter('finops_anomaly_detections_total', 'Cost anomalies detected', ['severity', 'service'])
savings_identified = Gauge('finops_savings_identified_dollars', 'Total savings identified in dollars', ['category'])
cost_variance = Histogram('finops_cost_variance_percentage', 'Cost variance from budget', ['service'])

class CostCategory(Enum):
    """Cost categories for classification"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    AI_API = "ai_api"
    DATABASE = "database"
    MONITORING = "monitoring"
    SECURITY = "security"
    OTHER = "other"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CostRecord:
    """Individual cost record"""
    service_name: str
    resource_id: str
    cost_category: CostCategory
    amount: float
    currency: str
    usage_quantity: float
    usage_unit: str
    date: datetime
    tags: Dict[str, str]
    region: str
    account_id: str

@dataclass
class CostAnomaly:
    """Cost anomaly detection result"""
    service_name: str
    anomaly_type: str
    severity: AlertSeverity
    current_cost: float
    expected_cost: float
    variance_percentage: float
    detection_date: datetime
    description: str
    recommendation: str

@dataclass
class SavingsRecommendation:
    """Cost savings recommendation"""
    category: str
    service_name: str
    recommendation_type: str
    description: str
    estimated_monthly_savings: float
    implementation_effort: str
    confidence_score: float
    created_date: datetime

@dataclass
class APIUsageMetrics:
    """AI API usage metrics"""
    provider: str
    model: str
    total_requests: int
    total_tokens: int
    total_cost: float
    average_cost_per_request: float
    average_tokens_per_request: float
    date: datetime

class CloudCostAnalyzer:
    """Cloud cost analysis and allocation"""
    
    def __init__(self, aws_session: boto3.Session):
        self.aws_session = aws_session
        self.cost_explorer = aws_session.client('ce')
        self.ec2 = aws_session.client('ec2')
        self.rds = aws_session.client('rds')
        self.s3 = aws_session.client('s3')
        
    async def get_service_costs(self, start_date: datetime, end_date: datetime) -> List[CostRecord]:
        """Get detailed cost breakdown by service"""
        logger.info(f"Retrieving service costs from {start_date} to {end_date}")
        
        try:
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'},
                    {'Type': 'TAG', 'Key': 'Environment'},
                    {'Type': 'TAG', 'Key': 'Team'},
                    {'Type': 'TAG', 'Key': 'Project'}
                ]
            )
            
            cost_records = []
            for result in response['ResultsByTime']:
                date = datetime.strptime(result['TimePeriod']['Start'], '%Y-%m-%d')
                
                for group in result['Groups']:
                    service_name = group['Keys'][0] if group['Keys'][0] != 'NoService' else 'Unallocated'
                    region = group['Keys'][1] if len(group['Keys']) > 1 else 'us-east-1'
                    
                    # Extract tags
                    tags = {}
                    if len(group['Keys']) > 2:
                        for i, key in enumerate(group['Keys'][2:], 2):
                            tag_name = ['Environment', 'Team', 'Project'][i-2] if i-2 < 3 else f'Tag{i-2}'
                            if key and key != 'NoTagValue':
                                tags[tag_name] = key
                    
                    # Get cost and usage metrics
                    cost_amount = float(group['Metrics']['BlendedCost']['Amount'])
                    usage_quantity = float(group['Metrics']['UsageQuantity']['Amount'])
                    usage_unit = group['Metrics']['UsageQuantity']['Unit']
                    
                    if cost_amount > 0:  # Only include records with actual costs
                        cost_record = CostRecord(
                            service_name=service_name,
                            resource_id=f"{service_name}-{region}-{date.strftime('%Y%m%d')}",
                            cost_category=self._categorize_service(service_name),
                            amount=cost_amount,
                            currency='USD',
                            usage_quantity=usage_quantity,
                            usage_unit=usage_unit,
                            date=date,
                            tags=tags,
                            region=region,
                            account_id=self.aws_session.get_credentials().access_key[:10]
                        )
                        cost_records.append(cost_record)
            
            logger.info(f"Retrieved {len(cost_records)} cost records")
            return cost_records
            
        except Exception as e:
            logger.error(f"Error retrieving service costs: {e}")
            return []
    
    def _categorize_service(self, service_name: str) -> CostCategory:
        """Categorize AWS service into cost category"""
        service_mapping = {
            'Amazon Elastic Compute Cloud': CostCategory.COMPUTE,
            'Amazon EC2': CostCategory.COMPUTE,
            'AWS Lambda': CostCategory.COMPUTE,
            'Amazon ECS': CostCategory.COMPUTE,
            'Amazon EKS': CostCategory.COMPUTE,
            'Amazon S3': CostCategory.STORAGE,
            'Amazon EBS': CostCategory.STORAGE,
            'Amazon EFS': CostCategory.STORAGE,
            'Amazon RDS': CostCategory.DATABASE,
            'Amazon DynamoDB': CostCategory.DATABASE,
            'Amazon ElastiCache': CostCategory.DATABASE,
            'Amazon CloudFront': CostCategory.NETWORK,
            'Amazon VPC': CostCategory.NETWORK,
            'AWS Data Transfer': CostCategory.NETWORK,
            'Amazon CloudWatch': CostCategory.MONITORING,
            'AWS X-Ray': CostCategory.MONITORING,
            'Amazon Bedrock': CostCategory.AI_API,
            'Amazon SageMaker': CostCategory.AI_API,
            'AWS Security Hub': CostCategory.SECURITY,
            'Amazon GuardDuty': CostCategory.SECURITY,
        }
        
        for service_key, category in service_mapping.items():
            if service_key.lower() in service_name.lower():
                return category
                
        return CostCategory.OTHER
    
    async def allocate_shared_costs(self, cost_records: List[CostRecord]) -> List[CostRecord]:
        """Allocate shared costs based on usage patterns"""
        logger.info("Allocating shared costs to services")
        
        # Group records by service and calculate total usage
        service_usage = {}
        shared_costs = []
        direct_costs = []
        
        for record in cost_records:
            if record.tags.get('Team') == 'Shared' or not record.tags.get('Team'):
                shared_costs.append(record)
            else:
                direct_costs.append(record)
                
            service_key = record.tags.get('Team', 'Unknown')
            if service_key not in service_usage:
                service_usage[service_key] = {'total_cost': 0, 'compute_usage': 0}
            service_usage[service_key]['total_cost'] += record.amount
            
            if record.cost_category == CostCategory.COMPUTE:
                service_usage[service_key]['compute_usage'] += record.usage_quantity
        
        # Calculate allocation ratios based on compute usage
        total_compute_usage = sum(metrics['compute_usage'] for metrics in service_usage.values())
        
        allocated_records = direct_costs.copy()
        
        for shared_record in shared_costs:
            for service, metrics in service_usage.items():
                if service != 'Unknown' and total_compute_usage > 0:
                    allocation_ratio = metrics['compute_usage'] / total_compute_usage
                    allocated_amount = shared_record.amount * allocation_ratio
                    
                    if allocated_amount > 0.01:  # Only allocate if significant
                        allocated_record = CostRecord(
                            service_name=f"{shared_record.service_name} (Allocated to {service})",
                            resource_id=f"{shared_record.resource_id}-allocated-{service}",
                            cost_category=shared_record.cost_category,
                            amount=allocated_amount,
                            currency=shared_record.currency,
                            usage_quantity=shared_record.usage_quantity * allocation_ratio,
                            usage_unit=shared_record.usage_unit,
                            date=shared_record.date,
                            tags={**shared_record.tags, 'AllocatedTo': service},
                            region=shared_record.region,
                            account_id=shared_record.account_id
                        )
                        allocated_records.append(allocated_record)
        
        logger.info(f"Allocated {len(shared_costs)} shared cost records to services")
        return allocated_records

class AIAPIOptimizer:
    """AI API usage optimization and cost management"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.api_providers = {
            'openai': {'base_url': 'https://api.openai.com/v1', 'models': ['gpt-4', 'gpt-3.5-turbo']},
            'anthropic': {'base_url': 'https://api.anthropic.com/v1', 'models': ['claude-3-opus', 'claude-3-sonnet']},
            'bedrock': {'base_url': 'https://bedrock-runtime.amazonaws.com', 'models': ['claude-v2', 'titan-text']}
        }
        
    async def analyze_api_usage(self, start_date: datetime, end_date: datetime) -> List[APIUsageMetrics]:
        """Analyze AI API usage patterns and costs"""
        logger.info(f"Analyzing AI API usage from {start_date} to {end_date}")
        
        usage_metrics = []
        
        for provider, config in self.api_providers.items():
            for model in config['models']:
                # Get usage data from Redis cache or API logs
                usage_key = f"api_usage:{provider}:{model}:{start_date.strftime('%Y%m%d')}"
                cached_data = self.redis_client.get(usage_key)
                
                if cached_data:
                    data = json.loads(cached_data)
                    metrics = APIUsageMetrics(
                        provider=provider,
                        model=model,
                        total_requests=data['total_requests'],
                        total_tokens=data['total_tokens'],
                        total_cost=data['total_cost'],
                        average_cost_per_request=data['total_cost'] / max(data['total_requests'], 1),
                        average_tokens_per_request=data['total_tokens'] / max(data['total_requests'], 1),
                        date=start_date
                    )
                    usage_metrics.append(metrics)
        
        return usage_metrics
    
    async def optimize_model_selection(self, usage_metrics: List[APIUsageMetrics]) -> List[SavingsRecommendation]:
        """Generate recommendations for optimal model selection"""
        logger.info("Generating AI model optimization recommendations")
        
        recommendations = []
        
        # Analyze cost per token across models
        cost_analysis = {}
        for metric in usage_metrics:
            provider_model = f"{metric.provider}:{metric.model}"
            if metric.total_tokens > 0:
                cost_per_token = metric.total_cost / metric.total_tokens
                cost_analysis[provider_model] = {
                    'cost_per_token': cost_per_token,
                    'total_cost': metric.total_cost,
                    'usage_volume': metric.total_requests
                }
        
        # Find optimization opportunities
        sorted_models = sorted(cost_analysis.items(), key=lambda x: x[1]['cost_per_token'])
        
        if len(sorted_models) >= 2:
            cheapest_model = sorted_models[0]
            expensive_models = sorted_models[1:]
            
            for model_name, metrics in expensive_models:
                if metrics['usage_volume'] > 1000:  # Only recommend for high-volume usage
                    potential_savings = (metrics['cost_per_token'] - cheapest_model[1]['cost_per_token']) * \
                                      (metrics['total_cost'] / metrics['cost_per_token'])  # Approximate tokens
                    
                    if potential_savings > 100:  # Minimum $100/month savings
                        recommendation = SavingsRecommendation(
                            category="AI API Optimization",
                            service_name=model_name.split(':')[0],
                            recommendation_type="Model Substitution",
                            description=f"Consider migrating from {model_name} to {cheapest_model[0]} for cost optimization. "
                                      f"Potential savings: ${potential_savings:.2f}/month",
                            estimated_monthly_savings=potential_savings,
                            implementation_effort="Medium",
                            confidence_score=0.8 if potential_savings > 500 else 0.6,
                            created_date=datetime.now()
                        )
                        recommendations.append(recommendation)
        
        # Cache optimization recommendations
        for recommendation in recommendations:
            api_usage_optimizations.labels(
                provider=recommendation.service_name,
                model=recommendation.recommendation_type
            ).inc()
        
        return recommendations
    
    async def implement_request_caching(self, cache_duration_hours: int = 24) -> Dict[str, Any]:
        """Implement intelligent request caching to reduce API costs"""
        logger.info(f"Implementing request caching with {cache_duration_hours}h duration")
        
        cache_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'estimated_savings': 0.0
        }
        
        # Implement semantic caching for similar requests
        async def cache_similar_requests():
            # This would integrate with your application's request handling
            # to cache semantically similar requests
            pass
        
        return cache_stats

class CostAnomalyDetector:
    """Machine learning-based cost anomaly detection"""
    
    def __init__(self):
        self.anomaly_threshold = 2.0  # Standard deviations
        self.min_data_points = 7  # Minimum days of data needed
        
    async def detect_anomalies(self, cost_records: List[CostRecord]) -> List[CostAnomaly]:
        """Detect cost anomalies using statistical analysis"""
        logger.info("Detecting cost anomalies")
        
        anomalies = []
        
        # Group costs by service and date
        df = pd.DataFrame([asdict(record) for record in cost_records])
        if df.empty:
            return anomalies
        
        df['date'] = pd.to_datetime(df['date'])
        daily_costs = df.groupby(['service_name', 'date'])['amount'].sum().reset_index()
        
        for service_name in daily_costs['service_name'].unique():
            service_data = daily_costs[daily_costs['service_name'] == service_name].copy()
            service_data = service_data.sort_values('date')
            
            if len(service_data) < self.min_data_points:
                continue
                
            # Calculate rolling statistics
            service_data['rolling_mean'] = service_data['amount'].rolling(window=7, min_periods=3).mean()
            service_data['rolling_std'] = service_data['amount'].rolling(window=7, min_periods=3).std()
            
            # Detect anomalies
            for _, row in service_data.iterrows():
                if pd.isna(row['rolling_mean']) or pd.isna(row['rolling_std']):
                    continue
                    
                if row['rolling_std'] > 0:
                    z_score = abs(row['amount'] - row['rolling_mean']) / row['rolling_std']
                    
                    if z_score > self.anomaly_threshold:
                        variance_pct = ((row['amount'] - row['rolling_mean']) / row['rolling_mean']) * 100
                        
                        severity = self._determine_severity(abs(variance_pct), row['amount'])
                        
                        anomaly = CostAnomaly(
                            service_name=service_name,
                            anomaly_type="Statistical Outlier",
                            severity=severity,
                            current_cost=row['amount'],
                            expected_cost=row['rolling_mean'],
                            variance_percentage=variance_pct,
                            detection_date=row['date'],
                            description=f"Cost spike detected: {variance_pct:.1f}% variance from expected",
                            recommendation=self._generate_anomaly_recommendation(service_name, variance_pct)
                        )
                        anomalies.append(anomaly)
                        
                        # Update Prometheus metrics
                        anomaly_detections.labels(
                            severity=severity.value,
                            service=service_name
                        ).inc()
        
        logger.info(f"Detected {len(anomalies)} cost anomalies")
        return anomalies
    
    def _determine_severity(self, variance_pct: float, cost_amount: float) -> AlertSeverity:
        """Determine anomaly severity based on variance and cost"""
        if variance_pct > 100 and cost_amount > 1000:
            return AlertSeverity.CRITICAL
        elif variance_pct > 50 and cost_amount > 500:
            return AlertSeverity.HIGH
        elif variance_pct > 25:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _generate_anomaly_recommendation(self, service_name: str, variance_pct: float) -> str:
        """Generate recommendation based on anomaly type"""
        if variance_pct > 0:
            return f"Investigate {service_name} for unexpected usage increase. " \
                   "Check for resource scaling, configuration changes, or security incidents."
        else:
            return f"Verify {service_name} functionality after significant cost decrease. " \
                   "May indicate service disruption or resource underutilization."

class ResourceTagger:
    """Automated resource tagging for cost allocation"""
    
    def __init__(self, aws_session: boto3.Session):
        self.aws_session = aws_session
        self.ec2 = aws_session.client('ec2')
        self.rds = aws_session.client('rds')
        self.s3 = aws_session.client('s3')
        
    async def auto_tag_resources(self) -> Dict[str, int]:
        """Automatically tag resources based on naming conventions and usage patterns"""
        logger.info("Starting automated resource tagging")
        
        tagging_results = {
            'ec2_instances': 0,
            'rds_instances': 0,
            's3_buckets': 0,
            'errors': 0
        }
        
        try:
            # Tag EC2 instances
            ec2_tagged = await self._tag_ec2_instances()
            tagging_results['ec2_instances'] = ec2_tagged
            
            # Tag RDS instances
            rds_tagged = await self._tag_rds_instances()
            tagging_results['rds_instances'] = rds_tagged
            
            # Tag S3 buckets
            s3_tagged = await self._tag_s3_buckets()
            tagging_results['s3_buckets'] = s3_tagged
            
        except Exception as e:
            logger.error(f"Error in automated tagging: {e}")
            tagging_results['errors'] += 1
        
        return tagging_results
    
    async def _tag_ec2_instances(self) -> int:
        """Tag EC2 instances based on naming and usage patterns"""
        try:
            response = self.ec2.describe_instances()
            tagged_count = 0
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_id = instance['InstanceId']
                    existing_tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    
                    new_tags = []
                    
                    # Infer environment from instance name or VPC
                    if 'Name' in existing_tags:
                        name = existing_tags['Name'].lower()
                        if 'prod' in name or 'production' in name:
                            new_tags.append({'Key': 'Environment', 'Value': 'production'})
                        elif 'dev' in name or 'development' in name:
                            new_tags.append({'Key': 'Environment', 'Value': 'development'})
                        elif 'staging' in name or 'stage' in name:
                            new_tags.append({'Key': 'Environment', 'Value': 'staging'})
                    
                    # Infer team from instance name patterns
                    if 'Name' in existing_tags:
                        name = existing_tags['Name'].lower()
                        if 'api' in name or 'backend' in name:
                            new_tags.append({'Key': 'Team', 'Value': 'backend'})
                        elif 'frontend' in name or 'web' in name:
                            new_tags.append({'Key': 'Team', 'Value': 'frontend'})
                        elif 'data' in name or 'analytics' in name:
                            new_tags.append({'Key': 'Team', 'Value': 'data'})
                    
                    # Add cost center based on instance type
                    instance_type = instance['InstanceType']
                    if instance_type.startswith('t'):
                        new_tags.append({'Key': 'CostCenter', 'Value': 'development'})
                    elif instance_type.startswith(('m', 'c', 'r')):
                        new_tags.append({'Key': 'CostCenter', 'Value': 'production'})
                    
                    # Add project tag if missing
                    if 'Project' not in existing_tags:
                        new_tags.append({'Key': 'Project', 'Value': 'ai-chatbot-system'})
                    
                    # Apply new tags
                    if new_tags:
                        self.ec2.create_tags(Resources=[instance_id], Tags=new_tags)
                        tagged_count += 1
            
            return tagged_count
            
        except Exception as e:
            logger.error(f"Error tagging EC2 instances: {e}")
            return 0
    
    async def _tag_rds_instances(self) -> int:
        """Tag RDS instances based on naming patterns"""
        try:
            response = self.rds.describe_db_instances()
            tagged_count = 0
            
            for db_instance in response['DBInstances']:
                db_identifier = db_instance['DBInstanceIdentifier']
                
                # Get existing tags
                tags_response = self.rds.list_tags_for_resource(
                    ResourceName=db_instance['DBInstanceArn']
                )
                existing_tags = {tag['Key']: tag['Value'] for tag in tags_response['TagList']}
                
                new_tags = []
                
                # Infer environment from DB identifier
                identifier_lower = db_identifier.lower()
                if 'prod' in identifier_lower:
                    new_tags.append({'Key': 'Environment', 'Value': 'production'})
                elif 'dev' in identifier_lower:
                    new_tags.append({'Key': 'Environment', 'Value': 'development'})
                
                # Add database team tag
                if 'Team' not in existing_tags:
                    new_tags.append({'Key': 'Team', 'Value': 'backend'})
                
                # Add project tag
                if 'Project' not in existing_tags:
                    new_tags.append({'Key': 'Project', 'Value': 'ai-chatbot-system'})
                
                # Apply new tags
                if new_tags:
                    self.rds.add_tags_to_resource(
                        ResourceName=db_instance['DBInstanceArn'],
                        Tags=new_tags
                    )
                    tagged_count += 1
            
            return tagged_count
            
        except Exception as e:
            logger.error(f"Error tagging RDS instances: {e}")
            return 0
    
    async def _tag_s3_buckets(self) -> int:
        """Tag S3 buckets based on naming patterns"""
        try:
            response = self.s3.list_buckets()
            tagged_count = 0
            
            for bucket in response['Buckets']:
                bucket_name = bucket['Name']
                
                try:
                    # Get existing tags
                    existing_tags = {}
                    try:
                        tags_response = self.s3.get_bucket_tagging(Bucket=bucket_name)
                        existing_tags = {tag['Key']: tag['Value'] for tag in tags_response['TagSet']}
                    except self.s3.exceptions.ClientError:
                        pass  # No existing tags
                    
                    new_tags = []
                    
                    # Infer purpose from bucket name
                    bucket_lower = bucket_name.lower()
                    if 'backup' in bucket_lower:
                        new_tags.append({'Key': 'Purpose', 'Value': 'backup'})
                    elif 'log' in bucket_lower:
                        new_tags.append({'Key': 'Purpose', 'Value': 'logging'})
                    elif 'data' in bucket_lower:
                        new_tags.append({'Key': 'Purpose', 'Value': 'data-storage'})
                    elif 'static' in bucket_lower or 'web' in bucket_lower:
                        new_tags.append({'Key': 'Purpose', 'Value': 'static-content'})
                    
                    # Add project tag
                    if 'Project' not in existing_tags:
                        new_tags.append({'Key': 'Project', 'Value': 'ai-chatbot-system'})
                    
                    # Apply new tags
                    if new_tags:
                        all_tags = list(existing_tags.items()) + [(tag['Key'], tag['Value']) for tag in new_tags]
                        tag_set = [{'Key': k, 'Value': v} for k, v in all_tags]
                        
                        self.s3.put_bucket_tagging(
                            Bucket=bucket_name,
                            Tagging={'TagSet': tag_set}
                        )
                        tagged_count += 1
                        
                except Exception as e:
                    logger.error(f"Error tagging bucket {bucket_name}: {e}")
                    continue
            
            return tagged_count
            
        except Exception as e:
            logger.error(f"Error tagging S3 buckets: {e}")
            return 0

class ChargebackReportGenerator:
    """Generate detailed chargeback reports for teams and projects"""
    
    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.session_factory = sessionmaker(bind=self.engine)
        
    async def generate_monthly_chargeback(self, month: int, year: int) -> Dict[str, Any]:
        """Generate monthly chargeback report"""
        logger.info(f"Generating chargeback report for {year}-{month:02d}")
        
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        chargeback_data = {
            'report_period': f"{year}-{month:02d}",
            'generated_date': datetime.now().isoformat(),
            'teams': {},
            'projects': {},
            'total_costs': 0.0,
            'cost_categories': {}
        }
        
        # This would integrate with your cost database
        # For demonstration, we'll use sample data structure
        
        sample_teams = {
            'backend': {'compute': 2500.00, 'storage': 800.00, 'database': 1200.00},
            'frontend': {'compute': 1500.00, 'storage': 300.00, 'network': 600.00},
            'data': {'compute': 3000.00, 'storage': 2000.00, 'ai_api': 4500.00},
            'devops': {'compute': 1000.00, 'monitoring': 500.00, 'security': 300.00}
        }
        
        total_cost = 0
        for team, costs in sample_teams.items():
            team_total = sum(costs.values())
            total_cost += team_total
            
            chargeback_data['teams'][team] = {
                'total_cost': team_total,
                'cost_breakdown': costs,
                'cost_per_category': {k: v for k, v in costs.items()}
            }
        
        chargeback_data['total_costs'] = total_cost
        
        return chargeback_data
    
    async def generate_detailed_report(self, team: str, start_date: datetime, end_date: datetime) -> bytes:
        """Generate detailed PDF report for a specific team"""
        logger.info(f"Generating detailed report for team {team}")
        
        # This would generate a PDF report using libraries like reportlab
        # For now, we'll return a placeholder
        report_content = f"""
        CHARGEBACK REPORT
        Team: {team}
        Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
        
        COST BREAKDOWN:
        - Compute: $2,500.00
        - Storage: $800.00
        - Database: $1,200.00
        - Network: $400.00
        
        Total: $4,900.00
        """
        
        return report_content.encode('utf-8')

class SavingsRecommendationEngine:
    """Generate intelligent cost savings recommendations"""
    
    def __init__(self):
        self.recommendation_rules = [
            self._right_sizing_recommendations,
            self._reserved_instance_recommendations,
            self._storage_optimization_recommendations,
            self._api_optimization_recommendations
        ]
    
    async def generate_recommendations(self, cost_records: List[CostRecord], 
                                    usage_metrics: List[APIUsageMetrics]) -> List[SavingsRecommendation]:
        """Generate comprehensive savings recommendations"""
        logger.info("Generating cost savings recommendations")
        
        all_recommendations = []
        
        for rule_func in self.recommendation_rules:
            try:
                recommendations = await rule_func(cost_records, usage_metrics)
                all_recommendations.extend(recommendations)
            except Exception as e:
                logger.error(f"Error in recommendation rule {rule_func.__name__}: {e}")
        
        # Sort by estimated savings (descending)
        all_recommendations.sort(key=lambda x: x.estimated_monthly_savings, reverse=True)
        
        # Update Prometheus metrics
        for rec in all_recommendations:
            savings_identified.labels(category=rec.category).set(rec.estimated_monthly_savings)
        
        return all_recommendations
    
    async def _right_sizing_recommendations(self, cost_records: List[CostRecord], 
                                          usage_metrics: List[APIUsageMetrics]) -> List[SavingsRecommendation]:
        """Generate right-sizing recommendations"""
        recommendations = []
        
        # Analyze compute costs for over-provisioned resources
        compute_costs = [record for record in cost_records if record.cost_category == CostCategory.COMPUTE]
        
        if compute_costs:
            # Group by service and analyze utilization patterns
            # This is a simplified example - in production, you'd analyze CloudWatch metrics
            
            high_cost_services = [record for record in compute_costs if record.amount > 1000]
            
            for record in high_cost_services:
                # Simulate utilization analysis (in practice, you'd query CloudWatch)
                simulated_utilization = 35  # 35% average utilization
                
                if simulated_utilization < 50:
                    potential_savings = record.amount * 0.3  # Estimate 30% savings
                    
                    recommendation = SavingsRecommendation(
                        category="Right-sizing",
                        service_name=record.service_name,
                        recommendation_type="Instance Downsizing",
                        description=f"Service {record.service_name} shows low utilization ({simulated_utilization}%). "
                                  f"Consider downsizing instances to reduce costs.",
                        estimated_monthly_savings=potential_savings,
                        implementation_effort="Low",
                        confidence_score=0.7,
                        created_date=datetime.now()
                    )
                    recommendations.append(recommendation)
        
        return recommendations
    
    async def _reserved_instance_recommendations(self, cost_records: List[CostRecord], 
                                               usage_metrics: List[APIUsageMetrics]) -> List[SavingsRecommendation]:
        """Generate reserved instance recommendations"""
        recommendations = []
        
        # Analyze stable workloads for RI opportunities
        stable_compute_costs = [record for record in cost_records 
                              if record.cost_category == CostCategory.COMPUTE and record.amount > 500]
        
        for record in stable_compute_costs:
            # Simulate RI savings calculation
            potential_savings = record.amount * 0.4  # Estimate 40% savings with 3-year RI
            
            recommendation = SavingsRecommendation(
                category="Reserved Instances",
                service_name=record.service_name,
                recommendation_type="Reserved Instance Purchase",
                description=f"Service {record.service_name} shows consistent usage pattern. "
                          f"Consider purchasing reserved instances for 40% cost reduction.",
                estimated_monthly_savings=potential_savings,
                implementation_effort="Medium",
                confidence_score=0.8,
                created_date=datetime.now()
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _storage_optimization_recommendations(self, cost_records: List[CostRecord], 
                                                  usage_metrics: List[APIUsageMetrics]) -> List[SavingsRecommendation]:
        """Generate storage optimization recommendations"""
        recommendations = []
        
        storage_costs = [record for record in cost_records if record.cost_category == CostCategory.STORAGE]
        
        for record in storage_costs:
            if record.amount > 200:  # Focus on significant storage costs
                # Simulate storage class analysis
                potential_savings = record.amount * 0.25  # Estimate 25% savings with lifecycle policies
                
                recommendation = SavingsRecommendation(
                    category="Storage Optimization",
                    service_name=record.service_name,
                    recommendation_type="Storage Class Optimization",
                    description=f"Implement S3 lifecycle policies for {record.service_name} to move "
                              f"infrequently accessed data to cheaper storage classes.",
                    estimated_monthly_savings=potential_savings,
                    implementation_effort="Low",
                    confidence_score=0.6,
                    created_date=datetime.now()
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _api_optimization_recommendations(self, cost_records: List[CostRecord], 
                                             usage_metrics: List[APIUsageMetrics]) -> List[SavingsRecommendation]:
        """Generate AI API optimization recommendations"""
        recommendations = []
        
        # Analyze AI API usage for optimization opportunities
        for metric in usage_metrics:
            if metric.total_cost > 1000:  # Focus on high-cost API usage
                # Simulate optimization analysis
                if metric.average_tokens_per_request > 1000:  # High token usage
                    potential_savings = metric.total_cost * 0.20  # 20% savings through prompt optimization
                    
                    recommendation = SavingsRecommendation(
                        category="AI API Optimization",
                        service_name=metric.provider,
                        recommendation_type="Prompt Optimization",
                        description=f"High token usage detected for {metric.model}. "
                                  f"Optimize prompts to reduce token consumption and costs.",
                        estimated_monthly_savings=potential_savings,
                        implementation_effort="Medium",
                        confidence_score=0.7,
                        created_date=datetime.now()
                    )
                    recommendations.append(recommendation)
        
        return recommendations

class FinOpsManager:
    """Main FinOps platform orchestrator"""
    
    def __init__(self, aws_session: boto3.Session, redis_client: redis.Redis, 
                 db_connection_string: str, email_config: Dict[str, str]):
        self.cost_analyzer = CloudCostAnalyzer(aws_session)
        self.api_optimizer = AIAPIOptimizer(redis_client)
        self.anomaly_detector = CostAnomalyDetector()
        self.resource_tagger = ResourceTagger(aws_session)
        self.chargeback_generator = ChargebackReportGenerator(db_connection_string)
        self.savings_engine = SavingsRecommendationEngine()
        self.email_config = email_config
        
    async def run_daily_finops_pipeline(self) -> Dict[str, Any]:
        """Run the daily FinOps pipeline"""
        logger.info("Starting daily FinOps pipeline")
        
        pipeline_results = {
            'start_time': datetime.now(),
            'cost_analysis': {},
            'anomaly_detection': {},
            'optimization_recommendations': {},
            'resource_tagging': {},
            'alerts_sent': 0
        }
        
        try:
            # 1. Cost Analysis and Allocation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            cost_records = await self.cost_analyzer.get_service_costs(start_date, end_date)
            allocated_costs = await self.cost_analyzer.allocate_shared_costs(cost_records)
            
            pipeline_results['cost_analysis'] = {
                'total_records': len(allocated_costs),
                'total_cost': sum(record.amount for record in allocated_costs)
            }
            
            # 2. Anomaly Detection
            anomalies = await self.anomaly_detector.detect_anomalies(allocated_costs)
            pipeline_results['anomaly_detection'] = {
                'total_anomalies': len(anomalies),
                'critical_anomalies': len([a for a in anomalies if a.severity == AlertSeverity.CRITICAL]),
                'high_anomalies': len([a for a in anomalies if a.severity == AlertSeverity.HIGH])
            }
            
            # 3. AI API Optimization
            api_metrics = await self.api_optimizer.analyze_api_usage(start_date, end_date)
            api_recommendations = await self.api_optimizer.optimize_model_selection(api_metrics)
            
            # 4. Generate Savings Recommendations
            all_recommendations = await self.savings_engine.generate_recommendations(allocated_costs, api_metrics)
            all_recommendations.extend(api_recommendations)
            
            pipeline_results['optimization_recommendations'] = {
                'total_recommendations': len(all_recommendations),
                'estimated_monthly_savings': sum(rec.estimated_monthly_savings for rec in all_recommendations)
            }
            
            # 5. Resource Tagging
            tagging_results = await self.resource_tagger.auto_tag_resources()
            pipeline_results['resource_tagging'] = tagging_results
            
            # 6. Send Alerts for Critical Issues
            critical_anomalies = [a for a in anomalies if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]]
            if critical_anomalies:
                alerts_sent = await self._send_anomaly_alerts(critical_anomalies)
                pipeline_results['alerts_sent'] = alerts_sent
            
            # 7. Update Prometheus Metrics
            cost_allocation_operations.labels(service='all', status='success').inc()
            
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration'] = (pipeline_results['end_time'] - pipeline_results['start_time']).total_seconds()
            
            logger.info(f"FinOps pipeline completed successfully in {pipeline_results['duration']} seconds")
            
        except Exception as e:
            logger.error(f"Error in FinOps pipeline: {e}")
            cost_allocation_operations.labels(service='all', status='error').inc()
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    async def _send_anomaly_alerts(self, anomalies: List[CostAnomaly]) -> int:
        """Send email alerts for cost anomalies"""
        logger.info(f"Sending alerts for {len(anomalies)} anomalies")
        
        alerts_sent = 0
        
        try:
            # Group anomalies by severity
            critical_anomalies = [a for a in anomalies if a.severity == AlertSeverity.CRITICAL]
            high_anomalies = [a for a in anomalies if a.severity == AlertSeverity.HIGH]
            
            if critical_anomalies or high_anomalies:
                subject = f"🚨 Cost Anomalies Detected - {len(critical_anomalies)} Critical, {len(high_anomalies)} High"
                
                body = self._generate_anomaly_email(critical_anomalies + high_anomalies)
                
                await self._send_email(
                    to_emails=self.email_config.get('alert_recipients', []),
                    subject=subject,
                    body=body
                )
                
                alerts_sent = 1
                
        except Exception as e:
            logger.error(f"Error sending anomaly alerts: {e}")
        
        return alerts_sent
    
    def _generate_anomaly_email(self, anomalies: List[CostAnomaly]) -> str:
        """Generate HTML email content for anomaly alerts"""
        html_content = """
        <html>
        <body>
        <h2>Cost Anomaly Alert</h2>
        <p>The following cost anomalies have been detected:</p>
        <table border="1" style="border-collapse: collapse;">
        <tr>
            <th>Service</th>
            <th>Severity</th>
            <th>Current Cost</th>
            <th>Expected Cost</th>
            <th>Variance</th>
            <th>Recommendation</th>
        </tr>
        """
        
        for anomaly in anomalies:
            severity_color = {
                AlertSeverity.CRITICAL: "#ff4444",
                AlertSeverity.HIGH: "#ff8800",
                AlertSeverity.MEDIUM: "#ffaa00",
                AlertSeverity.LOW: "#44aa44"
            }.get(anomaly.severity, "#666666")
            
            html_content += f"""
            <tr>
                <td>{anomaly.service_name}</td>
                <td style="color: {severity_color}; font-weight: bold;">{anomaly.severity.value.upper()}</td>
                <td>${anomaly.current_cost:.2f}</td>
                <td>${anomaly.expected_cost:.2f}</td>
                <td>{anomaly.variance_percentage:.1f}%</td>
                <td>{anomaly.recommendation}</td>
            </tr>
            """
        
        html_content += """
        </table>
        <p>Please investigate these anomalies and take appropriate action.</p>
        <p><em>This alert was generated by the AI Chatbot FinOps Platform</em></p>
        </body>
        </html>
        """
        
        return html_content
    
    async def _send_email(self, to_emails: List[str], subject: str, body: str):
        """Send email notification"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(to_emails)
            
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent successfully to {len(to_emails)} recipients")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")

async def main():
    """Main entry point for FinOps platform"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize AWS session
    aws_session = boto3.Session()
    
    # Initialize Redis client
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # Database connection
    db_connection_string = "postgresql://username:password@localhost:5432/finops"
    
    # Email configuration
    email_config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'finops@company.com',
        'password': 'app_password',
        'from_email': 'finops@company.com',
        'alert_recipients': ['devops@company.com', 'finance@company.com']
    }
    
    # Initialize FinOps Manager
    finops_manager = FinOpsManager(
        aws_session=aws_session,
        redis_client=redis_client,
        db_connection_string=db_connection_string,
        email_config=email_config
    )
    
    # Run daily pipeline
    results = await finops_manager.run_daily_finops_pipeline()
    
    print("FinOps Pipeline Results:")
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())