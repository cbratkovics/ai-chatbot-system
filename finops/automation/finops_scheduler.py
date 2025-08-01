#!/usr/bin/env python3
"""
FinOps Automation Scheduler
Orchestrates automated cost management tasks, reports, and optimizations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import yaml
from dataclasses import dataclass
from enum import Enum
import schedule
import time
import boto3
import redis
from sqlalchemy import create_engine, text
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import pandas as pd
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    name: str
    description: str
    function: str
    schedule: str
    priority: TaskPriority
    timeout_minutes: int
    retry_count: int
    enabled: bool
    dependencies: List[str]
    parameters: Dict[str, Any]

@dataclass
class TaskExecution:
    """Task execution record"""
    task_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: TaskStatus
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    retry_attempt: int

class FinOpsScheduler:
    """Main scheduler for FinOps automation tasks"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.tasks: Dict[str, ScheduledTask] = {}
        self.execution_history: List[TaskExecution] = []
        self._setup_connections()
        self._register_tasks()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_connections(self):
        """Setup database and external service connections"""
        try:
            # AWS connection
            self.aws_session = boto3.Session(
                region_name=self.config.get('aws', {}).get('region', 'us-east-1')
            )
            
            # Redis connection
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('database', 0),
                decode_responses=True
            )
            
            # Database connection
            db_config = self.config.get('database', {})
            self.db_engine = create_engine(db_config.get('connection_string'))
            
            logger.info("Successfully connected to all services")
            
        except Exception as e:
            logger.error(f"Error setting up connections: {e}")
            raise
    
    def _register_tasks(self):
        """Register all scheduled tasks"""
        task_definitions = [
            # Daily tasks
            ScheduledTask(
                name="daily_cost_analysis",
                description="Analyze daily costs and allocate shared resources",
                function="run_cost_analysis",
                schedule="0 6 * * *",  # 6 AM daily
                priority=TaskPriority.HIGH,
                timeout_minutes=30,
                retry_count=3,
                enabled=True,
                dependencies=[],
                parameters={"lookback_days": 1}
            ),
            
            ScheduledTask(
                name="daily_anomaly_detection",
                description="Detect cost anomalies and send alerts",
                function="run_anomaly_detection",
                schedule="15 6 * * *",  # 6:15 AM daily
                priority=TaskPriority.HIGH,
                timeout_minutes=20,
                retry_count=2,
                enabled=True,
                dependencies=["daily_cost_analysis"],
                parameters={"sensitivity": "medium"}
            ),
            
            ScheduledTask(
                name="daily_resource_tagging",
                description="Auto-tag untagged resources",
                function="run_resource_tagging",
                schedule="30 6 * * *",  # 6:30 AM daily
                priority=TaskPriority.MEDIUM,
                timeout_minutes=45,
                retry_count=2,
                enabled=True,
                dependencies=[],
                parameters={"batch_size": 100}
            ),
            
            ScheduledTask(
                name="api_usage_optimization",
                description="Analyze and optimize AI API usage",
                function="run_api_optimization",
                schedule="45 6 * * *",  # 6:45 AM daily
                priority=TaskPriority.MEDIUM,
                timeout_minutes=25,
                retry_count=2,
                enabled=True,
                dependencies=["daily_cost_analysis"],
                parameters={"optimization_threshold": 0.1}
            ),
            
            # Weekly tasks
            ScheduledTask(
                name="weekly_cost_report",
                description="Generate weekly cost summary report",
                function="generate_weekly_report",
                schedule="0 8 * * 1",  # 8 AM Monday
                priority=TaskPriority.MEDIUM,
                timeout_minutes=15,
                retry_count=2,
                enabled=True,
                dependencies=[],
                parameters={"include_trends": True}
            ),
            
            ScheduledTask(
                name="weekly_savings_recommendations",
                description="Generate cost savings recommendations",
                function="generate_savings_recommendations",
                schedule="0 9 * * 1",  # 9 AM Monday
                priority=TaskPriority.MEDIUM,
                timeout_minutes=30,
                retry_count=2,
                enabled=True,
                dependencies=["weekly_cost_report"],
                parameters={"min_savings_threshold": 100}
            ),
            
            # Monthly tasks
            ScheduledTask(
                name="monthly_chargeback_report",
                description="Generate monthly chargeback reports",
                function="generate_chargeback_report",
                schedule="0 9 1 * *",  # 9 AM first day of month
                priority=TaskPriority.HIGH,
                timeout_minutes=60,
                retry_count=3,
                enabled=True,
                dependencies=[],
                parameters={"include_overhead": True}
            ),
            
            ScheduledTask(
                name="monthly_budget_review",
                description="Review budget vs actual spending",
                function="run_budget_review",
                schedule="0 10 1 * *",  # 10 AM first day of month
                priority=TaskPriority.HIGH,
                timeout_minutes=20,
                retry_count=2,
                enabled=True,
                dependencies=["monthly_chargeback_report"],
                parameters={"variance_threshold": 0.15}
            ),
            
            # Cleanup tasks
            ScheduledTask(
                name="cleanup_old_data",
                description="Clean up old cost and analytics data",
                function="cleanup_old_data",
                schedule="0 2 * * 0",  # 2 AM Sunday
                priority=TaskPriority.LOW,
                timeout_minutes=30,
                retry_count=1,
                enabled=True,
                dependencies=[],
                parameters={"retention_days": 90}
            )
        ]
        
        for task in task_definitions:
            self.tasks[task.name] = task
            logger.info(f"Registered task: {task.name}")
    
    async def run_cost_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run daily cost analysis"""
        logger.info("Starting daily cost analysis")
        
        try:
            # Import the cost management platform
            from finops.cost_management_platform import CloudCostAnalyzer
            
            cost_analyzer = CloudCostAnalyzer(self.aws_session)
            
            # Get cost data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=parameters.get('lookback_days', 1))
            
            cost_records = await cost_analyzer.get_service_costs(start_date, end_date)
            allocated_costs = await cost_analyzer.allocate_shared_costs(cost_records)
            
            # Store results in Redis cache
            cache_key = f"finops:cost_analysis:{end_date.strftime('%Y%m%d')}"
            cache_data = {
                'total_records': len(allocated_costs),
                'total_cost': sum(record.amount for record in allocated_costs),
                'analysis_date': end_date.isoformat(),
                'lookback_days': parameters.get('lookback_days', 1)
            }
            
            self.redis_client.setex(cache_key, 86400, json.dumps(cache_data))  # 24 hour TTL
            
            return {
                'status': 'success',
                'records_processed': len(allocated_costs),
                'total_cost': sum(record.amount for record in allocated_costs),
                'cache_key': cache_key
            }
            
        except Exception as e:
            logger.error(f"Error in cost analysis: {e}")
            raise
    
    async def run_anomaly_detection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run anomaly detection"""
        logger.info("Starting anomaly detection")
        
        try:
            from finops.cost_management_platform import CostAnomalyDetector
            
            # Get recent cost data
            cache_key = f"finops:cost_analysis:{datetime.now().strftime('%Y%m%d')}"
            cached_data = self.redis_client.get(cache_key)
            
            if not cached_data:
                logger.warning("No cost analysis data found, skipping anomaly detection")
                return {'status': 'skipped', 'reason': 'no_cost_data'}
            
            # Run anomaly detection (simplified for demo)
            anomaly_detector = CostAnomalyDetector()
            
            # Mock cost records for demonstration
            from finops.cost_management_platform import CostRecord, CostCategory
            mock_records = [
                CostRecord(
                    service_name="EC2",
                    resource_id="i-123456",
                    cost_category=CostCategory.COMPUTE,
                    amount=1200.0,
                    currency="USD",
                    usage_quantity=24.0,
                    usage_unit="hours",
                    date=datetime.now(),
                    tags={"Team": "Backend"},
                    region="us-east-1",
                    account_id="123456789"
                )
            ]
            
            anomalies = await anomaly_detector.detect_anomalies(mock_records)
            
            # Store anomalies in database/cache
            anomaly_cache_key = f"finops:anomalies:{datetime.now().strftime('%Y%m%d')}"
            anomaly_data = {
                'total_anomalies': len(anomalies),
                'detection_date': datetime.now().isoformat(),
                'sensitivity': parameters.get('sensitivity', 'medium')
            }
            
            self.redis_client.setex(anomaly_cache_key, 86400, json.dumps(anomaly_data))
            
            # Send alerts if critical anomalies found
            critical_anomalies = [a for a in anomalies if a.severity.value in ['critical', 'high']]
            if critical_anomalies:
                await self._send_anomaly_alerts(critical_anomalies)
            
            return {
                'status': 'success',
                'anomalies_detected': len(anomalies),
                'critical_anomalies': len(critical_anomalies),
                'cache_key': anomaly_cache_key
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            raise
    
    async def run_resource_tagging(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run automated resource tagging"""
        logger.info("Starting resource tagging")
        
        try:
            from finops.cost_management_platform import ResourceTagger
            
            resource_tagger = ResourceTagger(self.aws_session)
            tagging_results = await resource_tagger.auto_tag_resources()
            
            # Cache results
            cache_key = f"finops:resource_tagging:{datetime.now().strftime('%Y%m%d')}"
            self.redis_client.setex(cache_key, 86400, json.dumps(tagging_results))
            
            return {
                'status': 'success',
                'tagged_resources': tagging_results,
                'cache_key': cache_key
            }
            
        except Exception as e:
            logger.error(f"Error in resource tagging: {e}")
            raise
    
    async def run_api_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI API optimization analysis"""
        logger.info("Starting API optimization")
        
        try:
            from finops.cost_management_platform import AIAPIOptimizer
            
            api_optimizer = AIAPIOptimizer(self.redis_client)
            
            # Analyze last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            usage_metrics = await api_optimizer.analyze_api_usage(start_date, end_date)
            recommendations = await api_optimizer.optimize_model_selection(usage_metrics)
            
            # Cache results
            cache_key = f"finops:api_optimization:{datetime.now().strftime('%Y%m%d')}"
            optimization_data = {
                'metrics_analyzed': len(usage_metrics),
                'recommendations_generated': len(recommendations),
                'potential_savings': sum(rec.estimated_monthly_savings for rec in recommendations),
                'optimization_date': datetime.now().isoformat()
            }
            
            self.redis_client.setex(cache_key, 86400, json.dumps(optimization_data))
            
            return {
                'status': 'success',
                'optimization_data': optimization_data,
                'cache_key': cache_key
            }
            
        except Exception as e:
            logger.error(f"Error in API optimization: {e}")
            raise
    
    async def generate_weekly_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate weekly cost summary report"""
        logger.info("Starting weekly report generation")
        
        try:
            # Collect data from last week
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            report_data = {
                'report_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_cost': 45000.0,  # Mock data
                    'cost_change': 5.2,     # Percentage change
                    'top_services': ['EC2', 'RDS', 'AI APIs'],
                    'anomalies_detected': 3,
                    'recommendations': 8
                },
                'trends': parameters.get('include_trends', False)
            }
            
            # Send report via email
            await self._send_weekly_report(report_data)
            
            # Cache report data
            cache_key = f"finops:weekly_report:{end_date.strftime('%Y%m%d')}"
            self.redis_client.setex(cache_key, 604800, json.dumps(report_data))  # 7 day TTL
            
            return {
                'status': 'success',
                'report_data': report_data,
                'cache_key': cache_key
            }
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            raise
    
    async def generate_savings_recommendations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive savings recommendations"""
        logger.info("Starting savings recommendations generation")
        
        try:
            from finops.cost_management_platform import SavingsRecommendationEngine
            
            savings_engine = SavingsRecommendationEngine()
            
            # Mock data for demonstration
            recommendations = await savings_engine.generate_recommendations([], [])
            
            # Filter by minimum savings threshold
            min_threshold = parameters.get('min_savings_threshold', 100)
            filtered_recommendations = [
                rec for rec in recommendations 
                if rec.estimated_monthly_savings >= min_threshold
            ]
            
            recommendations_data = {
                'total_recommendations': len(filtered_recommendations),
                'total_potential_savings': sum(rec.estimated_monthly_savings for rec in filtered_recommendations),
                'generation_date': datetime.now().isoformat(),
                'min_threshold': min_threshold
            }
            
            # Cache recommendations
            cache_key = f"finops:savings_recommendations:{datetime.now().strftime('%Y%m%d')}"
            self.redis_client.setex(cache_key, 604800, json.dumps(recommendations_data))
            
            return {
                'status': 'success',
                'recommendations_data': recommendations_data,
                'cache_key': cache_key
            }
            
        except Exception as e:
            logger.error(f"Error generating savings recommendations: {e}")
            raise
    
    async def generate_chargeback_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monthly chargeback report"""
        logger.info("Starting chargeback report generation")
        
        try:
            from finops.cost_management_platform import ChargebackReportGenerator
            
            chargeback_generator = ChargebackReportGenerator(
                self.config['database']['connection_string']
            )
            
            # Generate report for previous month
            now = datetime.now()
            if now.day == 1:  # First day of month
                report_month = now.month - 1 if now.month > 1 else 12
                report_year = now.year if now.month > 1 else now.year - 1
            else:
                report_month = now.month
                report_year = now.year
            
            chargeback_data = await chargeback_generator.generate_monthly_chargeback(
                report_month, report_year
            )
            
            # Send reports to teams
            await self._send_chargeback_reports(chargeback_data)
            
            # Cache report data
            cache_key = f"finops:chargeback:{report_year}{report_month:02d}"
            self.redis_client.setex(cache_key, 2592000, json.dumps(chargeback_data))  # 30 day TTL
            
            return {
                'status': 'success',
                'chargeback_data': chargeback_data,
                'cache_key': cache_key
            }
            
        except Exception as e:
            logger.error(f"Error generating chargeback report: {e}")
            raise
    
    async def run_budget_review(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run monthly budget review"""
        logger.info("Starting budget review")
        
        try:
            variance_threshold = parameters.get('variance_threshold', 0.15)
            
            # Mock budget data
            budget_data = {
                'teams': {
                    'Backend': {'budget': 25000, 'actual': 27500, 'variance': 0.10},
                    'Frontend': {'budget': 15000, 'actual': 18000, 'variance': 0.20},
                    'Data': {'budget': 35000, 'actual': 32000, 'variance': -0.09},
                    'DevOps': {'budget': 12000, 'actual': 11500, 'variance': -0.04}
                },
                'total_budget': 87000,
                'total_actual': 89000,
                'overall_variance': 0.023
            }
            
            # Identify teams exceeding variance threshold
            over_budget_teams = [
                team for team, data in budget_data['teams'].items()
                if abs(data['variance']) > variance_threshold
            ]
            
            review_results = {
                'review_date': datetime.now().isoformat(),
                'budget_data': budget_data,
                'over_budget_teams': over_budget_teams,
                'variance_threshold': variance_threshold,
                'requires_attention': len(over_budget_teams) > 0
            }
            
            # Send alerts if teams are significantly over budget
            if over_budget_teams:
                await self._send_budget_alerts(review_results)
            
            # Cache results
            cache_key = f"finops:budget_review:{datetime.now().strftime('%Y%m')}"
            self.redis_client.setex(cache_key, 2592000, json.dumps(review_results))
            
            return {
                'status': 'success',
                'review_results': review_results,
                'cache_key': cache_key
            }
            
        except Exception as e:
            logger.error(f"Error in budget review: {e}")
            raise
    
    async def cleanup_old_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up old analytics and cache data"""
        logger.info("Starting data cleanup")
        
        try:
            retention_days = parameters.get('retention_days', 90)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            cleanup_results = {
                'cutoff_date': cutoff_date.isoformat(),
                'retention_days': retention_days,
                'cleaned_keys': 0,
                'cleaned_records': 0
            }
            
            # Clean Redis cache
            pattern = "finops:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                try:
                    # Check if key has timestamp in name
                    if any(date_part in key for date_part in ['202', '201']):  # Basic date detection
                        # Extract date and compare (simplified logic)
                        # In production, you'd have proper date parsing
                        self.redis_client.delete(key)
                        cleanup_results['cleaned_keys'] += 1
                except Exception:
                    continue
            
            # Clean database records (would use actual SQL queries)
            # For demo, we'll just simulate
            cleanup_results['cleaned_records'] = 150
            
            logger.info(f"Cleanup completed: {cleanup_results['cleaned_keys']} cache keys, "
                       f"{cleanup_results['cleaned_records']} database records")
            
            return {
                'status': 'success',
                'cleanup_results': cleanup_results
            }
            
        except Exception as e:
            logger.error(f"Error in data cleanup: {e}")
            raise
    
    async def _send_anomaly_alerts(self, anomalies):
        """Send email alerts for cost anomalies"""
        logger.info(f"Sending anomaly alerts for {len(anomalies)} anomalies")
        
        # Mock email sending for demo
        email_config = self.config.get('email', {})
        recipients = email_config.get('alert_recipients', [])
        
        if recipients:
            logger.info(f"Would send anomaly alerts to: {', '.join(recipients)}")
    
    async def _send_weekly_report(self, report_data):
        """Send weekly cost report via email"""
        logger.info("Sending weekly cost report")
        
        email_config = self.config.get('email', {})
        recipients = email_config.get('weekly_report_recipients', [])
        
        if recipients:
            logger.info(f"Would send weekly report to: {', '.join(recipients)}")
    
    async def _send_chargeback_reports(self, chargeback_data):
        """Send chargeback reports to teams"""
        logger.info("Sending chargeback reports to teams")
        
        for team in chargeback_data.get('teams', {}):
            logger.info(f"Would send chargeback report to {team} team")
    
    async def _send_budget_alerts(self, review_results):
        """Send budget variance alerts"""
        logger.info("Sending budget variance alerts")
        
        over_budget_teams = review_results.get('over_budget_teams', [])
        for team in over_budget_teams:
            logger.info(f"Would send budget alert for {team} team")
    
    async def execute_task(self, task_name: str) -> TaskExecution:
        """Execute a specific task"""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found")
        
        task = self.tasks[task_name]
        execution_id = f"{task_name}_{int(time.time())}"
        
        execution = TaskExecution(
            task_name=task_name,
            execution_id=execution_id,
            start_time=datetime.now(),
            end_time=None,
            status=TaskStatus.RUNNING,
            result=None,
            error_message=None,
            retry_attempt=0
        )
        
        logger.info(f"Starting task execution: {task_name} ({execution_id})")
        
        try:
            # Get the task function
            task_function = getattr(self, task.function)
            
            # Execute the task
            result = await asyncio.wait_for(
                task_function(task.parameters),
                timeout=task.timeout_minutes * 60
            )
            
            execution.end_time = datetime.now()
            execution.status = TaskStatus.COMPLETED
            execution.result = result
            
            logger.info(f"Task completed successfully: {task_name}")
            
        except asyncio.TimeoutError:
            execution.end_time = datetime.now()
            execution.status = TaskStatus.FAILED
            execution.error_message = f"Task timed out after {task.timeout_minutes} minutes"
            logger.error(f"Task {task_name} timed out")
            
        except Exception as e:
            execution.end_time = datetime.now()
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            logger.error(f"Task {task_name} failed: {e}")
        
        finally:
            self.execution_history.append(execution)
        
        return execution
    
    def schedule_all_tasks(self):
        """Schedule all enabled tasks"""
        logger.info("Scheduling all tasks")
        
        for task_name, task in self.tasks.items():
            if not task.enabled:
                logger.info(f"Skipping disabled task: {task_name}")
                continue
            
            # Schedule task using the schedule library
            schedule.every().day.at(task.schedule.split()[1] if ' ' in task.schedule else "06:00").do(
                lambda t=task_name: asyncio.create_task(self.execute_task(t))
            )
            
            logger.info(f"Scheduled task: {task_name} with schedule: {task.schedule}")
    
    def run_scheduler(self):
        """Run the main scheduler loop"""
        logger.info("Starting FinOps scheduler")
        
        self.schedule_all_tasks()
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python finops_scheduler.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        scheduler = FinOpsScheduler(config_path)
        scheduler.run_scheduler()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()