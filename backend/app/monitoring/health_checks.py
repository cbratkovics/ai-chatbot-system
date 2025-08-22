from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    def __init__(
        self,
        name: str,
        check_func: Callable,
        critical: bool = False,
        timeout: float = 5.0
    ):
        self.name = name
        self.check_func = check_func
        self.critical = critical
        self.timeout = timeout
        self.last_check: Optional[datetime] = None
        self.last_status = HealthStatus.HEALTHY
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0
        
    async def execute(self) -> Dict[str, Any]:
        try:
            result = await asyncio.wait_for(
                self.check_func(),
                timeout=self.timeout
            )
            
            self.last_check = datetime.now()
            self.consecutive_failures = 0
            self.last_error = None
            
            if isinstance(result, bool):
                self.last_status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            elif isinstance(result, dict):
                status_str = result.get("status", "healthy")
                self.last_status = HealthStatus[status_str.upper()]
            else:
                self.last_status = HealthStatus.HEALTHY
            
            return {
                "name": self.name,
                "status": self.last_status.value,
                "critical": self.critical,
                "last_check": self.last_check.isoformat(),
                "details": result if isinstance(result, dict) else None
            }
            
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            self.last_error = "Health check timed out"
            self.last_status = HealthStatus.UNHEALTHY
            
            return {
                "name": self.name,
                "status": HealthStatus.UNHEALTHY.value,
                "critical": self.critical,
                "error": self.last_error,
                "consecutive_failures": self.consecutive_failures
            }
            
        except Exception as e:
            self.consecutive_failures += 1
            self.last_error = str(e)
            self.last_status = HealthStatus.UNHEALTHY
            
            return {
                "name": self.name,
                "status": HealthStatus.UNHEALTHY.value,
                "critical": self.critical,
                "error": self.last_error,
                "consecutive_failures": self.consecutive_failures
            }


class HealthCheckManager:
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.check_interval = 30  # seconds
        self.is_running = False
        self._background_task = None
        
    def register_check(
        self,
        name: str,
        check_func: Callable,
        critical: bool = False,
        timeout: float = 5.0
    ):
        self.checks[name] = HealthCheck(name, check_func, critical, timeout)
        logger.info(f"Registered health check: {name} (critical: {critical})")
    
    def unregister_check(self, name: str):
        if name in self.checks:
            del self.checks[name]
            logger.info(f"Unregistered health check: {name}")
    
    async def run_check(self, name: str) -> Optional[Dict[str, Any]]:
        if name not in self.checks:
            return None
        
        return await self.checks[name].execute()
    
    async def run_all_checks(self) -> Dict[str, Any]:
        results = await asyncio.gather(
            *[check.execute() for check in self.checks.values()],
            return_exceptions=True
        )
        
        check_results = []
        overall_status = HealthStatus.HEALTHY
        
        for result in results:
            if isinstance(result, Exception):
                check_results.append({
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(result)
                })
                overall_status = HealthStatus.UNHEALTHY
            else:
                check_results.append(result)
                
                if result["status"] == HealthStatus.UNHEALTHY.value:
                    if result.get("critical"):
                        overall_status = HealthStatus.UNHEALTHY
                    elif overall_status != HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.DEGRADED
                elif result["status"] == HealthStatus.DEGRADED.value:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": check_results
        }
    
    async def start_background_checks(self):
        if self.is_running:
            return
        
        self.is_running = True
        self._background_task = asyncio.create_task(self._run_periodic_checks())
        logger.info("Started background health checks")
    
    async def stop_background_checks(self):
        self.is_running = False
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped background health checks")
    
    async def _run_periodic_checks(self):
        while self.is_running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(self.check_interval)
    
    def get_readiness(self) -> Dict[str, Any]:
        # Check if all critical services are healthy
        critical_healthy = all(
            check.last_status == HealthStatus.HEALTHY
            for check in self.checks.values()
            if check.critical
        )
        
        return {
            "ready": critical_healthy,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_liveness(self) -> Dict[str, Any]:
        # Simple liveness check - just return alive
        return {
            "alive": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_startup(self) -> Dict[str, Any]:
        # Check if minimum services are initialized
        initialized = len(self.checks) > 0
        
        return {
            "started": initialized,
            "checks_registered": len(self.checks),
            "timestamp": datetime.now().isoformat()
        }