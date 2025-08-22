import asyncio
from typing import Callable, Any, Optional, List, Type
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)


class RetryStrategy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions or [Exception]
        
    def _calculate_delay(self, attempt: int) -> float:
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter (0-25% of delay)
            delay += delay * random.uniform(0, 0.25)
        
        return delay
    
    def _should_retry(self, exception: Exception) -> bool:
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.retry_exceptions
        )
    
    async def execute(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry successful after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e) or attempt >= self.max_retries:
                    logger.error(f"Failed after {attempt + 1} attempts: {e}")
                    raise e
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                
                if on_retry:
                    await on_retry(attempt, e, delay)
                
                await asyncio.sleep(delay)
        
        raise last_exception


class ExponentialBackoffRetry(RetryStrategy):
    def __init__(self, max_retries: int = 5, initial_delay: float = 1.0):
        super().__init__(
            max_retries=max_retries,
            base_delay=initial_delay,
            exponential_base=2.0,
            jitter=True
        )


class LinearRetry(RetryStrategy):
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        super().__init__(
            max_retries=max_retries,
            base_delay=delay,
            exponential_base=1.0,  # Linear progression
            jitter=False
        )


class FibonacciRetry(RetryStrategy):
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        super().__init__(
            max_retries=max_retries,
            base_delay=base_delay
        )
        self.fib_sequence = [1, 1]
    
    def _calculate_delay(self, attempt: int) -> float:
        # Generate Fibonacci sequence up to current attempt
        while len(self.fib_sequence) <= attempt:
            self.fib_sequence.append(
                self.fib_sequence[-1] + self.fib_sequence[-2]
            )
        
        delay = min(
            self.base_delay * self.fib_sequence[attempt],
            self.max_delay
        )
        
        if self.jitter:
            delay += delay * random.uniform(0, 0.25)
        
        return delay


class AdaptiveRetry:
    def __init__(self, initial_strategy: RetryStrategy):
        self.strategy = initial_strategy
        self.success_history = []
        self.failure_history = []
        self.adaptation_threshold = 10
        
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        start_time = datetime.now()
        
        try:
            result = await self.strategy.execute(func, *args, **kwargs)
            self._record_success(start_time)
            return result
            
        except Exception as e:
            self._record_failure(start_time)
            self._adapt_strategy()
            raise e
    
    def _record_success(self, start_time: datetime):
        duration = (datetime.now() - start_time).total_seconds()
        self.success_history.append({
            "timestamp": datetime.now(),
            "duration": duration
        })
        
        # Keep only recent history
        if len(self.success_history) > 100:
            self.success_history.pop(0)
    
    def _record_failure(self, start_time: datetime):
        duration = (datetime.now() - start_time).total_seconds()
        self.failure_history.append({
            "timestamp": datetime.now(),
            "duration": duration
        })
        
        # Keep only recent history
        if len(self.failure_history) > 100:
            self.failure_history.pop(0)
    
    def _adapt_strategy(self):
        recent_failures = len([
            f for f in self.failure_history
            if (datetime.now() - f["timestamp"]).total_seconds() < 300
        ])
        
        if recent_failures > self.adaptation_threshold:
            # Too many recent failures - increase delays
            self.strategy.base_delay = min(
                self.strategy.base_delay * 1.5,
                30.0
            )
            self.strategy.max_retries = min(
                self.strategy.max_retries + 1,
                10
            )
            logger.info(f"Adapted retry strategy: base_delay={self.strategy.base_delay:.1f}s, "
                       f"max_retries={self.strategy.max_retries}")
        
        elif recent_failures < 2 and len(self.success_history) > 20:
            # System is stable - can reduce delays
            self.strategy.base_delay = max(
                self.strategy.base_delay * 0.8,
                0.5
            )
            logger.info(f"Optimized retry strategy: base_delay={self.strategy.base_delay:.1f}s")