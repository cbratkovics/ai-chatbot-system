import asyncio
from typing import Optional, Any, Dict
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


class BackpressureHandler:
    def __init__(
        self,
        max_queue_size: int = 1000,
        max_rate_per_second: int = 100,
        buffer_high_watermark: int = 800,
        buffer_low_watermark: int = 200
    ):
        self.max_queue_size = max_queue_size
        self.max_rate_per_second = max_rate_per_second
        self.buffer_high_watermark = buffer_high_watermark
        self.buffer_low_watermark = buffer_low_watermark
        
        self.queue = deque(maxlen=max_queue_size)
        self.is_paused = False
        self.last_send_time = time.time()
        self.messages_sent_in_window = 0
        self.window_start_time = time.time()
        
    async def can_send(self) -> bool:
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.window_start_time >= 1.0:
            self.messages_sent_in_window = 0
            self.window_start_time = current_time
        
        # Check rate limit
        if self.messages_sent_in_window >= self.max_rate_per_second:
            return False
        
        # Check if paused due to backpressure
        if self.is_paused:
            if len(self.queue) <= self.buffer_low_watermark:
                self.is_paused = False
                logger.info("Backpressure released")
            else:
                return False
        
        # Check buffer size
        if len(self.queue) >= self.buffer_high_watermark:
            self.is_paused = True
            logger.warning(f"Backpressure applied: queue size {len(self.queue)}")
            return False
        
        return True
    
    async def add_message(self, message: Any) -> bool:
        if len(self.queue) >= self.max_queue_size:
            logger.error("Queue full, dropping message")
            return False
        
        self.queue.append(message)
        return True
    
    async def get_message(self) -> Optional[Any]:
        if not self.queue:
            return None
        
        if await self.can_send():
            self.messages_sent_in_window += 1
            return self.queue.popleft()
        
        return None
    
    async def process_queue(self, send_callback):
        while True:
            message = await self.get_message()
            if message:
                try:
                    await send_callback(message)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    # Re-queue the message
                    self.queue.appendleft(message)
                    await asyncio.sleep(0.1)
            else:
                # No messages or can't send due to backpressure
                await asyncio.sleep(0.01)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "queue_size": len(self.queue),
            "is_paused": self.is_paused,
            "messages_sent_in_window": self.messages_sent_in_window,
            "max_queue_size": self.max_queue_size,
            "max_rate_per_second": self.max_rate_per_second,
            "buffer_utilization": len(self.queue) / self.max_queue_size * 100
        }
    
    def clear_queue(self):
        self.queue.clear()
        self.is_paused = False
        self.messages_sent_in_window = 0
        logger.info("Backpressure queue cleared")