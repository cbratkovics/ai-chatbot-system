"""Stream handler for SSE and chunked responses."""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)


class StreamHandler:
    """Handles streaming responses for SSE and chunked transfers."""

    def __init__(
        self,
        buffer_size: int = 10,
        chunk_size: int = 1024,
        timeout: int = 60,
        metrics_collector: Optional[Any] = None,
    ):
        """Initialize stream handler.

        Args:
            buffer_size: Buffer size for streaming
            chunk_size: Size of each chunk
            timeout: Stream timeout in seconds
            metrics_collector: Metrics collector instance
        """
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.metrics_collector = metrics_collector

    def create_sse_stream(
        self, data_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Create Server-Sent Events stream.

        Args:
            data_generator: Async generator of data

        Yields:
            SSE formatted chunks
        """

        async def _sse_generator():
            try:
                async for data in data_generator:
                    yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                logger.error(f"SSE stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return _sse_generator()

    def chunk_response(self, data: str, chunk_size: Optional[int] = None) -> List[str]:
        """Split response into chunks.

        Args:
            data: Data to chunk
            chunk_size: Size of each chunk

        Returns:
            List of chunks
        """
        chunk_size = chunk_size or self.chunk_size
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    async def stream_with_timeout(
        self, data_generator: AsyncGenerator[Any, None], timeout: Optional[int] = None
    ) -> AsyncGenerator[Any, None]:
        """Stream with timeout protection.

        Args:
            data_generator: Async generator of data
            timeout: Timeout in seconds

        Yields:
            Data chunks
        """
        timeout = timeout or self.timeout

        async def _timeout_generator():
            try:
                async for data in asyncio.wait_for(
                    self._consume_generator(data_generator), timeout=timeout
                ):
                    yield data
            except asyncio.TimeoutError:
                logger.error(f"Stream timeout after {timeout} seconds")
                raise

        return _timeout_generator()

    async def _consume_generator(
        self, data_generator: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[Any, None]:
        """Consume async generator.

        Args:
            data_generator: Async generator

        Yields:
            Data from generator
        """
        async for data in data_generator:
            yield data

    async def handle_stream(
        self, data_generator: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[Any, None]:
        """Handle stream with error recovery.

        Args:
            data_generator: Async generator of data

        Yields:
            Data chunks
        """

        async def _handled_generator():
            try:
                async for data in data_generator:
                    yield data
            except Exception as e:
                logger.error(f"Stream error: {e}")
                raise

        return _handled_generator()

    async def buffer_stream(
        self, data_generator: AsyncGenerator[Any, None], buffer_size: Optional[int] = None
    ) -> AsyncGenerator[List[Any], None]:
        """Buffer stream data.

        Args:
            data_generator: Async generator of data
            buffer_size: Buffer size

        Yields:
            Buffered data batches
        """
        buffer_size = buffer_size or self.buffer_size

        async def _buffered_generator():
            buffer = []

            async for data in data_generator:
                buffer.append(data)

                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []

            if buffer:
                yield buffer

        return _buffered_generator()

    async def compress_stream(
        self, data_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """Compress stream data.

        Args:
            data_generator: Async generator of string data

        Yields:
            Compressed data chunks
        """
        import zlib

        async def _compressed_generator():
            compressor = zlib.compressobj()

            async for data in data_generator:
                compressed = compressor.compress(data.encode())
                if compressed:
                    yield compressed

            final = compressor.flush()
            if final:
                yield final

        return _compressed_generator()

    async def rate_limit_stream(
        self, data_generator: AsyncGenerator[Any, None], rate: float = 10.0
    ) -> AsyncGenerator[Any, None]:
        """Rate limit stream output.

        Args:
            data_generator: Async generator of data
            rate: Maximum items per second

        Yields:
            Rate-limited data
        """
        interval = 1.0 / rate

        async def _rate_limited_generator():
            async for data in data_generator:
                yield data
                await asyncio.sleep(interval)

        return _rate_limited_generator()

    async def stream_with_metrics(
        self, data_generator: AsyncGenerator[Any, None], stream_id: str
    ) -> AsyncGenerator[Any, None]:
        """Stream with metrics collection.

        Args:
            data_generator: Async generator of data
            stream_id: Stream identifier

        Yields:
            Data chunks
        """

        async def _metered_generator():
            chunks_sent = 0
            bytes_sent = 0
            start_time = asyncio.get_event_loop().time()

            try:
                async for data in data_generator:
                    yield data

                    chunks_sent += 1
                    if isinstance(data, (str, bytes)):
                        bytes_sent += len(data)

                    if self.metrics_collector:
                        self.metrics_collector.record_gauge(
                            "stream.chunks_sent", chunks_sent, {"stream_id": stream_id}
                        )

            finally:
                duration = asyncio.get_event_loop().time() - start_time

                if self.metrics_collector:
                    self.metrics_collector.record_latency(
                        "stream.duration", duration, {"stream_id": stream_id}
                    )
                    self.metrics_collector.record_gauge(
                        "stream.bytes_sent", bytes_sent, {"stream_id": stream_id}
                    )

                logger.info(
                    f"Stream {stream_id} completed: "
                    f"{chunks_sent} chunks, {bytes_sent} bytes, "
                    f"{duration:.2f}s"
                )

        return _metered_generator()

    async def merge_streams(
        self, *generators: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[Any, None]:
        """Merge multiple streams.

        Args:
            *generators: Multiple async generators

        Yields:
            Merged data from all streams
        """

        async def _merged_generator():
            tasks = [asyncio.create_task(self._consume_to_queue(gen, queue)) for gen in generators]

            queue = asyncio.Queue()

            while any(not task.done() for task in tasks):
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield data
                except asyncio.TimeoutError:
                    continue

            while not queue.empty():
                yield await queue.get()

        return _merged_generator()

    async def _consume_to_queue(self, generator: AsyncGenerator[Any, None], queue: asyncio.Queue):
        """Consume generator to queue.

        Args:
            generator: Async generator
            queue: Target queue
        """
        async for data in generator:
            await queue.put(data)
