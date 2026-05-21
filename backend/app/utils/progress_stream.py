import asyncio
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ProgressManager:
    def __init__(self):
        self.queues = defaultdict(set)

    async def publish(self, progress_id: str, data: dict):
        if progress_id in self.queues:
            for queue in list(self.queues[progress_id]):
                await queue.put(data)

    async def listen(self, progress_id: str):
        queue = asyncio.Queue()
        self.queues[progress_id].add(queue)
        try:
            while True:
                data = await queue.get()
                yield data
        finally:
            self.queues[progress_id].discard(queue)
            if not self.queues[progress_id]:
                del self.queues[progress_id]


async def _yield_control():
    await asyncio.sleep(0)


progress_manager = ProgressManager()
