import asyncio
import logging
from typing import Optional
from .types import QuantumJob
from .hal import QuantumBackend

logger = logging.getLogger("QOS.Scheduler")

class PriorityScheduler:
    def __init__(self, backend: QuantumBackend):
        self.queue = asyncio.PriorityQueue()
        self.backend = backend

    async def schedule(self, job: QuantumJob):
        logger.info(f"SCHEDULER: Job {job.job_id} queued with priority {job.priority}")
        await self.queue.put(job)

    async def next_job(self) -> Optional[QuantumJob]:
        if self.queue.empty():
            return None
        job: QuantumJob = await self.queue.get()
        health_map = await self.backend.get_qubit_health()
        avg_fidelity = sum(health_map.values()) / len(health_map)
        
        if avg_fidelity < job.min_fidelity_required:
            logger.warning(f"SCHEDULER: Low fidelity ({avg_fidelity:.2f}). Re-queueing {job.job_id}.")
            await self.queue.put(job) 
            await asyncio.sleep(1) 
            return None
        return job
