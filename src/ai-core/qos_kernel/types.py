import uuid
from dataclasses import dataclass, field
from typing import Dict, Any

try:
    from qiskit import QuantumCircuit
except ImportError:
    class QuantumCircuit: pass

@dataclass(order=True)
class QuantumJob:
    """Standard Unit of Work for the QOS."""
    priority: int
    circuit: QuantumCircuit = field(compare=False)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    min_fidelity_required: float = field(default=0.90, compare=False)

@dataclass
class ExecutionResult:
    job_id: str
    counts: Dict[str, int]
    mitigated: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
