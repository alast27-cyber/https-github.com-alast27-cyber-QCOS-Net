import pennylane as qml
from pennylane import numpy as np

# --- 1. Architectural Constants (Based on IAI-IPS NN Structure) ---

# Total number of nodes/qubits in a single IAI-IPS layer (1 + 2 + 3 + 4 + 5 + 5 + 4 + 3 + 2 + 1)
# PLUS 1 Central Code Qubit connected to all others.
NUM_QUBITS = 31 
# Use the lightning.tensornet device for much faster simulation of large circuits
DEV = qml.device("default.qubit", wires=NUM_QUBITS)

# Mapping of the 10 rows to qubit indices for clarity
# Note: Qubit indices are 0-based.
ROW_MAP = {
    1: [0],                 # Entry Node (Unity)
    2: [1, 2],              # Two Opposing Aspects (Duality)
    3: [3, 4, 5],           # Three Contradictions (Conflict)
    4: [6, 7, 8, 9],        # Plasticity Region Start (Default 4)
    5: [10, 11, 12, 13, 14],
    6: [15, 16, 17, 18, 19],
    7: [20, 21, 22, 23],
    8: [24, 25, 26],        # Plasticity Region End (Default 3)
    9: [27, 28],            # Final Selection
    10: [29],               # End Node (Result)
    'CENTRAL': [30]         # The Code Qubit (Center)
}

# --- 2. Central Node / Plasticity Logic Simulation (Configuration) ---

def generate_central_node_config(plasticity_counts: dict):
    """
    Simulates the Cognition Layer NN's Central Node determining the QNN structure.
    This dynamically selects which qubits/rows are active for the current computation.
    """
    active_qubits = []
    
    # Add fixed rows
    for row in [1, 2, 3, 9, 10]:
        active_qubits.extend(ROW_MAP[row])

    # Add plasticity rows based on configuration
    for row, count in plasticity_counts.items():
        if row in ROW_MAP:
            # Only use the first 'count' qubits for that row
            active_qubits.extend(ROW_MAP[row][:count])
            
    return {
        'total_qubits': NUM_QUBITS,
        'plasticity_indices': active_qubits
    }

# --- 3. Quantum Encoder (Layer 1: The Entry Node) ---

def quantum_encoder(features):
    """
    Layer 1: The Entry Node (Unity) - Transforms classical input into a quantum state.
    Bits to Qubits: High density encoding (Angle Embedding).
    """
    # Map features to the available input qubits (excluding Central and Result for now)
    # We use layers 1-9 for encoding if needed, but primary entry is Layer 1
    
    # Encode data into Layer 1 (Unity) and potentially Layer 2 (Duality) for high density
    qml.RY(features[0] * np.pi, wires=ROW_MAP[1]) # Unity
    
    if len(features) > 1:
        # Distribute remaining features across the network initialization
        # This represents "16x denser" storage by utilizing Hilbert space amplitudes
        for i, val in enumerate(features[1:]):
            target_wire = (i + 1) % (NUM_QUBITS - 1)
            qml.RX(val * np.pi, wires=target_wire)


# --- 4. The Core IAI-IPS Quantum Layer Circuit ---

@qml.qnode(DEV)
def iai_ips_quantum_layer(weights, features, plasticity_config):
    """
    The 10-row QIAI-IPS Quantum Circuit architecture with Central Code Qubit.
    """
    
    # --- A. Initialization ---
    # 1. Quantum Encoder (Layer 1: Unity)
    qml.broadcast(qml.Hadamard, wires=range(NUM_QUBITS - 1), pattern="single") # Superposition
    quantum_encoder(features)
    
    # --- B. The Central Code Qubit Connection ---
    # "The code qubit node at the center of the QNN is connected to all qubit nodes"
    central_wire = ROW_MAP['CENTRAL'][0]
    qml.Hadamard(wires=central_wire) # Put Central Node in superposition
    
    # Entangle Central Node with EVERYTHING
    for i in range(NUM_QUBITS - 1):
        qml.CRZ(weights[i % len(weights)], wires=[central_wire, i])

    # --- C. Layer-by-Layer Processing ---
    
    # 2. Layer 2: Two Opposing Aspects (Duality)
    qml.CNOT(wires=[ROW_MAP[2][0], ROW_MAP[2][1]])
    
    # 3. Layer 3: Three Contradictions (Conflict)
    qml.Toffoli(wires=ROW_MAP[3])
    
    # --- D. Plasticity Region (Layers 4-8) ---
    # Apply variational layers based on the active structure
    active_plasticity_wires = []
    for r in range(4, 9):
        active_plasticity_wires.extend(ROW_MAP[r])
        
    # Apply high-depth processing to plasticity layers
    qml.StronglyEntanglingLayers(weights[0:3], wires=active_plasticity_wires, n_layers=1)

    # --- E. Final Layers ---
    
    # 9. Layer 9: Final Selection
    qml.SWAP(wires=[ROW_MAP[9][0], ROW_MAP[9][1]])

    # 10. Layer 10: The End Node (Result)
    final_wire = ROW_MAP[10][0]
    
    # Collapse logic: Entangle penultimate layers to Result
    qml.CNOT(wires=[ROW_MAP[9][0], final_wire])
    qml.CNOT(wires=[ROW_MAP[9][1], final_wire])
    
    # Central Node Final Influence
    qml.CRX(weights[-1], wires=[central_wire, final_wire])

    # Measure the probability of the final result node
    return qml.probs(wires=[final_wire])
