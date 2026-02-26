import pennylane as qml
from pennylane import numpy as np

# --- 1. Architectural Constants ---
NUM_QUBITS = 24 
DEV = qml.device("default.qubit", wires=NUM_QUBITS)

ROW_MAP = {
    1: [0], 2: [1, 2], 3: [3, 4, 5],
    4: [6, 7, 8], 5: [9, 10, 11, 12],
    6: [13, 14, 15, 16], 7: [17, 18, 19],
    8: [20, 21], 9: [22], 10: [23],
    'CENTRAL': [23]
}

def generate_central_node_config(plasticity_counts: dict = None):
    return {'total_qubits': NUM_QUBITS, 'active_indices': list(range(NUM_QUBITS))}

def quantum_encoder(features):
    feat = np.clip(features, -1, 1)
    qml.RY(feat[0] * np.pi, wires=0) 
    if len(feat) > 1:
        for i, val in enumerate(feat[1:]):
            qml.RX(val * np.pi, wires=(i + 1) % NUM_QUBITS)

@qml.qnode(DEV)
def iai_ips_quantum_layer(weights, features, plasticity_config):
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
    
    quantum_encoder(features)
    
    for i in range(NUM_QUBITS - 1):
        qml.CRZ(weights[i % len(weights)], wires=[i, (i + 1) % NUM_QUBITS])

    p_wires = list(range(6, 22))
    num_p = len(p_wires)
    req_weights = num_p * 3 
    
    # Handle the 50-parameter weight file via wrapping
    adj_weights = np.pad(weights, (0, max(0, req_weights - len(weights))), mode='wrap')[:req_weights]
    p_params = adj_weights.reshape(1, num_p, 3)
    
    qml.StronglyEntanglingLayers(p_params, wires=p_wires)
    return qml.probs(wires=[NUM_QUBITS - 1])
