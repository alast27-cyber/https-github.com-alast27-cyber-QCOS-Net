<<<<<<< HEAD
import pennylane as qml
from pennylane import numpy as np

# --- 1. Architectural Constants ---
# REDUCED TO 10 QUBITS FOR LOCAL SIMULATION
NUM_QUBITS = 10 
DEV = qml.device("default.qubit", wires=NUM_QUBITS)

# Updated Mapping for 10 Qubits (Used for conceptual architecture, not strictly indexed)
ROW_MAP = {
    1: [0], 
    2: [1], 
    3: [2, 3], 
    4: [4, 5], 
    5: [6, 7],
    6: [8], 
    7: [9], 
    8: [0], 
    9: [1, 2], 
    10: [3]
}

def generate_central_node_config(plasticity_counts: dict):
    """Generates the configuration dict for the QNN based on CLNN governance."""
    return {
        'total_qubits': NUM_QUBITS, 
        'plasticity_indices': list(range(NUM_QUBITS)),
        'active_weights': 26
    }

def quantum_encoder(features):
    """
    Encodes classical features (C, E) into the quantum state.
    Expects features shape: (NUM_QUBITS, 3)
    """
    limit = min(len(features), NUM_QUBITS)
    for i in range(limit): 
        # Features is shape (N, 3), use valid wires
        qml.Rot(features[i][0], features[i][1], features[i][2], wires=i)

@qml.qnode(DEV)
def iai_ips_quantum_layer(weights, features, plasticity_config):
    """The core IAI-IPS Quantum Circuit (10-Qubit Version)."""
    
    # 1. ENCODING
    for wire in range(NUM_QUBITS):
        qml.Hadamard(wires=wire)
        
    quantum_encoder(features)
    
    # 2. PROCESSING
    # Layer 2: Duality
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    
    # Layer 3: Conflict
    w_layer3 = weights[2:5].reshape(1, 3)
    qml.BasicEntanglerLayers(weights=w_layer3, wires=[2, 3, 4])
    
    # Layers 4-8: Plasticity (Instinct)
    instinct_wires = [4, 5, 6, 7, 8, 9]
    
    idx = 5
    for wire in instinct_wires:
        w_val = weights[idx % 20] 
        qml.RY(w_val, wires=wire)
        idx += 1
        
    # Ring Entanglement on instinct wires
    for i in range(len(instinct_wires) - 1):
        qml.CNOT(wires=[instinct_wires[i], instinct_wires[i+1]])

    # 3. DECISION
    final_wire = 0
    
    # Entangle instinct result back to decision wire
    qml.CNOT(wires=[instinct_wires[-1], final_wire])
    
    # Return Expectation Value V
    return qml.expval(qml.PauliZ(final_wire))


class IPSNN_QNN_Manager:
    """Manages the Quantum Layer for the Instinctive Problem Solving Network."""
    def __init__(self, num_weights=26):
        self.num_weights = num_weights
        
        # Try to load trained weights
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "trained_weights.npy")
        
        if os.path.exists(weights_path):
            print(f"[IPSNN] Loading TRAINED weights from {weights_path}")
            # Ensure weights are loaded as numpy array
            loaded_weights = np.load(weights_path)
            if loaded_weights.ndim == 0: # Handle scalar if mock training created one
                loaded_weights = np.array([loaded_weights.item()])
            
            # Ensure correct size (26)
            if loaded_weights.size < num_weights:
                 # Pad or resize if necessary, assume we only need the number requested
                 loaded_weights = np.resize(loaded_weights, num_weights)
                 
            self.current_weights = loaded_weights[:num_weights]
            # Add small noise to simulate 'live' quantum state
            self.current_weights = self.current_weights + np.random.normal(0, 0.01, size=num_weights)
        else:
            print("[IPSNN] No trained weights found. Initializing RANDOM weights.")
            # Ensure requires_grad=True is used for new weights
            self.current_weights = np.random.uniform(low=0, high=2 * np.pi, size=(num_weights,), requires_grad=True)
            
        print(f"IPSNN QNN Initialized ({NUM_QUBITS} Qubits).")

    def generate_action(self, features, plasticity_config):
        """
        Runs the QNN to generate a single scalar action/solution (V-Score).
        """
        # Ensure input features are compatible with the 10-qubit circuit
        if features.shape[0] > NUM_QUBITS:
             features = features[:NUM_QUBITS]
             
        # Execute the QNode
        v_score = iai_ips_quantum_layer(self.current_weights, features, plasticity_config)
=======
import pennylane as qml
from pennylane import numpy as np

# --- 1. Architectural Constants ---
# REDUCED TO 10 QUBITS FOR LOCAL SIMULATION
NUM_QUBITS = 10 
DEV = qml.device("default.qubit", wires=NUM_QUBITS)

# Updated Mapping for 10 Qubits (Used for conceptual architecture, not strictly indexed)
ROW_MAP = {
    1: [0], 
    2: [1], 
    3: [2, 3], 
    4: [4, 5], 
    5: [6, 7],
    6: [8], 
    7: [9], 
    8: [0], 
    9: [1, 2], 
    10: [3]
}

def generate_central_node_config(plasticity_counts: dict):
    """Generates the configuration dict for the QNN based on CLNN governance."""
    return {
        'total_qubits': NUM_QUBITS, 
        'plasticity_indices': list(range(NUM_QUBITS)),
        'active_weights': 26
    }

def quantum_encoder(features):
    """
    Encodes classical features (C, E) into the quantum state.
    Expects features shape: (NUM_QUBITS, 3)
    """
    limit = min(len(features), NUM_QUBITS)
    for i in range(limit): 
        # Features is shape (N, 3), use valid wires
        qml.Rot(features[i][0], features[i][1], features[i][2], wires=i)

@qml.qnode(DEV)
def iai_ips_quantum_layer(weights, features, plasticity_config):
    """The core IAI-IPS Quantum Circuit (10-Qubit Version)."""
    
    # 1. ENCODING
    for wire in range(NUM_QUBITS):
        qml.Hadamard(wires=wire)
        
    quantum_encoder(features)
    
    # 2. PROCESSING
    # Layer 2: Duality
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[0], wires=0)
    qml.RZ(weights[1], wires=1)
    
    # Layer 3: Conflict
    w_layer3 = weights[2:5].reshape(1, 3)
    qml.BasicEntanglerLayers(weights=w_layer3, wires=[2, 3, 4])
    
    # Layers 4-8: Plasticity (Instinct)
    instinct_wires = [4, 5, 6, 7, 8, 9]
    
    idx = 5
    for wire in instinct_wires:
        w_val = weights[idx % 20] 
        qml.RY(w_val, wires=wire)
        idx += 1
        
    # Ring Entanglement on instinct wires
    for i in range(len(instinct_wires) - 1):
        qml.CNOT(wires=[instinct_wires[i], instinct_wires[i+1]])

    # 3. DECISION
    final_wire = 0
    
    # Entangle instinct result back to decision wire
    qml.CNOT(wires=[instinct_wires[-1], final_wire])
    
    # Return Expectation Value V
    return qml.expval(qml.PauliZ(final_wire))


class IPSNN_QNN_Manager:
    """Manages the Quantum Layer for the Instinctive Problem Solving Network."""
    def __init__(self, num_weights=26):
        self.num_weights = num_weights
        
        # Try to load trained weights
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "trained_weights.npy")
        
        if os.path.exists(weights_path):
            print(f"[IPSNN] Loading TRAINED weights from {weights_path}")
            # Ensure weights are loaded as numpy array
            loaded_weights = np.load(weights_path)
            if loaded_weights.ndim == 0: # Handle scalar if mock training created one
                loaded_weights = np.array([loaded_weights.item()])
            
            # Ensure correct size (26)
            if loaded_weights.size < num_weights:
                 # Pad or resize if necessary, assume we only need the number requested
                 loaded_weights = np.resize(loaded_weights, num_weights)
                 
            self.current_weights = loaded_weights[:num_weights]
            # Add small noise to simulate 'live' quantum state
            self.current_weights = self.current_weights + np.random.normal(0, 0.01, size=num_weights)
        else:
            print("[IPSNN] No trained weights found. Initializing RANDOM weights.")
            # Ensure requires_grad=True is used for new weights
            self.current_weights = np.random.uniform(low=0, high=2 * np.pi, size=(num_weights,), requires_grad=True)
            
        print(f"IPSNN QNN Initialized ({NUM_QUBITS} Qubits).")

    def generate_action(self, features, plasticity_config):
        """
        Runs the QNN to generate a single scalar action/solution (V-Score).
        """
        # Ensure input features are compatible with the 10-qubit circuit
        if features.shape[0] > NUM_QUBITS:
             features = features[:NUM_QUBITS]
             
        # Execute the QNode
        v_score = iai_ips_quantum_layer(self.current_weights, features, plasticity_config)
>>>>>>> d6de685d2c7b77476426b95b7cfd6d529b95af6d
        return v_score