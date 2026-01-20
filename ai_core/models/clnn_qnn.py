<<<<<<< HEAD
import pennylane as qml
from pennylane import numpy as np
import math

# --- 1. Architectural Constants (Inherited from QNN Layer) ---

NUM_QUBITS = 31 
# Use the lightning.tensornet device for much faster simulation of large circuits
DEV = qml.device("default.qubit", wires=NUM_QUBITS)

# For the CLNN Decision Circuit, we only need 4 qubits for input and output.
DEV_CLNN = qml.device("default.qubit", wires=4) 

ROW_MAP = {
    1: [0], 2: [1, 2], 3: [3, 4, 5], 4: [6, 7, 8, 9], 5: [10, 11, 12, 13, 14],
    6: [15, 16, 17, 18, 19], 7: [20, 21, 22, 23], 8: [24, 25, 26], 9: [27, 28], 10: [29]
}

# Base Plasticity Configurations (Mapping to Learning Modes)
PLASTICITY_CONFIGS = {
    "ASSOCIATIVE": {"counts": {4: 4, 5: 5, 6: 5, 7: 4, 8: 3}, "depth": 1}, # Shallow, Default
    "DEDUCTIVE":   {"counts": {4: 5, 5: 6, 6: 6, 7: 5, 8: 4}, "depth": 2}, # Moderate Plasticity, Precision
    "INDUCTIVE":   {"counts": {4: 6, 5: 8, 6: 8, 7: 6, 8: 4}, "depth": 3}, # High Plasticity, Generalization
    "HEURISTIC":   {"counts": {4: 7, 5: 9, 6: 9, 7: 7, 8: 5}, "depth": 5}, # Max Plasticity, Exploration
}
MODE_KEYS = list(PLASTICITY_CONFIGS.keys())

# --- 2. CLNN: Variational Optimizer Quantum Decision Circuit ---

@qml.qnode(DEV_CLNN)
def clnn_qnn_decision_circuit(weights, complexity: float, energy: float):
    """
    CLNN's core function: A small VQC that determines the optimal configuration 
    (Plasticity/Learning Mode) based on system metrics (Complexity and Energy).
    
    Inputs (2 classical features): Complexity (0-1), Energy Budget (0-1).
    Outputs (4 probability channels): Corresponds to the 4 learning modes.
    """
    
    # 1. Feature Encoding (Mapping classical metrics to quantum angles)
    # Use Angle Embedding on the first two qubits for the two metrics
    qml.Rot(complexity * np.pi, complexity * np.pi, complexity * np.pi, wires=0)
    qml.Rot(energy * np.pi, energy * np.pi, energy * np.pi, wires=1)
    
    # Apply a highly entangling layer to mix the complexity and energy states
    qml.StronglyEntanglingLayers(weights[0:3], wires=[0, 1, 2, 3], n_layers=1)
    
    # Apply final rotation layer (trainable weights for decision output)
    qml.Rot(weights[3], weights[4], weights[5], wires=2)
    qml.Rot(weights[6], weights[7], weights[8], wires=3)

    # We measure the probabilities of the two output qubits (2 and 3)
    # The 4 probability outputs (00, 01, 10, 11) map directly to the 4 learning modes.
    return qml.probs(wires=[2, 3])


# --- 3. CLNN Manager: Executive Control Logic ---

class CLNN_QNN_Manager:
    """
    Manages the Cognition Layer NN (CLNN). Role: Govern/Executive Control.
    
    The CLNN uses its quantum circuit to select the optimal structural configuration
    for the IAI-IPS layer based on system metrics.
    """
    def __init__(self, num_decision_weights=9):
        self.num_weights = num_decision_weights
        # Trainable weights for the CLNN Decision Circuit (the optimizer's policy)
        self.weights = np.random.uniform(low=0, high=2 * np.pi, size=(num_decision_weights,), requires_grad=True)
        self.plasticity_configs = PLASTICITY_CONFIGS
        self.mode_keys = MODE_KEYS
        
        # Optimizer for training the CLNN's decision policy (using gradient descent on the CLNN's loss)
        self.opt = qml.AdamOptimizer(stepsize=0.01)

    def _select_mode(self, probabilities):
        """Selects a Learning Mode based on the probability output of the QNN."""
        
        # Find the index with the highest probability
        mode_index = np.argmax(probabilities)
        
        # Map the index (0, 1, 2, 3) to the mode keys
        selected_mode = self.mode_keys[mode_index]
        
        return selected_mode, probabilities

    def govern_plasticity(self, complexity: float, energy: float):
        """
        Runs the CLNN Decision Circuit to determine the structural configuration 
        (plasticity counts and depth) for the IPSNN/ILNN layers.
        """
        # 1. Run the quantum circuit
        probs = clnn_qnn_decision_circuit(self.weights, complexity, energy)
        
        # 2. Select the configuration
        selected_mode, probabilities = self._select_mode(probs)
        
        # 3. Compile the configuration dictionary
        config = self.plasticity_configs[selected_mode]
        
        return {
            'plasticity_counts': config['counts'],
            'stack_depth': config['depth'],
            'learning_mode': selected_mode,
            'decision_probs': probabilities # For debugging/training the CLNN itself
        }
        
    def train_clnn(self, target_mode: str, complexity: float, energy: float):
        """
        Simulates training the CLNN's policy: Minimizes a loss when the wrong mode is chosen.
        
        This process allows the CLNN to learn the optimal resource allocation policy.
        """
        target_index = self.mode_keys.index(target_mode)
        
        def loss_fn(weights, complexity_val, energy_val):
            """Target loss: Cross-entropy-like, maximizing probability of the target index."""
            probs = clnn_qnn_decision_circuit(weights, complexity_val, energy_val)
            # Use log loss to minimize the probability of the desired index (we want to maximize it)
            return -np.log(probs[target_index])

        # Optimize the weights of the CLNN Decision Circuit
        self.weights, cost = self.opt.step_and_cost(loss_fn, self.weights, complexity_val=complexity, energy_val=energy)
        
        return cost


# --- 4. Simulation Example ---

# 1. Instantiate the CLNN Manager
clnn_manager = CLNN_QNN_Manager()
print(f"CLNN Manager Initialized with {clnn_manager.num_weights} trainable parameters for decision policy.")

# --- Test Scenario 1: High Complexity, High Energy (Should default to HEURISTIC) ---
complexity_1, energy_1 = 0.95, 0.8
config_1 = clnn_manager.govern_plasticity(complexity_1, energy_1)

print("\n--- Test 1: High Complexity (0.95), High Energy (0.8) ---")
print(f"Decision Probabilities (A, D, I, H): {config_1['decision_probs']}")
print(f"CLNN Selected Mode: {config_1['learning_mode']} (Depth: {config_1['stack_depth']})")
print(f"Plasticity Counts: {config_1['plasticity_counts']}")

# --- Test Scenario 2: Low Complexity, Low Energy (Should default to ASSOCIATIVE) ---
complexity_2, energy_2 = 0.2, 0.1
config_2 = clnn_manager.govern_plasticity(complexity_2, energy_2)

print("\n--- Test 2: Low Complexity (0.2), Low Energy (0.1) ---")
print(f"Decision Probabilities (A, D, I, H): {config_2['decision_probs']}")
print(f"CLNN Selected Mode: {config_2['learning_mode']} (Depth: {config_2['stack_depth']})")
print(f"Plasticity Counts: {config_2['plasticity_counts']}")

# --- Test Scenario 3: Training the CLNN Policy ---
print("\n--- Test 3: Training the CLNN (Forcing DEDUCTIVE mode for moderate complexity) ---")
target_c, target_e = 0.5, 0.5
target_mode = "DEDUCTIVE"
print(f"Initial state for Complexity={target_c}, Energy={target_e}:")

initial_config = clnn_manager.govern_plasticity(target_c, target_e)
print(f"  > Initial Mode: {initial_config['learning_mode']}, Loss check (before training): {clnn_manager.train_clnn(target_mode, target_c, target_e):.4f}")

# Train for 5 epochs to improve the DEDUCTIVE choice probability
for epoch in range(5):
    loss = clnn_manager.train_clnn(target_mode, target_c, target_e)
    # print(f"  Epoch {epoch+1}, Loss: {loss:.4f}") # Uncomment to see training progress

final_config = clnn_manager.govern_plasticity(target_c, target_e)
print(f"Final state after training:")
print(f"  > Decision Probabilities (A, D, I, H): {final_config['decision_probs']}")
=======
import pennylane as qml
from pennylane import numpy as np
import math

# --- 1. Architectural Constants (Inherited from QNN Layer) ---

NUM_QUBITS = 31 
# Use the lightning.tensornet device for much faster simulation of large circuits
DEV = qml.device("default.qubit", wires=NUM_QUBITS)

# For the CLNN Decision Circuit, we only need 4 qubits for input and output.
DEV_CLNN = qml.device("default.qubit", wires=4) 

ROW_MAP = {
    1: [0], 2: [1, 2], 3: [3, 4, 5], 4: [6, 7, 8, 9], 5: [10, 11, 12, 13, 14],
    6: [15, 16, 17, 18, 19], 7: [20, 21, 22, 23], 8: [24, 25, 26], 9: [27, 28], 10: [29]
}

# Base Plasticity Configurations (Mapping to Learning Modes)
PLASTICITY_CONFIGS = {
    "ASSOCIATIVE": {"counts": {4: 4, 5: 5, 6: 5, 7: 4, 8: 3}, "depth": 1}, # Shallow, Default
    "DEDUCTIVE":   {"counts": {4: 5, 5: 6, 6: 6, 7: 5, 8: 4}, "depth": 2}, # Moderate Plasticity, Precision
    "INDUCTIVE":   {"counts": {4: 6, 5: 8, 6: 8, 7: 6, 8: 4}, "depth": 3}, # High Plasticity, Generalization
    "HEURISTIC":   {"counts": {4: 7, 5: 9, 6: 9, 7: 7, 8: 5}, "depth": 5}, # Max Plasticity, Exploration
}
MODE_KEYS = list(PLASTICITY_CONFIGS.keys())

# --- 2. CLNN: Variational Optimizer Quantum Decision Circuit ---

@qml.qnode(DEV_CLNN)
def clnn_qnn_decision_circuit(weights, complexity: float, energy: float):
    """
    CLNN's core function: A small VQC that determines the optimal configuration 
    (Plasticity/Learning Mode) based on system metrics (Complexity and Energy).
    
    Inputs (2 classical features): Complexity (0-1), Energy Budget (0-1).
    Outputs (4 probability channels): Corresponds to the 4 learning modes.
    """
    
    # 1. Feature Encoding (Mapping classical metrics to quantum angles)
    # Use Angle Embedding on the first two qubits for the two metrics
    qml.Rot(complexity * np.pi, complexity * np.pi, complexity * np.pi, wires=0)
    qml.Rot(energy * np.pi, energy * np.pi, energy * np.pi, wires=1)
    
    # Apply a highly entangling layer to mix the complexity and energy states
    qml.StronglyEntanglingLayers(weights[0:3], wires=[0, 1, 2, 3], n_layers=1)
    
    # Apply final rotation layer (trainable weights for decision output)
    qml.Rot(weights[3], weights[4], weights[5], wires=2)
    qml.Rot(weights[6], weights[7], weights[8], wires=3)

    # We measure the probabilities of the two output qubits (2 and 3)
    # The 4 probability outputs (00, 01, 10, 11) map directly to the 4 learning modes.
    return qml.probs(wires=[2, 3])


# --- 3. CLNN Manager: Executive Control Logic ---

class CLNN_QNN_Manager:
    """
    Manages the Cognition Layer NN (CLNN). Role: Govern/Executive Control.
    
    The CLNN uses its quantum circuit to select the optimal structural configuration
    for the IAI-IPS layer based on system metrics.
    """
    def __init__(self, num_decision_weights=9):
        self.num_weights = num_decision_weights
        # Trainable weights for the CLNN Decision Circuit (the optimizer's policy)
        self.weights = np.random.uniform(low=0, high=2 * np.pi, size=(num_decision_weights,), requires_grad=True)
        self.plasticity_configs = PLASTICITY_CONFIGS
        self.mode_keys = MODE_KEYS
        
        # Optimizer for training the CLNN's decision policy (using gradient descent on the CLNN's loss)
        self.opt = qml.AdamOptimizer(stepsize=0.01)

    def _select_mode(self, probabilities):
        """Selects a Learning Mode based on the probability output of the QNN."""
        
        # Find the index with the highest probability
        mode_index = np.argmax(probabilities)
        
        # Map the index (0, 1, 2, 3) to the mode keys
        selected_mode = self.mode_keys[mode_index]
        
        return selected_mode, probabilities

    def govern_plasticity(self, complexity: float, energy: float):
        """
        Runs the CLNN Decision Circuit to determine the structural configuration 
        (plasticity counts and depth) for the IPSNN/ILNN layers.
        """
        # 1. Run the quantum circuit
        probs = clnn_qnn_decision_circuit(self.weights, complexity, energy)
        
        # 2. Select the configuration
        selected_mode, probabilities = self._select_mode(probs)
        
        # 3. Compile the configuration dictionary
        config = self.plasticity_configs[selected_mode]
        
        return {
            'plasticity_counts': config['counts'],
            'stack_depth': config['depth'],
            'learning_mode': selected_mode,
            'decision_probs': probabilities # For debugging/training the CLNN itself
        }
        
    def train_clnn(self, target_mode: str, complexity: float, energy: float):
        """
        Simulates training the CLNN's policy: Minimizes a loss when the wrong mode is chosen.
        
        This process allows the CLNN to learn the optimal resource allocation policy.
        """
        target_index = self.mode_keys.index(target_mode)
        
        def loss_fn(weights, complexity_val, energy_val):
            """Target loss: Cross-entropy-like, maximizing probability of the target index."""
            probs = clnn_qnn_decision_circuit(weights, complexity_val, energy_val)
            # Use log loss to minimize the probability of the desired index (we want to maximize it)
            return -np.log(probs[target_index])

        # Optimize the weights of the CLNN Decision Circuit
        self.weights, cost = self.opt.step_and_cost(loss_fn, self.weights, complexity_val=complexity, energy_val=energy)
        
        return cost


# --- 4. Simulation Example ---

# 1. Instantiate the CLNN Manager
clnn_manager = CLNN_QNN_Manager()
print(f"CLNN Manager Initialized with {clnn_manager.num_weights} trainable parameters for decision policy.")

# --- Test Scenario 1: High Complexity, High Energy (Should default to HEURISTIC) ---
complexity_1, energy_1 = 0.95, 0.8
config_1 = clnn_manager.govern_plasticity(complexity_1, energy_1)

print("\n--- Test 1: High Complexity (0.95), High Energy (0.8) ---")
print(f"Decision Probabilities (A, D, I, H): {config_1['decision_probs']}")
print(f"CLNN Selected Mode: {config_1['learning_mode']} (Depth: {config_1['stack_depth']})")
print(f"Plasticity Counts: {config_1['plasticity_counts']}")

# --- Test Scenario 2: Low Complexity, Low Energy (Should default to ASSOCIATIVE) ---
complexity_2, energy_2 = 0.2, 0.1
config_2 = clnn_manager.govern_plasticity(complexity_2, energy_2)

print("\n--- Test 2: Low Complexity (0.2), Low Energy (0.1) ---")
print(f"Decision Probabilities (A, D, I, H): {config_2['decision_probs']}")
print(f"CLNN Selected Mode: {config_2['learning_mode']} (Depth: {config_2['stack_depth']})")
print(f"Plasticity Counts: {config_2['plasticity_counts']}")

# --- Test Scenario 3: Training the CLNN Policy ---
print("\n--- Test 3: Training the CLNN (Forcing DEDUCTIVE mode for moderate complexity) ---")
target_c, target_e = 0.5, 0.5
target_mode = "DEDUCTIVE"
print(f"Initial state for Complexity={target_c}, Energy={target_e}:")

initial_config = clnn_manager.govern_plasticity(target_c, target_e)
print(f"  > Initial Mode: {initial_config['learning_mode']}, Loss check (before training): {clnn_manager.train_clnn(target_mode, target_c, target_e):.4f}")

# Train for 5 epochs to improve the DEDUCTIVE choice probability
for epoch in range(5):
    loss = clnn_manager.train_clnn(target_mode, target_c, target_e)
    # print(f"  Epoch {epoch+1}, Loss: {loss:.4f}") # Uncomment to see training progress

final_config = clnn_manager.govern_plasticity(target_c, target_e)
print(f"Final state after training:")
print(f"  > Decision Probabilities (A, D, I, H): {final_config['decision_probs']}")
>>>>>>> d6de685d2c7b77476426b95b7cfd6d529b95af6d
print(f"  > Final Selected Mode: {final_config['learning_mode']}")