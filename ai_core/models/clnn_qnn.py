import pennylane as qml
from pennylane import numpy as np
import math
from .qnn_layer import iai_ips_quantum_layer, generate_central_node_config, NUM_QUBITS

# Base Plasticity Configurations (Mapping to Learning Modes)
PLASTICITY_CONFIGS = {
    "ASSOCIATIVE": {"counts": {4: 4, 5: 5, 6: 5, 7: 4, 8: 3}, "depth": 1},
    "DEDUCTIVE":   {"counts": {4: 5, 5: 6, 6: 6, 7: 5, 8: 4}, "depth": 2},
    "INDUCTIVE":   {"counts": {4: 6, 5: 8, 6: 8, 7: 6, 8: 4}, "depth": 3},
    "HEURISTIC":   {"counts": {4: 7, 5: 9, 6: 9, 7: 7, 8: 5}, "depth": 5},
}
MODE_KEYS = list(PLASTICITY_CONFIGS.keys())

class CLNN_QNN_Manager:
    """
    Manages the Cognitive Quantum Neural Network (Cognitive-QNN).
    This is Stack #3: The Administrator / Governance Layer.
    
    It uses the QIAI-IPS architecture to make high-level governance decisions
    (selecting learning modes and plasticity depths).
    """
    def __init__(self, num_decision_weights=50):
        self.num_weights = num_decision_weights
        self.weights = np.random.uniform(low=0, high=2 * np.pi, size=(num_decision_weights,), requires_grad=True)
        self.plasticity_configs = PLASTICITY_CONFIGS
        self.mode_keys = MODE_KEYS
        
        # Self-optimization
        self.opt = qml.AdamOptimizer(stepsize=0.01)
        print(f"[CLNN-QNN] Cognitive Layer Active: {NUM_QUBITS} Qubits | Governance Mode")

    def _select_mode(self, probability_output):
        """Selects a Learning Mode based on the quantum probability distribution."""
        # Map the continuous probability (0-1) to one of the 4 modes
        mode_index = int(probability_output * len(self.mode_keys))
        mode_index = min(mode_index, len(self.mode_keys) - 1)
        
        selected_mode = self.mode_keys[mode_index]
        return selected_mode

    def govern_plasticity(self, complexity: float, energy: float):
        """
        Runs the Cognitive QNN to determine the structural configuration 
        for the other stacks.
        """
        # Encode governance metrics (C, E) into the QNN
        features = np.array([complexity, energy, 0.0, 0.0]) # Pad for minimal feature set
        
        # Use a balanced plasticity for the governance decision itself
        qnn_config = generate_central_node_config(self.plasticity_configs["DEDUCTIVE"]["counts"])
        
        # Execute
        probs = iai_ips_quantum_layer(self.weights, features, qnn_config)
        decision_val = probs[1] # Probability of |1>
        
        # Select Mode based on quantum collapse
        selected_mode = self._select_mode(decision_val)
        config = self.plasticity_configs[selected_mode]
        
        return {
            'plasticity_counts': config['counts'],
            'stack_depth': config['depth'],
            'learning_mode': selected_mode,
            'decision_value': decision_val
        }
