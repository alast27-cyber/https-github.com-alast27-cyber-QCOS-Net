import pennylane as qml
from pennylane import numpy as np
from .qnn_layer import iai_ips_quantum_layer, generate_central_node_config, NUM_QUBITS

# The four learning modes as defined in the Q-IAI architecture
LEARNING_MODES = ["ONION_PEELING", "DIALECTICS", "HISTORICAL", "PREDICTIVE"]

class QILL_QNN_Manager:
    """
    Manages the Quantum Intuitive Learning Layer (Q-ILL).
    This is Layer 1: The Entry Point.

    It translates raw input features into one of the four primary learning modes,
    setting the context for the entire cognitive process.
    """
    def __init__(self, num_weights=50):
        self.num_weights = num_weights
        self.weights = np.random.uniform(low=0, high=2 * np.pi, size=(num_weights,), requires_grad=True)
        self.learning_modes = LEARNING_MODES
        print(f"[QILL-QNN] Intuitive Layer Active: {NUM_QUBITS} Qubits | Mode Classifier")

    def _select_learning_mode(self, probability_output):
        """Selects a Learning Mode based on the quantum probability distribution."""
        mode_index = int(probability_output * len(self.learning_modes))
        mode_index = min(mode_index, len(self.learning_modes) - 1)
        return self.learning_modes[mode_index]

    def classify_contradiction(self, features):
        """
        Runs the Intuitive QNN to classify the input and determine the learning mode.
        """
        # For the initial classification, we use a default, balanced plasticity configuration.
        # This can be thought of as the "default state" of intuition.
        default_plasticity = {4: 5, 5: 6, 6: 6, 7: 5, 8: 4}
        qnn_config = generate_central_node_config(default_plasticity)

        # Execute the quantum circuit
        probs = iai_ips_quantum_layer(self.weights, features, qnn_config)
        decision_val = probs[1]  # Probability of measuring |1>

        # Select the learning mode based on the measurement outcome
        selected_mode = self._select_learning_mode(decision_val)

        return {
            'learning_mode': selected_mode,
            'confidence': decision_val
        }
