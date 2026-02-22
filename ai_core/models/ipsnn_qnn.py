import pennylane as qml
from pennylane import numpy as np
import os
from .qnn_layer import iai_ips_quantum_layer, generate_central_node_config, NUM_QUBITS

class IPSNN_QNN_Manager:
    """
    Manages the Instinctive Problem Solving Quantum Neural Network (IPS-QNN).
    This is Stack #2: The Action Generator.
    """
    def __init__(self, num_weights=50):
        self.num_weights = num_weights
        
        # Load or Initialize Weights
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "trained_weights.npy")
        
        if os.path.exists(weights_path):
            print(f"[IPS-QNN] Loading evolved synaptic weights from {weights_path}")
            loaded_weights = np.load(weights_path)
            # Resize if architecture changed
            if loaded_weights.size != num_weights:
                new_weights = np.random.uniform(0, 2*np.pi, size=(num_weights,))
                min_len = min(len(loaded_weights), num_weights)
                new_weights[:min_len] = loaded_weights[:min_len]
                self.current_weights = new_weights
            else:
                self.current_weights = loaded_weights
        else:
            print("[IPS-QNN] Initializing fresh Quantum Neural Network weights.")
            self.current_weights = np.random.uniform(low=0, high=2 * np.pi, size=(num_weights,), requires_grad=True)
            
        print(f"[IPS-QNN] QNN Core Active: {NUM_QUBITS} Qubits | 10 Layers | Central Code Node Online")

    def generate_action(self, features, plasticity_config):
        """
        Synthesizes an 'Instinctive' solution using the QNN.
        """
        # Feature preprocessing
        if len(features.shape) > 1:
            flat_features = features.flatten()
        else:
            flat_features = features
            
        # Determine configuration from CLNN input
        # Default counts if not provided
        counts = plasticity_config.get('plasticity_counts', {4: 4, 5: 5, 6: 5, 7: 4, 8: 3})
        qnn_config = generate_central_node_config(counts)
        
        # Execute QIAI-IPS Quantum Circuit
        # We pass the full weights tensor; the circuit uses what it needs
        probs = iai_ips_quantum_layer(self.current_weights, flat_features, qnn_config)
        
        # Return the expectation value (or probability of |1>)
        return probs[1] 

    def save_ikm(self, key, weights):
        """Saves an Instinctive Kernel Module (IKM) snapshot."""
        # Simulation of saving optimized weights to a high-speed lookup
        pass

    def load_ikm(self, key):
        """Hot-swaps weights from a saved IKM."""
        return True
