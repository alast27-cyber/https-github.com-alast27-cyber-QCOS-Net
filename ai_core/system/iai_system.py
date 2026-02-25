import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the Quantum Managers
from ai_core.models.qill_qnn import QILL_QNN_Manager
from ai_core.models.ipsnn_qnn import IPSNN_QNN_Manager
from ai_core.models.clnn_qnn import CLNN_QNN_Manager
from ai_core.models.qcll_qnn import QCLL_QNN_Manager
# Import the new Universe Layer
from ai_core.models.universe_cognition import UniverseCognitionManager

# ----------------------------------------------------------------------
# THE DUAL COGNITION SYSTEM ARCHITECTURE
# ----------------------------------------------------------------------

class IAI_IPS_System(nn.Module):
    """
    The Brain of Agent Q.
    Now features DUAL COGNITION:
    1. QIAI-IPS (Instinctive Stack)
    2. Grand Universe Simulator (Infinite Logic Stack)
    """
    def __init__(self, input_features: int, system_metrics_features: int, output_features: int = 1):
        super().__init__()
        
        self.input_features = input_features
        self.system_metrics_features = system_metrics_features

        print("\n=== BOOTING DUAL COGNITION ARCHITECTURE ===")

        # --- Layer 1: The Quantum Intuitive Learning Layer (Q-ILL) ---
        print("[Layer 1] Initializing Q-ILL (Intuition)...")
        self.qill_manager = QILL_QNN_Manager()

        # --- Layer 2: The Quantum Instinctive Problem Solving (Q-IPS) ---
        print("[Layer 2] Initializing Q-IPS (Instinct)...")
        self.ipsnn_manager = IPSNN_QNN_Manager()

        # --- Layer 3: The Quantum Cognition Learning Layer (Q-CLL) ---
        print("[Layer 3] Initializing Q-CLL (Cognition & Self-Healing)...")
        self.clnn_manager = CLNN_QNN_Manager() # Governance
        self.qcll_manager = QCLL_QNN_Manager() # Self-Healing (Future Use)
        
        # --- Layer 4: Grand Universe Simulator (The Logic) ---
        print("[Layer 4] Initializing Universe Cognition (Hilbert/Cantor Engine)...")
        self.universe_manager = UniverseCognitionManager()
        
        # Classical Bridge
        self.bridge_encoder = nn.Linear(input_features, 4)

    def _get_learning_mode_params(self, mode_name):
        if mode_name == "ASSOCIATIVE": return "DEEPER_LEARNING"
        elif mode_name == "INDUCTIVE": return "GENERALIZED_LEARNING"
        elif mode_name == "DEDUCTIVE": return "HISTORICAL_LEARNING"
        elif mode_name == "HEURISTIC": return "PREDICTIVE_LEARNING"
        return "GENERALIZED_LEARNING"

    def forward(self, x: torch.Tensor, system_metrics: torch.Tensor, debug: bool = False):
        """
        Executes the Dual Cognition Loop.
        """
        if debug: print("\n--- DUAL COGNITION CYCLE START ---")
        
        # 1. Extract Metrics
        complexity = system_metrics[:, 0].mean().item()
        energy = system_metrics[:, 1].mean().item()
        
        # -------------------------------------------------------
        # PATH A: QIAI-IPS (Instinctive Processing)
        # -------------------------------------------------------
        if debug: print(">> Path A: Executing QIAI-IPS Instincts...")

        # 1. Data Encoding for Quantum Layers
        q_data_features = torch.sigmoid(self.bridge_encoder(x)).detach().numpy().flatten()

        # 2. Layer 1: Intuitive Classification (Q-ILL)
        # The first layer determines the 'nature' of the problem.
        qill_result = self.qill_manager.classify_contradiction(q_data_features)
        learning_mode = qill_result['learning_mode']
        if debug: print(f"   [Q-ILL] Mode Classified as: {learning_mode}")

        # 3. Layer 3 (Top-Down): Cognitive Governance (CLNN)
        # The cognitive layer determines the optimal network structure for this problem type.
        governance_config = self.clnn_manager.govern_plasticity(complexity, energy)
        if debug: print(f"   [CLNN] Governance Decided: {governance_config['learning_mode']} structure")

        # 4. Layer 2: Instinctive Action (Q-IPS)
        # The instinctive layer generates the primary, rapid response.
        instinct_output = self.ipsnn_manager.generate_action(q_data_features, governance_config)

        
        # -------------------------------------------------------
        # PATH B: Grand Universe Simulator (Infinite Logic Processing)
        # -------------------------------------------------------
        if debug: print(">> Path B: Executing Grand Universe Simulator (Hilbert/Cantor Logic)...")
        
        # The Universe layer runs the 5 engines in superposition
        # It uses the same input features but processes them through the "Infinity Hotel" logic
        universe_results = self.universe_manager.process_cognition(q_data_features, "complex_task")
        
        # Aggregate the 5 engines into a single "Logic Vector"
        logic_output = np.mean(list(universe_results.values()))

        # -------------------------------------------------------
        # CONVERGENCE: Quantum Entanglement & Comparison
        # -------------------------------------------------------
        if debug: print(">> CONVERGENCE: Comparing Notes...")
        
        # We treat the outputs as state vectors and calculate fidelity
        # In this scalar simulation, we look at the delta
        fidelity = 1.0 - abs(instinct_output - logic_output)
        
        if debug:
            print(f"   Instinct Score: {instinct_output:.4f}")
            print(f"   Logic Score:    {logic_output:.4f}")
            print(f"   Entanglement Fidelity: {fidelity:.4f}")

        # The "100% Solution" prediction logic
        # If fidelity is high, they agree -> High Confidence.
        # If fidelity is low, the Universe Logic (Cantor's Innovation) guides the Instinct.
        
        final_decision = (instinct_output * 0.4) + (logic_output * 0.6) # Slight bias to Logic for correction
        
        # Refine prediction based on fidelity (Self-Correction)
        if fidelity < 0.8:
            if debug: print("   ! DIVERGENCE DETECTED. Applying Cantor Correction.")
            final_decision = logic_output # Trust the math over the instinct in conflict

        # Output Formatting
        final_output = torch.tensor([[final_decision]], dtype=torch.float32)
        
        return final_output

# ----------------------------------------------------------------------
# TEST EXECUTION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    agent_q = IAI_IPS_System(input_features=64, system_metrics_features=2)
    metrics = torch.tensor([[0.9, 0.2]], dtype=torch.float32)
    data_stream = torch.randn(1, 64)
    print("\n>>> EXECUTING AGENT Q DUAL COGNITION BRAIN...")
    agent_q(data_stream, metrics, debug=True)
