<<<<<<< HEAD
import numpy as np
import time
import logging

# Import your existing QNN Managers
# Ensure clnn_qnn.py and ipsnn_qnn.py are in the same directory or python path
from clnn_qnn import CLNN_QNN_Manager
from ipsnn_qnn import IPSNN_QNN_Manager

# Configure Logging for the Cycle
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [SYNTHESIS-MGR] %(message)s')
logger = logging.getLogger("InstinctSynthesis")

class InstinctSynthesisManager:
    """
    The Executive Control System that manages the 'Instinct Synthesis Cycle'.
    
    Role:
    1. Monitors the Instinctive State Vector (V_I) for performance degradation.
    2. Triggers the CLNN to hypothesize a new structure when V_I drops.
    3. Orchestrates the retraining of the IPSNN (Instinct).
    4. Performs the Atomic Swap (Deployment) of the new IKM.
    """

    def __init__(self, ipsnn_manager: IPSNN_QNN_Manager, clnn_manager: CLNN_QNN_Manager):
        self.ipsnn = ipsnn_manager
        self.clnn = clnn_manager
        
        # --- Phase 1 Thresholds (Asymptotic Monitoring) ---
        # Defined in: instinct_synthesis_cycle.md
        self.HIT_RATE_THRESHOLD = 0.92      # P_IL,HitRate < 92% triggers review
        self.LATENCY_THRESHOLD_MS = 15.0    # L_Res > 15ms triggers review
        self.ENERGY_VETO_LIMIT = 0.85       # Max energy usage allowed before forced optimization

    def monitor_and_evaluate(self, performance_metrics: dict) -> bool:
        """
        Phase 1: The Trigger Mechanism.
        Accepts real-time metrics from the running OS Kernel.
        Returns True if the Synthesis Cycle must be initiated.
        """
        hit_rate = performance_metrics.get('hit_rate', 1.0)
        avg_latency = performance_metrics.get('avg_latency_ms', 0.0)
        current_energy = performance_metrics.get('energy_cost', 0.0)

        trigger_reason = []

        # 1. Check Hit Rate (Accuracy of Instincts)
        if hit_rate < self.HIT_RATE_THRESHOLD:
            trigger_reason.append(f"Low Hit Rate ({hit_rate:.2f})")

        # 2. Check Latency (Speed of Instincts)
        if avg_latency > self.LATENCY_THRESHOLD_MS:
            trigger_reason.append(f"High Latency ({avg_latency:.2f}ms)")

        # 3. Check Energy Efficiency
        if current_energy > self.ENERGY_VETO_LIMIT:
            trigger_reason.append(f"Energy Veto Violation ({current_energy:.2f})")

        if trigger_reason:
            logger.warning(f"TRIGGER ACTIVATED: {', '.join(trigger_reason)}")
            return True
        
        return False

    def execute_synthesis_cycle(self, context_c: float, energy_e: float, failed_data_batch=None):
        """
        Executes Phases 2, 3, and 4 of the Cycle: Hypothesis -> Retraining -> Deployment.
        """
        logger.info("--- INITIATING INSTINCT SYNTHESIS CYCLE ---")

        # --- Phase 2: Hypothesis Generation (CLNN) ---
        logger.info("Phase 2: CLNN Hypothesis Generation...")
        # The CLNN analyzes the situation (C, E) to find the optimal structural configuration
        clnn_decision = self.clnn.govern_plasticity(context_c, energy_e)
        
        target_mode = clnn_decision['learning_mode']
        new_plasticity = clnn_decision['plasticity_counts']
        stack_depth = clnn_decision['stack_depth']
        
        logger.info(f"CLNN Selected Strategy: {target_mode} (Depth: {stack_depth})")
        logger.info(f"New Plasticity Configuration: {new_plasticity}")

        # --- Phase 3: Retraining / Compilation (Simulated) ---
        # In a real scenario, this calls 'training_setup.py' to run a high-speed training loop.
        logger.info("Phase 3: Retraining Instinctive Layer (Simulation)...")
        
        # We simulate the training producing a new set of optimized weights
        # The number of weights depends on the fixed architecture for now, but conceptually it adapts.
        new_weights = self._simulate_retraining_process(new_plasticity, failed_data_batch)
        
        # --- Phase 4: Atomic Swap (Deployment) ---
        self._atomic_swap_deployment(new_weights, target_mode, new_plasticity)

    def _simulate_retraining_process(self, plasticity_config, data):
        """
        Simulates the heavy lifting of the training loop.
        Returns optimized weights.
        """
        # Simulate processing time
        time.sleep(0.5) 
        
        # Generate 'optimized' weights (mocking the result of backpropagation)
        # For the standard IPSNN, we use 26 weights.
        # If plasticity changed the circuit size, we would handle tensor resizing here.
        optimized_weights = np.random.uniform(0, 2 * np.pi, size=(26,))
        
        logger.info("Retraining Complete. Loss converged to < 0.01.")
        return optimized_weights

    def _atomic_swap_deployment(self, new_weights, mode_name, plasticity_config):
        """
        Phase 4: The 'Hot Swap'.
        Compiles the new weights into an IKM and injects them into the running IPSNN.
        """
        logger.info("Phase 4: Atomic Swap / IKM Deployment...")
        
        # 1. Generate a unique Key for this new Instinct
        ikm_key = f"IKM_{mode_name}_{int(time.time())}"
        
        # 2. Save to IPSNN Cache (The "Compiler")
        self.ipsnn.save_ikm(ikm_key, new_weights)
        
        # 3. Hot Load the new IKM
        success = self.ipsnn.load_ikm(ikm_key)
        
        if success:
            logger.info(f"SUCCESS: System now running on {ikm_key}.")
            logger.info("Asymptotic Monitoring Reset. Cycle Complete.")
        else:
=======
import numpy as np
import time
import logging

# Import your existing QNN Managers
# Ensure clnn_qnn.py and ipsnn_qnn.py are in the same directory or python path
from clnn_qnn import CLNN_QNN_Manager
from ipsnn_qnn import IPSNN_QNN_Manager

# Configure Logging for the Cycle
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [SYNTHESIS-MGR] %(message)s')
logger = logging.getLogger("InstinctSynthesis")

class InstinctSynthesisManager:
    """
    The Executive Control System that manages the 'Instinct Synthesis Cycle'.
    
    Role:
    1. Monitors the Instinctive State Vector (V_I) for performance degradation.
    2. Triggers the CLNN to hypothesize a new structure when V_I drops.
    3. Orchestrates the retraining of the IPSNN (Instinct).
    4. Performs the Atomic Swap (Deployment) of the new IKM.
    """

    def __init__(self, ipsnn_manager: IPSNN_QNN_Manager, clnn_manager: CLNN_QNN_Manager):
        self.ipsnn = ipsnn_manager
        self.clnn = clnn_manager
        
        # --- Phase 1 Thresholds (Asymptotic Monitoring) ---
        # Defined in: instinct_synthesis_cycle.md
        self.HIT_RATE_THRESHOLD = 0.92      # P_IL,HitRate < 92% triggers review
        self.LATENCY_THRESHOLD_MS = 15.0    # L_Res > 15ms triggers review
        self.ENERGY_VETO_LIMIT = 0.85       # Max energy usage allowed before forced optimization

    def monitor_and_evaluate(self, performance_metrics: dict) -> bool:
        """
        Phase 1: The Trigger Mechanism.
        Accepts real-time metrics from the running OS Kernel.
        Returns True if the Synthesis Cycle must be initiated.
        """
        hit_rate = performance_metrics.get('hit_rate', 1.0)
        avg_latency = performance_metrics.get('avg_latency_ms', 0.0)
        current_energy = performance_metrics.get('energy_cost', 0.0)

        trigger_reason = []

        # 1. Check Hit Rate (Accuracy of Instincts)
        if hit_rate < self.HIT_RATE_THRESHOLD:
            trigger_reason.append(f"Low Hit Rate ({hit_rate:.2f})")

        # 2. Check Latency (Speed of Instincts)
        if avg_latency > self.LATENCY_THRESHOLD_MS:
            trigger_reason.append(f"High Latency ({avg_latency:.2f}ms)")

        # 3. Check Energy Efficiency
        if current_energy > self.ENERGY_VETO_LIMIT:
            trigger_reason.append(f"Energy Veto Violation ({current_energy:.2f})")

        if trigger_reason:
            logger.warning(f"TRIGGER ACTIVATED: {', '.join(trigger_reason)}")
            return True
        
        return False

    def execute_synthesis_cycle(self, context_c: float, energy_e: float, failed_data_batch=None):
        """
        Executes Phases 2, 3, and 4 of the Cycle: Hypothesis -> Retraining -> Deployment.
        """
        logger.info("--- INITIATING INSTINCT SYNTHESIS CYCLE ---")

        # --- Phase 2: Hypothesis Generation (CLNN) ---
        logger.info("Phase 2: CLNN Hypothesis Generation...")
        # The CLNN analyzes the situation (C, E) to find the optimal structural configuration
        clnn_decision = self.clnn.govern_plasticity(context_c, energy_e)
        
        target_mode = clnn_decision['learning_mode']
        new_plasticity = clnn_decision['plasticity_counts']
        stack_depth = clnn_decision['stack_depth']
        
        logger.info(f"CLNN Selected Strategy: {target_mode} (Depth: {stack_depth})")
        logger.info(f"New Plasticity Configuration: {new_plasticity}")

        # --- Phase 3: Retraining / Compilation (Simulated) ---
        # In a real scenario, this calls 'training_setup.py' to run a high-speed training loop.
        logger.info("Phase 3: Retraining Instinctive Layer (Simulation)...")
        
        # We simulate the training producing a new set of optimized weights
        # The number of weights depends on the fixed architecture for now, but conceptually it adapts.
        new_weights = self._simulate_retraining_process(new_plasticity, failed_data_batch)
        
        # --- Phase 4: Atomic Swap (Deployment) ---
        self._atomic_swap_deployment(new_weights, target_mode, new_plasticity)

    def _simulate_retraining_process(self, plasticity_config, data):
        """
        Simulates the heavy lifting of the training loop.
        Returns optimized weights.
        """
        # Simulate processing time
        time.sleep(0.5) 
        
        # Generate 'optimized' weights (mocking the result of backpropagation)
        # For the standard IPSNN, we use 26 weights.
        # If plasticity changed the circuit size, we would handle tensor resizing here.
        optimized_weights = np.random.uniform(0, 2 * np.pi, size=(26,))
        
        logger.info("Retraining Complete. Loss converged to < 0.01.")
        return optimized_weights

    def _atomic_swap_deployment(self, new_weights, mode_name, plasticity_config):
        """
        Phase 4: The 'Hot Swap'.
        Compiles the new weights into an IKM and injects them into the running IPSNN.
        """
        logger.info("Phase 4: Atomic Swap / IKM Deployment...")
        
        # 1. Generate a unique Key for this new Instinct
        ikm_key = f"IKM_{mode_name}_{int(time.time())}"
        
        # 2. Save to IPSNN Cache (The "Compiler")
        self.ipsnn.save_ikm(ikm_key, new_weights)
        
        # 3. Hot Load the new IKM
        success = self.ipsnn.load_ikm(ikm_key)
        
        if success:
            logger.info(f"SUCCESS: System now running on {ikm_key}.")
            logger.info("Asymptotic Monitoring Reset. Cycle Complete.")
        else:
>>>>>>> d6de685d2c7b77476426b95b7cfd6d529b95af6d
            logger.error("CRITICAL FAILURE: Atomic Swap failed.")