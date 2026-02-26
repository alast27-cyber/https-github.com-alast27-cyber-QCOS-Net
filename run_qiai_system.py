import os
import sys
import torch
import numpy as np
import logging

# --- BOOTSTRAP: Add project root to path before any other imports ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now standard imports will work across the entire directory tree
from ai_core.bridge.qiai_ips_bridge import QIAI_IPS_Bridge
from ai_core.interface.qiai_chat_interface import QIAI_Terminal_Assistant

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] %(message)s')
logger = logging.getLogger("Q-IAI-ROOT")

def execute_full_simulation():
    print("\n" + "="*60)
    print("    AGENT Q: FULL QUANTUM-SEMANTIC OPERATING SYSTEM BOOT")
    print("="*60)
    
    # 1. Initialize the Bridge (The Brain)
    logger.info("Initializing Cognitive Layers...")
    bridge = QIAI_IPS_Bridge()
    
    # 2. Simulate a Real-World Conflict Scenario
    # High Complexity (0.95) and critical Energy state (0.10)
    logger.info("Injecting Kernel Telemetry & Logs...")
    telemetry_metrics = torch.tensor([[0.95, 0.10]]) 
    system_logs = (
        "CRITICAL: Memory management unit (MMU) reports page table walk "
        "overflow. Instinct suggests flushing cache, but Universe Logic "
        "detects a recursive pointer loop."
    )
    
    # 3. The Dual Cognition Cycle
    print("\n" + "-"*40)
    print("RUNNING DUAL COGNITION CYCLE")
    print("-" * 40)
    
    # This triggers IPSNN, Universe Cognition, and the LLM Supervisor
    decision, intent_vec, mode = bridge.forward(telemetry_metrics, system_logs)
    
    # 4. Results
    print("-" * 40)
    print(f"DECISION MODE: {mode}")
    print(f"V-SCORE (Action Probability): {decision:.4f}")
    print(f"INTENT VECTOR: Generated ({intent_vec.shape[1]} dimensions)")
    print("-" * 40)

    print("\n" + "="*60)
    print("    SIMULATION COMPLETE - AGENT Q IS STABLE")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        execute_full_simulation()
        
        # Launch the Chat Interface
        print("Starting Terminal Assistant for real-time interaction...")
        assistant = QIAI_Terminal_Assistant()
        assistant.chat_loop()
    except KeyboardInterrupt:
        print("\nSystem shutdown by user.")
    except Exception as e:
        logger.error(f"Fatal System Error: {e}")
