import torch
from system.iai_system import IAI_IPS_System

def run_qiai_simulation():
    """
    Initializes and runs a full simulation of the Q-IAI Dual Cognition System.
    """
    print('========================================================')
    print('======      AGENT Q: Q-IAI SYSTEM SIMULATION      ======')
    print('========================================================')

    # 1. Instantiate the master system
    # This will print the boot sequence of all integrated QNN layers.
    agent_q = IAI_IPS_System(input_features=64, system_metrics_features=2)

    # 2. Simulate incoming data streams
    # System Metrics: [Complexity, Energy]
    # A high-complexity, low-energy task.
    system_metrics = torch.tensor([[0.9, 0.2]], dtype=torch.float32)
    
    # Raw Data Stream: A 64-dimensional vector representing system telemetry.
    raw_data_stream = torch.randn(1, 64)

    # 3. Execute the Dual Cognition Cycle
    # The `forward` method of the IAI_IPS_System will be called.
    print('\n>>> EXECUTING AGENT Q DUAL COGNITION BRAIN...')
    final_decision = agent_q(raw_data_stream, system_metrics, debug=True)

    print('\n--- SIMULATION COMPLETE ---')
    print('Final Decision Output: {:.4f}'.format(final_decision.item()))
    print('========================================================')

if __name__ == "__main__":
    run_qiai_simulation()
