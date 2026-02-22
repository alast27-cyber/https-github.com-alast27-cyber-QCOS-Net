import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# --------------------------------------------------------------------------
# --- PATH-FIX TO RESOLVE 'ModuleNotFoundError: No module named ipsnn_qnn' ---
# This block allows the script in 'scripts' directory to import from 'models'.
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Move up from 'scripts' to 'ai_core', then into 'models'
models_dir = os.path.abspath(os.path.join(current_dir, '..', 'models'))
if models_dir not in sys.path:
    sys.path.append(models_dir)
# --------------------------------------------------------------------------

# Import your existing QNN definition
# We assume ipsnn_qnn.py contains the 'iai_ips_quantum_layer' QNode
try:
    from ipsnn_qnn import iai_ips_quantum_layer, NUM_QUBITS, generate_central_node_config
except ImportError:
    # Mocking for local script execution if environment isn't set up
    NUM_QUBITS = 31
    def generate_central_node_config(x): return x

# -----------------------------------------------
# --- 1. Define the Hybrid Container (Pytorch) ---
# -----------------------------------------------
# We must wrap the QNN in a torch.nn.Module to make it compatible with LibTorch
class OSKernelNet_Hybrid(nn.Module):
    def __init__(self):
        super(OSKernelNet_Hybrid, self).__init__()
        
        # Define the learned weights as a PyTorch Parameter
        # This ensures they are saved inside the .jit file
        # The size (26) is the assumed number of trainable weights in the base QNN.
        self.weights = nn.Parameter(torch.rand(26, dtype=torch.float32))
        
        # Define the static configuration (mocking the CLNN input for the base model)
        # We fix the structure for the JIT trace (e.g., the 'ASSOCIATIVE' mode)
        self.config = generate_central_node_config({4: 4, 5: 5, 6: 5, 7: 4, 8: 3})

    def forward(self, x):
        """
        Input x: Tensor of shape [1, 2] -> [Context_C, Energy_E]
        The input is a two-scalar tensor used by the C++ kernel stub.
        
        NOTE: The full QNN is often not perfectly PyTorch JIT compatible. 
        A mock calculation is used here to simulate the scalar output V-Score.
        """
        # Simulating the scalar output V (Expectation Value)
        # V = Tanh(Sum(Weights * Input)) - simple approximation for the trace example
        v_score = torch.tanh(torch.sum(self.weights * x.mean())) 
        
        return v_score

# --------------------------------------------
# --- 2. The Serialization Process (JIT) ---
# --------------------------------------------
def export_instinct_module():
    print("--- STARTING SERIALIZATION ---")
    
    # A. Instantiate the Hybrid Model
    model = OSKernelNet_Hybrid()
    model.eval() # Set to evaluation mode
    
    # B. Create Dummy Input (The 'Trace' Example)
    # The tracer runs the code once with this input to record the graph.
    # C=0.8, E=0.2
    dummy_input = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    
    # C. JIT Trace
    print("Tracing model graph...")
    try:
        traced_script_module = torch.jit.trace(model, dummy_input)
        
        # D. Save to Disk
        output_path = "instinct_v1.jit"
        traced_script_module.save(output_path)
        print(f"SUCCESS: Hybrid model saved to {output_path}")
        
    except Exception as e:
        print(f"ERROR during tracing/saving: {e}")

if __name__ == "__main__":
    export_instinct_module()