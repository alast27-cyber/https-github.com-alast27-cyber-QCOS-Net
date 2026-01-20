<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. CORE COMPONENT: The IAI-IPS Base Layer (10-Row Structure)
# ----------------------------------------------------------------------

class IAI_IPS_NN_Layer(nn.Module):
    """
    The core 10-row, 31-node IAI-IPS Neural Network Layer. 
    Implements the dynamic plasticity based on the Central Node's configuration.
    """
    FIXED_NODE_COUNTS = {1: 1, 2: 2, 3: 3, 9: 2, 10: 1}
    DEFAULT_PLASTICITY_COUNTS = {4: 4, 5: 5, 6: 5, 7: 4, 8: 3}
    
    def __init__(self, input_features: int, central_node_config: dict = None, activation=nn.ReLU):
        super().__init__()
        
        # --- Determine Node Counts (Plasticity) ---
        self.node_counts = self.DEFAULT_PLASTICITY_COUNTS.copy()
        if central_node_config and 'plasticity_counts' in central_node_config:
            self.node_counts.update(central_node_config['plasticity_counts'])

        # Calculate all row sizes for sequential connectivity
        row_sizes = [
            self.FIXED_NODE_COUNTS[1], self.FIXED_NODE_COUNTS[2], self.FIXED_NODE_COUNTS[3],
            self.node_counts[4], self.node_counts[5], self.node_counts[6], 
            self.node_counts[7], self.node_counts[8],
            self.FIXED_NODE_COUNTS[9], self.FIXED_NODE_COUNTS[10]
        ]

        # --- Define Fixed Layers ---
        self.row1 = nn.Linear(input_features, row_sizes[0])  # Entry Node
        self.row2 = nn.Linear(row_sizes[0], row_sizes[1])    # Duality
        self.row3 = nn.Linear(row_sizes[1], row_sizes[2])    # Conflict

        # --- Define Plasticity Region (Rows 4-8) ---
        plasticity_layers = []
        current_in_size = row_sizes[2] 

        for r in range(4, 9):
            out_size = self.node_counts.get(r)
            
            plasticity_layers.append(nn.Linear(current_in_size, out_size))
            plasticity_layers.append(activation())
            current_in_size = out_size

        self.plasticity_region = nn.Sequential(*plasticity_layers)
        
        # --- Define Fixed Exit Layers ---
        self.row9 = nn.Linear(current_in_size, self.FIXED_NODE_COUNTS[9])
        self.row10 = nn.Linear(self.FIXED_NODE_COUNTS[9], self.FIXED_NODE_COUNTS[10]) # Result Node
        
        self.activation = activation()
        # print(f"IAI-IPS Layer initialized with plasticity counts: {self.node_counts}")

    def forward(self, x: torch.Tensor, debug: bool = False):
        if debug: 
            print(f"  > Input shape: {x.shape}")
        
        # Sequential Forward Pass
        x = self.activation(self.row1(x))
        x = self.activation(self.row2(x))
        x = self.activation(self.row3(x))
        x = self.plasticity_region(x)
        x = self.activation(self.row9(x))
        x = self.row10(x) # Final output layer often without activation

        if debug:
            print(f"  > Final Layer Output (Row 10) shape: {x.shape}")
        
        return x

# ----------------------------------------------------------------------
# 2. CENTRAL NODE LOGIC (Simulates the CLNN's Optimization)
# ----------------------------------------------------------------------

def generate_central_node_config(complexity_score: float, energy_budget: float):
    """
    Simulates the Variational Optimizer (Central Node / CLNN's function).
    It determines the structural parameters of the ILNN and IPSNN.
    """
    new_counts = IAI_IPS_NN_Layer.DEFAULT_PLASTICITY_COUNTS.copy()
    stack_depth = 3 # Default stack depth

    # Logic: If complexity is high, increase node count (plasticity)
    if complexity_score > 0.8 and energy_budget > 0.5:
        new_counts = {4: 6, 5: 8, 6: 8, 7: 6, 8: 4} # High plasticity config
        stack_depth = 5 
        print("CLNN: High complexity/energy mode activated. Increasing plasticity.")
    else:
        print("CLNN: Default complexity/energy mode activated. Using default plasticity.")
        
    return {
        'plasticity_counts': new_counts,
        'stack_depth': stack_depth
    }

# ----------------------------------------------------------------------
# 3. THE FULL IAI-IPS SYSTEM ARCHITECTURE
# ----------------------------------------------------------------------

class IAI_IPS_System(nn.Module):
    """
    The full three-network architecture: CLNN (Govern), ILNN (Learn), IPSNN (Synthesize).
    """
    def __init__(self, input_features: int, system_metrics_features: int, output_features: int = 1):
        super().__init__()
        
        self.input_features = input_features
        self.system_metrics_features = system_metrics_features

        # 1. Cognition Layer NN (CLNN): The Administrator / OS 
        # Output features are typically control signals (e.g., config parameters).
        # We simplify its structure for this conceptual example.
        self.CLNN = nn.Linear(system_metrics_features, 3) # Placeholder for control signal output
        
        # The ILNN and IPSNN are defined later in forward() 
        # to incorporate the dynamic configuration from the CLNN.
        self.ilnn_instance = None
        self.ipsnn_instance = None
        
        # We ensure the final output layer of IPSNN matches the desired output_features
        self.final_output_layer = nn.Linear(IAI_IPS_NN_Layer.FIXED_NODE_COUNTS[10], output_features)

    def forward(self, x: torch.Tensor, system_metrics: torch.Tensor, debug: bool = False):
        """
        Executes the cognitive loop: GOVERN -> LEARN -> SYNTHESIZE.
        
        Args:
            x (torch.Tensor): The main input data (e.g., streaming data, image features).
            system_metrics (torch.Tensor): The system state (e.g., complexity, energy budget).
        """
        if debug: 
            print("\n--- 1. GOVERN (CLNN Execution) ---")

        # CLNN's primary function is to determine configuration (plasticity)
        # We extract key metrics from the system_metrics tensor for the Central Node logic
        complexity_score = system_metrics[:, 0].mean().item()
        energy_budget = system_metrics[:, 1].mean().item()
        
        # The administrative logic runs externally (representing the CLNN's function)
        config = generate_central_node_config(complexity_score, energy_budget)
        
        # NOTE: Since the architecture is dynamic (plastic), we must re-instantiate 
        # the ILNN and IPSNN or use a custom PyTorch technique to swap layers based on 'config'.
        # For simplicity, we re-instantiate the sub-networks here.
        if self.ilnn_instance is None or self.ilnn_instance.node_counts != config['plasticity_counts']:
            self.ilnn_instance = IAI_IPS_NN_Layer(self.input_features, config, activation=nn.ReLU)
            self.ipsnn_instance = IAI_IPS_NN_Layer(IAI_IPS_NN_Layer.FIXED_NODE_COUNTS[10], config, activation=nn.ReLU)
            
            if debug: print("CLNN forced re-initialization of ILNN/IPSNN due to plasticity change.")

        # --- 2. LEARN (ILNN Execution) ---
        if debug: 
            print("\n--- 2. LEARN (ILNN: Data Stream Processor) ---")

        # ILNN processes the input data
        ilnn_output = self.ilnn_instance(x, debug=debug) # Output: Knowledge States/Models

        # --- 3. SYNTHESIZE (IPSNN Execution) ---
        if debug: 
            print("\n--- 3. SYNTHESIZE (IPSNN: Action Generator) ---")

        # IPSNN takes the knowledge state (ILNN output) and synthesizes the final action
        ipsnn_output = self.ipsnn_instance(ilnn_output, debug=debug) 
        
        # The final result output, ensuring it has the correct final dimension
        final_result = self.final_output_layer(ipsnn_output)

        if debug:
            print(f"\n--- Final System Output ---")
            print(f"Final Action/Solution shape: {final_result.shape}")

        return final_result

# ----------------------------------------------------------------------
# 4. TEST EXECUTION (Run this block)
# ----------------------------------------------------------------------

# Define simulation parameters
INPUT_FEATURES = 64        # Size of raw data input
METRICS_FEATURES = 2       # [Complexity Score, Energy Budget]
OUTPUT_FEATURES = 1        # Single result (e.g., binary action)

# Initialize the full system
system = IAI_IPS_System(input_features=INPUT_FEATURES, system_metrics_features=METRICS_FEATURES, output_features=OUTPUT_FEATURES)

# Create dummy input data and system metrics tensor
# System metrics tensor for a batch of 1: [Complexity_Score, Energy_Budget]
# Here, we set high complexity and high energy budget to trigger plasticity
system_metrics_data = torch.tensor([[0.95, 0.8]], dtype=torch.float32)
input_data = torch.randn(1, INPUT_FEATURES)

# Run the full forward pass
final_output = system(input_data, system_metrics_data, debug=True)
=======
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. CORE COMPONENT: The IAI-IPS Base Layer (10-Row Structure)
# ----------------------------------------------------------------------

class IAI_IPS_NN_Layer(nn.Module):
    """
    The core 10-row, 31-node IAI-IPS Neural Network Layer. 
    Implements the dynamic plasticity based on the Central Node's configuration.
    """
    FIXED_NODE_COUNTS = {1: 1, 2: 2, 3: 3, 9: 2, 10: 1}
    DEFAULT_PLASTICITY_COUNTS = {4: 4, 5: 5, 6: 5, 7: 4, 8: 3}
    
    def __init__(self, input_features: int, central_node_config: dict = None, activation=nn.ReLU):
        super().__init__()
        
        # --- Determine Node Counts (Plasticity) ---
        self.node_counts = self.DEFAULT_PLASTICITY_COUNTS.copy()
        if central_node_config and 'plasticity_counts' in central_node_config:
            self.node_counts.update(central_node_config['plasticity_counts'])

        # Calculate all row sizes for sequential connectivity
        row_sizes = [
            self.FIXED_NODE_COUNTS[1], self.FIXED_NODE_COUNTS[2], self.FIXED_NODE_COUNTS[3],
            self.node_counts[4], self.node_counts[5], self.node_counts[6], 
            self.node_counts[7], self.node_counts[8],
            self.FIXED_NODE_COUNTS[9], self.FIXED_NODE_COUNTS[10]
        ]

        # --- Define Fixed Layers ---
        self.row1 = nn.Linear(input_features, row_sizes[0])  # Entry Node
        self.row2 = nn.Linear(row_sizes[0], row_sizes[1])    # Duality
        self.row3 = nn.Linear(row_sizes[1], row_sizes[2])    # Conflict

        # --- Define Plasticity Region (Rows 4-8) ---
        plasticity_layers = []
        current_in_size = row_sizes[2] 

        for r in range(4, 9):
            out_size = self.node_counts.get(r)
            
            plasticity_layers.append(nn.Linear(current_in_size, out_size))
            plasticity_layers.append(activation())
            current_in_size = out_size

        self.plasticity_region = nn.Sequential(*plasticity_layers)
        
        # --- Define Fixed Exit Layers ---
        self.row9 = nn.Linear(current_in_size, self.FIXED_NODE_COUNTS[9])
        self.row10 = nn.Linear(self.FIXED_NODE_COUNTS[9], self.FIXED_NODE_COUNTS[10]) # Result Node
        
        self.activation = activation()
        # print(f"IAI-IPS Layer initialized with plasticity counts: {self.node_counts}")

    def forward(self, x: torch.Tensor, debug: bool = False):
        if debug: 
            print(f"  > Input shape: {x.shape}")
        
        # Sequential Forward Pass
        x = self.activation(self.row1(x))
        x = self.activation(self.row2(x))
        x = self.activation(self.row3(x))
        x = self.plasticity_region(x)
        x = self.activation(self.row9(x))
        x = self.row10(x) # Final output layer often without activation

        if debug:
            print(f"  > Final Layer Output (Row 10) shape: {x.shape}")
        
        return x

# ----------------------------------------------------------------------
# 2. CENTRAL NODE LOGIC (Simulates the CLNN's Optimization)
# ----------------------------------------------------------------------

def generate_central_node_config(complexity_score: float, energy_budget: float):
    """
    Simulates the Variational Optimizer (Central Node / CLNN's function).
    It determines the structural parameters of the ILNN and IPSNN.
    """
    new_counts = IAI_IPS_NN_Layer.DEFAULT_PLASTICITY_COUNTS.copy()
    stack_depth = 3 # Default stack depth

    # Logic: If complexity is high, increase node count (plasticity)
    if complexity_score > 0.8 and energy_budget > 0.5:
        new_counts = {4: 6, 5: 8, 6: 8, 7: 6, 8: 4} # High plasticity config
        stack_depth = 5 
        print("CLNN: High complexity/energy mode activated. Increasing plasticity.")
    else:
        print("CLNN: Default complexity/energy mode activated. Using default plasticity.")
        
    return {
        'plasticity_counts': new_counts,
        'stack_depth': stack_depth
    }

# ----------------------------------------------------------------------
# 3. THE FULL IAI-IPS SYSTEM ARCHITECTURE
# ----------------------------------------------------------------------

class IAI_IPS_System(nn.Module):
    """
    The full three-network architecture: CLNN (Govern), ILNN (Learn), IPSNN (Synthesize).
    """
    def __init__(self, input_features: int, system_metrics_features: int, output_features: int = 1):
        super().__init__()
        
        self.input_features = input_features
        self.system_metrics_features = system_metrics_features

        # 1. Cognition Layer NN (CLNN): The Administrator / OS 
        # Output features are typically control signals (e.g., config parameters).
        # We simplify its structure for this conceptual example.
        self.CLNN = nn.Linear(system_metrics_features, 3) # Placeholder for control signal output
        
        # The ILNN and IPSNN are defined later in forward() 
        # to incorporate the dynamic configuration from the CLNN.
        self.ilnn_instance = None
        self.ipsnn_instance = None
        
        # We ensure the final output layer of IPSNN matches the desired output_features
        self.final_output_layer = nn.Linear(IAI_IPS_NN_Layer.FIXED_NODE_COUNTS[10], output_features)

    def forward(self, x: torch.Tensor, system_metrics: torch.Tensor, debug: bool = False):
        """
        Executes the cognitive loop: GOVERN -> LEARN -> SYNTHESIZE.
        
        Args:
            x (torch.Tensor): The main input data (e.g., streaming data, image features).
            system_metrics (torch.Tensor): The system state (e.g., complexity, energy budget).
        """
        if debug: 
            print("\n--- 1. GOVERN (CLNN Execution) ---")

        # CLNN's primary function is to determine configuration (plasticity)
        # We extract key metrics from the system_metrics tensor for the Central Node logic
        complexity_score = system_metrics[:, 0].mean().item()
        energy_budget = system_metrics[:, 1].mean().item()
        
        # The administrative logic runs externally (representing the CLNN's function)
        config = generate_central_node_config(complexity_score, energy_budget)
        
        # NOTE: Since the architecture is dynamic (plastic), we must re-instantiate 
        # the ILNN and IPSNN or use a custom PyTorch technique to swap layers based on 'config'.
        # For simplicity, we re-instantiate the sub-networks here.
        if self.ilnn_instance is None or self.ilnn_instance.node_counts != config['plasticity_counts']:
            self.ilnn_instance = IAI_IPS_NN_Layer(self.input_features, config, activation=nn.ReLU)
            self.ipsnn_instance = IAI_IPS_NN_Layer(IAI_IPS_NN_Layer.FIXED_NODE_COUNTS[10], config, activation=nn.ReLU)
            
            if debug: print("CLNN forced re-initialization of ILNN/IPSNN due to plasticity change.")

        # --- 2. LEARN (ILNN Execution) ---
        if debug: 
            print("\n--- 2. LEARN (ILNN: Data Stream Processor) ---")

        # ILNN processes the input data
        ilnn_output = self.ilnn_instance(x, debug=debug) # Output: Knowledge States/Models

        # --- 3. SYNTHESIZE (IPSNN Execution) ---
        if debug: 
            print("\n--- 3. SYNTHESIZE (IPSNN: Action Generator) ---")

        # IPSNN takes the knowledge state (ILNN output) and synthesizes the final action
        ipsnn_output = self.ipsnn_instance(ilnn_output, debug=debug) 
        
        # The final result output, ensuring it has the correct final dimension
        final_result = self.final_output_layer(ipsnn_output)

        if debug:
            print(f"\n--- Final System Output ---")
            print(f"Final Action/Solution shape: {final_result.shape}")

        return final_result

# ----------------------------------------------------------------------
# 4. TEST EXECUTION (Run this block)
# ----------------------------------------------------------------------

# Define simulation parameters
INPUT_FEATURES = 64        # Size of raw data input
METRICS_FEATURES = 2       # [Complexity Score, Energy Budget]
OUTPUT_FEATURES = 1        # Single result (e.g., binary action)

# Initialize the full system
system = IAI_IPS_System(input_features=INPUT_FEATURES, system_metrics_features=METRICS_FEATURES, output_features=OUTPUT_FEATURES)

# Create dummy input data and system metrics tensor
# System metrics tensor for a batch of 1: [Complexity_Score, Energy_Budget]
# Here, we set high complexity and high energy budget to trigger plasticity
system_metrics_data = torch.tensor([[0.95, 0.8]], dtype=torch.float32)
input_data = torch.randn(1, INPUT_FEATURES)

# Run the full forward pass
final_output = system(input_data, system_metrics_data, debug=True)
>>>>>>> d6de685d2c7b77476426b95b7cfd6d529b95af6d
print(f"System Final Output Value: {final_output.item():.4f}")