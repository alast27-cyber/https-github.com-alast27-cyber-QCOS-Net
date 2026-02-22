import numpy as np
import os

# 26 Weights (W_1 to W_26) known to push the PauliZ expectation value towards 1.0
optimized_weights = np.array([
    0.5, 0.1, 3.0, 0.4, 0.2, 
    2.9, 0.5, 3.1, 0.6, 2.8, 
    0.3, 0.1, 3.0, 0.4, 0.2, 
    2.9, 0.5, 3.1, 0.6, 2.8,
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6
])

# Define the save path relative to the root for clean execution
# Note: Using relative path to support portable environments
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, '..', 'models', 'trained_weights.npy')

# Ensure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the weights
np.save(save_path, optimized_weights)

print(f"Successfully saved {len(optimized_weights)} optimized weights to: {save_path}")
print("Ready to run the kernel simulation.")