<<<<<<< HEAD
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
save_path = os.path.join(
    r"C:\Users\alast\OneDrive\Apps\QCOS_Project\ai_core\models", 
    "trained_weights.npy"
)

# Ensure the directory exists (it should, but safety first)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the weights
np.save(save_path, optimized_weights)

print(f"Successfully saved {len(optimized_weights)} optimized weights to: {save_path}")
=======
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
save_path = os.path.join(
    r"C:\Users\alast\OneDrive\Apps\QCOS_Project\ai_core\models", 
    "trained_weights.npy"
)

# Ensure the directory exists (it should, but safety first)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the weights
np.save(save_path, optimized_weights)

print(f"Successfully saved {len(optimized_weights)} optimized weights to: {save_path}")
>>>>>>> d6de685d2c7b77476426b95b7cfd6d529b95af6d
print("Ready to run the kernel simulation again.")