import numpy as np
import os

# Define the path to save the weights
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(current_dir, '..', 'models'))
weights_path = os.path.join(models_dir, 'trained_weights.npy')

# Generate some dummy weights (e.g., 26 parameters for the QNN)
dummy_weights = np.random.rand(26)

# Save the weights
np.save(weights_path, dummy_weights)

print(f"Successfully saved trained weights to: {weights_path}")
print("Ready to run the kernel simulation again.")
