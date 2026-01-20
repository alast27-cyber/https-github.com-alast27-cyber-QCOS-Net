<<<<<<< HEAD
import torch
import torch.nn as nn

# --- Configuration Constants (matching the I/O Protocol) ---
TIME_STEPS = 60      # T: 60 seconds of history
METRIC_FEATURES = 8  # M: CPU, Mem, I/O, etc.
SEMANTIC_DIM = 768   # D_sem: Sentence Transformer embedding size
ENCODER_OUT_DIM = 128# E: Desired fixed-size vector for each branch
NUM_ACTIONS = 5      # A: Number of discrete OS actions

class OSKernelNet(nn.Module):
    def __init__(self, input_size=METRIC_FEATURES, 
                 intent_dim=SEMANTIC_DIM, 
                 hidden_size=ENCODER_OUT_DIM):
        
        super(OSKernelNet, self).__init__()
        
        # 1. Time-Series Encoder Branch (LSTM for System State)
        # Input: (B, T, M) -> Output (B, E)
        self.ts_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2, # Two LSTM layers for depth
            batch_first=True,
            dropout=0.1 # Dropout for regularization
        )
        
        # 2. Semantic Encoder Branch (Linear Layer for Intent)
        # Input: (B, D_sem) -> Output (B, E)
        self.sem_encoder = nn.Sequential(
            nn.Linear(intent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size) # Maps to the same hidden_size E
        )
        
        # Feature Fusion Dimension (2 * E)
        fusion_dim = 2 * hidden_size
        
        # 3. Multi-Task Decision Head
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Head A: Predictive Metrics (Resource Spike & Failure Probability)
        # Output: (B, 2)
        self.pred_head = nn.Linear(fusion_dim // 2, 2)
        
        # Head B: Action Recommendation (5 Discrete Actions)
        # Output: (B, A) -> 5
        self.action_head = nn.Linear(fusion_dim // 2, NUM_ACTIONS)

    def forward(self, sys_state: torch.Tensor, intent_vec: torch.Tensor):
        
        # 1. Process Time-Series (System State)
        # The LSTM returns two things: (output, (h_n, c_n))
        # We only care about h_n, the final hidden state, which summarizes the sequence
        _, (h_n, _) = self.ts_encoder(sys_state)
        # h_n shape: (num_layers, B, hidden_size). We take the last layer's hidden state.
        ts_features = h_n[-1, :, :] # Shape: (B, E)

        # 2. Process Semantic Intent
        sem_features = self.sem_encoder(intent_vec) # Shape: (B, E)
        
        # 3. Feature Fusion: Concatenate the two feature vectors
        fused_features = torch.cat((ts_features, sem_features), dim=1) # Shape: (B, 2*E)
        
        # Pass through the shared MLP
        shared_output = self.shared_mlp(fused_features) # Shape: (B, E)
        
        # 4. Independent Predictions
        
        # Head A: Predictive Metrics (Sigmoid for probability)
        # Example output: [0.95, 0.05] -> 95% spike probability, 5% failure probability
        pred_metrics_logits = self.pred_head(shared_output)
        pred_metrics = torch.sigmoid(pred_metrics_logits)
        
        # Head B: Action Recommendation (Softmax for action distribution)
        # Example output: [0.1, 0.8, 0.05, 0.05, 0.0] -> 80% confidence for Action 2
        action_logits = self.action_head(shared_output)
        action_rec = torch.softmax(action_logits, dim=1)
        
        # Return the two distinct, final outputs
        return pred_metrics, action_rec

=======
import torch
import torch.nn as nn

# --- Configuration Constants (matching the I/O Protocol) ---
TIME_STEPS = 60      # T: 60 seconds of history
METRIC_FEATURES = 8  # M: CPU, Mem, I/O, etc.
SEMANTIC_DIM = 768   # D_sem: Sentence Transformer embedding size
ENCODER_OUT_DIM = 128# E: Desired fixed-size vector for each branch
NUM_ACTIONS = 5      # A: Number of discrete OS actions

class OSKernelNet(nn.Module):
    def __init__(self, input_size=METRIC_FEATURES, 
                 intent_dim=SEMANTIC_DIM, 
                 hidden_size=ENCODER_OUT_DIM):
        
        super(OSKernelNet, self).__init__()
        
        # 1. Time-Series Encoder Branch (LSTM for System State)
        # Input: (B, T, M) -> Output (B, E)
        self.ts_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2, # Two LSTM layers for depth
            batch_first=True,
            dropout=0.1 # Dropout for regularization
        )
        
        # 2. Semantic Encoder Branch (Linear Layer for Intent)
        # Input: (B, D_sem) -> Output (B, E)
        self.sem_encoder = nn.Sequential(
            nn.Linear(intent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size) # Maps to the same hidden_size E
        )
        
        # Feature Fusion Dimension (2 * E)
        fusion_dim = 2 * hidden_size
        
        # 3. Multi-Task Decision Head
        self.shared_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Head A: Predictive Metrics (Resource Spike & Failure Probability)
        # Output: (B, 2)
        self.pred_head = nn.Linear(fusion_dim // 2, 2)
        
        # Head B: Action Recommendation (5 Discrete Actions)
        # Output: (B, A) -> 5
        self.action_head = nn.Linear(fusion_dim // 2, NUM_ACTIONS)

    def forward(self, sys_state: torch.Tensor, intent_vec: torch.Tensor):
        
        # 1. Process Time-Series (System State)
        # The LSTM returns two things: (output, (h_n, c_n))
        # We only care about h_n, the final hidden state, which summarizes the sequence
        _, (h_n, _) = self.ts_encoder(sys_state)
        # h_n shape: (num_layers, B, hidden_size). We take the last layer's hidden state.
        ts_features = h_n[-1, :, :] # Shape: (B, E)

        # 2. Process Semantic Intent
        sem_features = self.sem_encoder(intent_vec) # Shape: (B, E)
        
        # 3. Feature Fusion: Concatenate the two feature vectors
        fused_features = torch.cat((ts_features, sem_features), dim=1) # Shape: (B, 2*E)
        
        # Pass through the shared MLP
        shared_output = self.shared_mlp(fused_features) # Shape: (B, E)
        
        # 4. Independent Predictions
        
        # Head A: Predictive Metrics (Sigmoid for probability)
        # Example output: [0.95, 0.05] -> 95% spike probability, 5% failure probability
        pred_metrics_logits = self.pred_head(shared_output)
        pred_metrics = torch.sigmoid(pred_metrics_logits)
        
        # Head B: Action Recommendation (Softmax for action distribution)
        # Example output: [0.1, 0.8, 0.05, 0.05, 0.0] -> 80% confidence for Action 2
        action_logits = self.action_head(shared_output)
        action_rec = torch.softmax(action_logits, dim=1)
        
        # Return the two distinct, final outputs
        return pred_metrics, action_rec

>>>>>>> d6de685d2c7b77476426b95b7cfd6d529b95af6d
# --- End of OSKernelNet Class ---