import torch
import torch.nn as nn
import torch.nn.functional as F

class IAI_IPS_QNN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
        super(IAI_IPS_QNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    model = IAI_IPS_QNN()
    dummy_input = torch.randn(1, 64)
    final_output = model(dummy_input)
    print(f"System Final Output Value: {final_output.item():.4f}")
