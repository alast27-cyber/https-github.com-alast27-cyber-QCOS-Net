import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QLLMCore(nn.Module):
    def __init__(self, num_qubits=4):
        super(QLLMCore, self).__init__()
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.num_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            
        self.qnode = qnode
        self.weight_shapes = {"weights": (3, self.num_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)
        
    def forward(self, x):
        return self.qlayer(x)

if __name__ == "__main__":
    model = QLLMCore()
    dummy_input = torch.rand(1, 4)
    out = model(dummy_input)
    print("QLLM Output:", out)
