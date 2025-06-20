 
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QNN(nn.Module):
    def __init__(self, n_qubits, quantum_circuit, n_layers):
        super(QNN, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.pre_net = nn.Linear(16, 16)
        self.q_params = nn.Parameter(torch.normal(mean=0, std=np.pi, size=(n_layers * n_qubits,)))
        self.post_net = nn.Linear(n_qubits, 1)
        self.quantum_circuit = quantum_circuit
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, 16)
        x = torch.tanh(self.pre_net(x))
        batch_size = x.size(0)
        x_device = x.device
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = [self.quantum_circuit(x[i], self.q_params) for i in range(batch_size)]
        x = torch.tensor(x, dtype=torch.float32, device=x_device)
        x = torch.sigmoid(self.post_net(x))
        return x