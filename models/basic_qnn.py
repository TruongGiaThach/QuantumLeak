 
import torch
import torch.nn as nn

class BasicQNN(nn.Module):
    def __init__(self, n_qubits, n_layers, quantum_circuit):
        super(BasicQNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.pre_net = nn.Linear(16 * 15 * 15, n_qubits)
        self.q_params = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.post_net = nn.Linear(n_qubits, 1)
        self.quantum_circuit = quantum_circuit

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(-1, 16 * 15 * 15)
        x = torch.tanh(self.pre_net(x))
        batch_size = x.size(0)
        x_device = x.device
        x = [self.quantum_circuit(x[i], self.q_params) for i in range(batch_size)]
        x = torch.tensor(x, dtype=torch.float32, device=x_device)
        logits = self.post_net(x)
        return logits