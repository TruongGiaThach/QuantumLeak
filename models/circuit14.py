
 
import torch
import torch.nn as nn
import pennylane as qml

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, quantum_circuit, device_name="lightning.qubit"):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.pre_net = nn.Linear(16 * 15 * 15, 16)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.crx_weights = nn.Parameter(torch.randn(n_layers, 6))
        self.fc = nn.Linear(n_qubits, 1)
        self.dev = qml.device(device_name, wires=n_qubits, shots=None)
        self.quantum_circuit = quantum_circuit

    def forward(self, inputs):
        x = torch.relu(self.conv(inputs))
        x = x.view(-1, 16 * 15 * 15)
        x = torch.tanh(self.pre_net(x))
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)

        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            q_out = self.quantum_circuit(x[i], self.weights, self.crx_weights)
            q_out = torch.stack(q_out).float()
            outputs.append(q_out)
        outputs = torch.stack(outputs)
        probs = self.fc(outputs).sigmoid()
        return probs
