 
import torch
import torch.nn as nn
import pennylane as qml
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
class DressedQuantumCircuit(nn.Module):
    def __init__(self, n_qubits, n_layers, device_name="default.qubit"):
        super(DressedQuantumCircuit, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pre_net = nn.Linear(512, n_qubits)
        self.post_net = nn.Linear(n_qubits, 1)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, requires_grad=True))
        self.dev = qml.device(device_name, wires=n_qubits)
        self.quantum_circuit = self._create_quantum_circuit()

    def _create_quantum_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def quantum_circuit(inputs, weights):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(inputs[i] * np.pi / 2, wires=i)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return quantum_circuit

    def forward(self, x):
        x = self.pre_net(x)
        q_out = torch.zeros(x.size(0), self.n_qubits, dtype=torch.float32).to(x.device)
        for i in range(x.size(0)):
            q_out[i] = torch.tensor(self.quantum_circuit(x[i], self.weights), dtype=torch.float32).to(x.device)
        return self.post_net(q_out)

class CQTransferLearningModel(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(CQTransferLearningModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.resnet.fc = nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.quantum_net = DressedQuantumCircuit(n_qubits, n_layers)

    def forward(self, x):
        features = self.resnet(x)
        return self.quantum_net(features)