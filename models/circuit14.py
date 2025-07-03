
from torchvision import transforms
import torch
import torch.nn as nn
import pennylane as qml
from quantum_circuits.circuit14_circuit import create_circuit14

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, device_name="lightning.gpu"): 
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.pre_net = nn.Linear(16 * 15 * 15, 2**n_qubits) 
        self.quantum_circuit, crx_per_layer = create_circuit14(
            n_qubits=self.n_qubits, 
            n_layers=self.n_layers, 
            device_name=device_name
        )
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.crx_weights = nn.Parameter(torch.randn(n_layers, crx_per_layer))
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, inputs):
        x = torch.relu(self.conv(inputs))
        x = x.view(-1, 16 * 15 * 15)
        x = torch.tanh(self.pre_net(x)) 

        # --- TEMP FIX NORM ERROR ---
        # Đảm bảo không có vector nào có norm chính xác bằng 0
        norms = torch.norm(x, p=2, dim=1, keepdim=True)
        zero_mask = (norms < 1e-8)
        noise = torch.randn_like(x) * 1e-8
        x = torch.where(zero_mask, noise, x)
        
        # Sau đó mới chuẩn hóa
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        # ---------------------------

        outputs = self.quantum_circuit(x, self.weights, self.crx_weights)
        logits = self.fc(outputs)
        return logits
        
class PureQuantumCircuit14(nn.Module):
    """
    Một kiến trúc QNN dựa trên Circuit 14, giảm thiểu tiền xử lý cổ điển
    và mã hóa dữ liệu ảnh trực tiếp hơn vào mạch lượng tử.
    """
    def __init__(self, n_qubits, n_layers, device_name="lightning.gpu"): 
        super(PureQuantumCircuit14, self).__init__()
        
        if 2**n_qubits < 64:
            raise ValueError("Cần ít nhất 6 qubits (2^6=64) để mã hóa ảnh 8x8.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.downsample = transforms.Resize((8, 8), antialias=True)

        self.quantum_circuit, crx_per_layer = create_circuit14(
            n_qubits=self.n_qubits, 
            n_layers=self.n_layers, 
            device_name=device_name
        )
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.crx_weights = nn.Parameter(torch.randn(n_layers, crx_per_layer))

        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, inputs):
        x = self.downsample(inputs)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
        outputs = self.quantum_circuit(x, self.weights, self.crx_weights)

        logits  = self.fc(outputs)
        return logits 
