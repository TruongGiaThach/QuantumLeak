
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
        self.pre_net = nn.Linear(16 * 15 * 15, self.n_qubits) 
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
        # tanh để đưa giá trị về [-1, 1], phù hợp để làm góc quay
        x = torch.tanh(self.pre_net(x)) 

        outputs = self.quantum_circuit(x, self.weights, self.crx_weights)
        if isinstance(outputs, list):
             # Chuyển đổi list của các tuple/tensor thành một tensor duy nhất
             # Giả sử mỗi phần tử trong list là một tuple các kết quả đo
             # outputs sẽ là list của các tensor con, mỗi cái shape [n_qubits]
             outputs = torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in outputs]).to(x.device)
        
        # Đảm bảo outputs có shape [batch_size, n_qubits]
        # Nếu outputs đang có shape [n_qubits, batch_size], ta cần chuyển vị nó
        if len(outputs.shape) > 1 and outputs.shape[0] != x.shape[0]:
             outputs = outputs.t()

        # Kiểm tra lại shape một lần cuối trước khi vào lớp Linear
        # assert outputs.shape == (x.shape[0], self.n_qubits)

        logits = self.fc(outputs)
        return logits
        
class PureQuantumCircuit14(nn.Module):
    """
    Một kiến trúc QNN dựa trên Circuit 14, giảm thiểu tiền xử lý cổ điển
    và mã hóa dữ liệu ảnh trực tiếp hơn vào mạch lượng tử.
    """
    def __init__(self, n_qubits, n_layers, device_name="lightning.gpu"): 
        super(PureQuantumCircuit14, self).__init__()
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
        self.feature_reduction = nn.Linear(64, n_qubits)
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, inputs):
        x = self.downsample(inputs)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.feature_reduction(x))
        outputs = self.quantum_circuit(x, self.weights, self.crx_weights)

        if isinstance(outputs, list):
             # Chuyển đổi list của các tuple/tensor thành một tensor duy nhất
             # Giả sử mỗi phần tử trong list là một tuple các kết quả đo
             # outputs sẽ là list của các tensor con, mỗi cái shape [n_qubits]
             outputs = torch.stack([torch.as_tensor(o, dtype=torch.float32) for o in outputs]).to(x.device)
        
        # Đảm bảo outputs có shape [batch_size, n_qubits]
        # Nếu outputs đang có shape [n_qubits, batch_size], ta cần chuyển vị nó
        if len(outputs.shape) > 1 and outputs.shape[0] != x.shape[0]:
             outputs = outputs.t()

        # Kiểm tra lại shape một lần cuối trước khi vào lớp Linear
        # assert outputs.shape == (x.shape[0], self.n_qubits)

        logits = self.fc(outputs)
        return logits
