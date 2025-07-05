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

        # Lớp bọc cổ điển không đổi
        self.pre_net = nn.Linear(512, n_qubits)
        self.post_net = nn.Linear(n_qubits, 1)
        
        # Trọng số cho mạch lượng tử, giữ nguyên như code gốc của bạn
        # nhưng ta sẽ cần 2 bộ trọng số cho mạch mới
        self.weights_ry = nn.Parameter(torch.randn(n_layers, n_qubits, requires_grad=True))
        self.weights_rz = nn.Parameter(torch.randn(n_layers, n_qubits, requires_grad=True))

        self.dev = qml.device(device_name, wires=n_qubits)
        self.quantum_circuit = self._create_quantum_circuit()

    def _create_quantum_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def quantum_circuit(inputs, weights_ry, weights_rz):
            # 1. Giữ Angle Encoding
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')

            # Các lớp biến phân (Variational Layers)
            for layer in range(self.n_layers):
                # Thêm các cổng RZ để tăng năng lực
                for i in range(self.n_qubits):
                    qml.RY(weights_ry[layer, i], wires=i)
                    qml.RZ(weights_rz[layer, i], wires=i)
                
                # --- THAY ĐỔI 1: MẠCH LƯỢNG TỬ MẠNH HƠN ---
                # Cấu trúc vướng víu vòng (ring entanglement) thay vì chuỗi
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                # ---------------------------------------------
            
            # Đo lường tất cả các qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return quantum_circuit

    def forward(self, x):
        # Lớp cổ điển tiền xử lý
        x = self.pre_net(x)

        # Giữ nguyên vòng lặp for để đảm bảo ổn định
        q_out_list = []
        for i in range(x.size(0)):
            q_res = self.quantum_circuit(x[i], self.weights_ry, self.weights_rz)
            # q_res là một list các giá trị đo, cần stack chúng thành tensor
            q_out_list.append(torch.stack(q_res))
        
        q_out = torch.stack(q_out_list)
        q_out = q_out.float()
        
        # --- THAY ĐỔI 2: TRẢ VỀ LOGITS ---
        # Bỏ qua lớp tanh, đưa thẳng kết quả đo vào lớp linear cuối
        logits = self.post_net(q_out)
        return logits
        # --------------------------------

class CQTransferLearningModel(nn.Module):
    def __init__(self, n_qubits, n_layers, device_name="default.qubit"):
        super(CQTransferLearningModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Thay đổi lớp conv đầu tiên để nhận ảnh 1 kênh
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        self.resnet.fc = nn.Identity()

        # --- GIỮ NGUYÊN TRIẾT LÝ: ĐÓNG BĂNG TOÀN BỘ RESNET ---
        for param in self.resnet.parameters():
            param.requires_grad = False
        # ----------------------------------------------------

        self.quantum_net = DressedQuantumCircuit(n_qubits, n_layers, device_name=device_name)

    def forward(self, x):
        # resnet chỉ hoạt động như một bộ trích xuất đặc trưng cố định
        features = self.resnet(x)
        return self.quantum_net(features)