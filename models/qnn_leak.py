# models/qnn_leak.py
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QNN(nn.Module):
    """
    Lớp mô hình thay thế (substitute) linh hoạt cho toàn bộ QNN Zoo.
    Sử dụng kiến trúc lai: Pool + Linear -> Quantum Circuit -> Linear.
    Nhận các tham số động để xây dựng các mạch khác nhau.
    """
    def __init__(self, n_qubits, quantum_circuit, params_shape: dict, device):
        super(QNN, self).__init__()
        
        # Phần cổ điển
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.pre_net = nn.Linear(16, n_qubits)
        self.post_net = nn.Linear(n_qubits, 1)
        
        # Phần lượng tử
        self.quantum_circuit = quantum_circuit

        # Khởi tạo các tham số theo phân phối Gaussian
        self.q_weights = None
        self.q_crx_weights = None

        if 'weights' in params_shape and params_shape['weights'] is not None:
            init_data = np.random.normal(0, np.pi, params_shape['weights'])
            self.q_weights = nn.Parameter(torch.tensor(init_data, dtype=torch.float32, device=device))
        
        if 'crx_weights' in params_shape and params_shape['crx_weights'] is not None:
            init_data = np.random.normal(0, np.pi, params_shape['crx_weights'])
            self.q_crx_weights = nn.Parameter(torch.tensor(init_data, dtype=torch.float32, device=device))

    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, 16)
        x = torch.tanh(self.pre_net(x))

        # Mạch lượng tử nhận vào x và một dict các tham số
        outputs = self.quantum_circuit(x, self.q_weights, self.q_crx_weights)
        
        if isinstance(outputs, list):
             outputs = torch.stack(outputs)
        if len(outputs.shape) > 1 and outputs.shape[0] != x.shape[0] and outputs.shape[1] == x.shape[0]:
             outputs = outputs.t()

        return self.post_net(outputs)