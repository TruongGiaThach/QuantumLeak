import pennylane as qml
from configs.config import N_QUBITS, N_LAYERS, QUANTUM_DEVICE

def create_quantum_circuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, device_name=QUANTUM_DEVICE):
    """
    Tạo một mạch lượng tử cho BasicQNN theo mô tả:
    Angle encoding vào RX, các tham số học được trong RY, vướng víu bằng CZ.
    """
    dev = qml.device(device_name, wires=n_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        """
        Args:
            inputs (torch.Tensor): Vector đầu vào có shape [n_qubits].
            weights (torch.Tensor): Vector trọng số có shape [n_layers, n_qubits].
        """
        for layer in range(n_layers):
            # Lớp mã hóa dữ liệu (Angle Encoding)
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            
            # Lớp biến phân (học được)
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            
            # Lớp vướng víu
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit