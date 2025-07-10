# quantum_circuits/circuit14_circuit.py (CẬP NHẬT)

import pennylane as qml
import torch
# Bỏ import config vì chúng ta sẽ truyền tham số vào
# from configs.config import N_QUBITS, N_LAYERS, QUANTUM_DEVICE 

def create_circuit14(n_qubits, n_layers, device_name="lightning.qubit"):
    """
    Tạo một qnode cho kiến trúc Circuit 14, với số qubit và số lớp tùy chỉnh.
    """
    dev = qml.device(device_name, wires=n_qubits, shots=None)
    
    # Số cổng CRX trong 1 lớp: (n_qubits - 1) kề nhau, 1 vòng, và 2 cái chéo
    # Đây là một giả định để mở rộng từ 4 qubit.
    # Với n_qubits=4, crx_per_layer = 3 + 1 + 2 = 6.
    # Với n_qubits=6, crx_per_layer = 5 + 1 + 2 = 8.
    crx_per_layer = (n_qubits - 1) + 1 + 2

    @qml.qnode(dev, interface='torch', diff_method='adjoint') 
    def circuit_14(inputs, weights, crx_weights):
        # inputs là vector đã được chuẩn hóa, có chiều dài 2^n_qubits
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        
        for l in range(n_layers):
            # Lớp các cổng xoay một qubit
            for qubit in range(n_qubits):
                qml.RX(weights[l, qubit, 0], wires=qubit)
                qml.RY(weights[l, qubit, 1], wires=qubit)
                qml.RZ(weights[l, qubit, 2], wires=qubit)
            
            # Lớp các cổng vướng víu CRX
            # `crx_weights` có shape [n_layers, crx_per_layer]
            crx_idx = 0
            # Các cổng kề nhau
            for qubit in range(n_qubits - 1):
                qml.CRX(crx_weights[l, crx_idx], wires=[qubit, qubit + 1])
                crx_idx += 1
            # Cổng vòng (cuối về đầu)
            qml.CRX(crx_weights[l, crx_idx], wires=[n_qubits - 1, 0])
            crx_idx += 1
            # Hai cổng chéo (ví dụ) để tăng khả năng vướng víu
            qml.CRX(crx_weights[l, crx_idx], wires=[0, n_qubits // 2])
            crx_idx += 1
            qml.CRX(crx_weights[l, crx_idx], wires=[1, (n_qubits // 2) + 1])
            crx_idx += 1

        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
        
    circuit_14_broadcasted = qml.transforms.broadcast_expand(circuit_14)

    return circuit_14_broadcasted, crx_per_layer