# quantum_circuits/substitution_circuit.py
import pennylane as qml
import torch

def create_substitute_circuit(n_qubits, n_layers, device_name="lightning.qubit"):
    """
    Tạo qnode cho kiến trúc VQC, sử dụng Angle Encoding hiệu quả.
    """
    dev = qml.device(device_name, wires=n_qubits, shots=None)
    
    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit(inputs, weights, crx_weights):
        # inputs: vector đặc trưng có chiều dài n_qubits
        # weights: tham số cho các cổng RZ, RY. Shape: [n_layers, n_qubits, 3]
        # crx_weights: tham số cho các cổng CRX. Shape: [n_layers, n_qubits]

        # --- ANGLE ENCODING ---
        # Mã hóa mỗi giá trị của `inputs` vào góc quay của cổng RY.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        # -----------------------------------------------------------------
        
        for l in range(n_layers):
            # Lớp các cổng xoay
            for i in range(n_qubits):
                qml.RZ(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
            
            # Lớp vướng víu CRX vòng
            for i in range(n_qubits):
                qml.CRX(crx_weights[l, i], wires=[i, (i + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
        
    params_shape = {
        'weights': (n_layers, n_qubits, 3),
        'crx_weights': (n_layers, n_qubits)
    }

    return qml.transforms.broadcast_expand(circuit), params_shape