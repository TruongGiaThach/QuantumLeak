 
import pennylane as qml
import torch
from configs.config import N_QUBITS, N_LAYERS, QUANTUM_DEVICE

def create_quantum_circuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, device_name=QUANTUM_DEVICE):
    dev = qml.device(device_name, wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(inputs[..., i], wires=i)
                qml.RY(weights[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit