 
import pennylane as qml
import torch
import numpy as np
from configs.config import N_QUBITS, N_LAYERS, QUANTUM_DEVICE

def create_transfer_learning_circuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, device_name=QUANTUM_DEVICE):
    dev = qml.device(device_name, wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_circuit(inputs, weights):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(inputs[i] * np.pi / 2, wires=i)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return quantum_circuit