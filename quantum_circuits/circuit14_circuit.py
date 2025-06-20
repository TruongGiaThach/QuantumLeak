
 
import pennylane as qml
import torch
from configs.config import N_QUBITS, N_LAYERS, QUANTUM_DEVICE

def create_circuit14(n_qubits=N_QUBITS, n_layers=N_LAYERS, device_name=QUANTUM_DEVICE):
    dev = qml.device(device_name, wires=n_qubits, shots=None)
    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit_14(inputs, weights, crx_weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.)
        for l in range(n_layers):
            for qubit in range(n_qubits):
                qml.RX(weights[l, qubit, 0], wires=qubit)
                qml.RY(weights[l, qubit, 1], wires=qubit)
                qml.RZ(weights[l, qubit, 2], wires=qubit)
            crx_idx = 0
            for qubit in range(n_qubits - 1):
                qml.CRX(crx_weights[l, crx_idx], wires=[qubit, qubit + 1])
                crx_idx += 1
            qml.CRX(crx_weights[l, crx_idx], wires=[n_qubits - 1, 0])
            crx_idx += 1
            for qubit in [0, 1]:
                target = (qubit + 3) % n_qubits
                qml.CRX(crx_weights[l, crx_idx], wires=[qubit, target])
                crx_idx += 1
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    return circuit_14
