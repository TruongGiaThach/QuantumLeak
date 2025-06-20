 
import pennylane as qml
import numpy as np
from configs.config import N_LAYERS, N_QUBITS, QUANTUM_DEVICE

def create_quanv_circuit():
    try:
        dev = qml.device(QUANTUM_DEVICE, wires=4)
    except:
        dev = qml.device("default.qubit", wires=4)
    rand_params = np.random.uniform(high=2 * np.pi, size=(N_LAYERS, 4))

    @qml.qnode(dev)
    def circuit(phi):
        for j in range(4):
            qml.RY(np.pi * phi[j], wires=j)
        qml.templates.RandomLayers(rand_params, wires=list(range(4)))
        return [qml.expval(qml.PauliZ(j)) for j in range(4)]
    return circuit