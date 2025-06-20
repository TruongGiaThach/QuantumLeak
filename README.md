# QNN on CIFAR-10

This project implements a Quantum Neural Network (QNN) for binary classification on the CIFAR-10 dataset (airplane vs. automobile) using PennyLane and PyTorch. It includes multiple QNN architectures (Quanv, Basic QNN, Circuit14, Transfer Learning) and a QuantumLeak attack to evaluate model robustness.

## Project Structure

-   `configs/`: Configuration file for hyperparameters and paths.
-   `data/`: Data loading and preprocessing pipeline.
-   `models/`: QNN model definitions.
-   `quantum_circuits/`: Quantum circuit implementations.
-   `utils/`: Training and visualization utilities.
-   `attacks/`: QuantumLeak attack implementation.
-   `main.py`: Main script to run experiments.
-   `requirements.txt`: Dependencies.
-   `saved_models/`: Directory for saved models and preprocessed data.
-   `results/`: Directory for QuantumLeak results and plots.

## Setup

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Ensure CUDA is available for GPU acceleration (optional).

## Usage

Run all experiments (Quanv, Basic QNN, Circuit14, Transfer Learning, and QuantumLeak attacks):

```bash
python main.py
```

## Experiments

-   **Quanv**: Quantum convolution-based model with classical CNN.
-   **Basic QNN**: Quantum neural network with variational quantum circuit.
-   **Circuit14**: VQC-based model with amplitude embedding and CRX gates.
-   **Transfer Learning**: Hybrid model using ResNet18 and a quantum circuit.
-   **QuantumLeak**: Attack to extract model behavior using noisy queries.

## Results

-   Models are saved in `saved_models/`.
-   QuantumLeak results (tables and figures) are saved in `results/`.
-   Training history and visualizations are generated for each experiment.

## Notes

-   Set `PREPROCESS = False` in `config.py` to load preprocessed quantum data.
-   Adjust hyperparameters in `config.py` as needed.
-   Quantum device defaults to `lightning.gpu` if available, else `default.qubit`.
