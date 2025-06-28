import torch
import pandas as pd
import numpy as np
from configs.config import *
from data.data_pipeline import CIFAR10DataPipeline, get_cq_dataloaders, create_out_of_domain_loader
from models.quanv_model import QuanvModel
from models.basic_qnn import BasicQNN
from models.circuit14 import QuantumLayer, PureQuantumCircuit14
from models.transfer_learning import CQTransferLearningModel
from quantum_circuits.quanv_circuit import create_quanv_circuit
from quantum_circuits.basic_qnn_circuit import create_quantum_circuit
from quantum_circuits.circuit14_circuit import create_circuit14
from quantum_circuits.transfer_learning_circuit import create_transfer_learning_circuit
from utils.training import train_model, evaluate_model_with_metrics
from utils.visualization import plot_training_history, plot_quanv_visualization, plot_model_comparison
from attacks.quantum_leak import CloudLeak, QuantumLeak, HuberLoss
from torch.utils.data import TensorDataset, DataLoader
def print_best_metrics(model_name, best_metrics):
    """Hàm tiện ích để in các chỉ số của mô hình tốt nhất."""
    print(f"\n--- {model_name} Victim Model (Best on Val Set) ---")
    print(f"Accuracy: {best_metrics.get('accuracy', 0):.2f}%")
    print(f"Precision: {best_metrics.get('precision', 0):.2f}")
    print(f"Recall: {best_metrics.get('recall', 0):.2f}")
    print(f"F1-Score: {best_metrics.get('f1', 0):.2f}")
    print("--------------------------------------------------")
def run_quanv_experiment():
    print("\n--- Running Quanv Experiment ---")
    save_dir = os.path.join(SAVE_PATH, "quanv")
    pipeline = CIFAR10DataPipeline(n_train=N_TRAIN, n_test=N_TEST, save_path=SAVE_PATH)
    train_loader, test_loader = get_cq_dataloaders(pipeline, batch_size=BATCH_SIZE, device=torch.device(DEVICE))
    train_images = []
    train_labels = []
    for images, labels in train_loader:
        train_images.append(images.cpu().numpy())
        train_labels.append(labels.cpu().numpy())
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    test_images = []
    test_labels = []
    for images, labels in test_loader:
        test_images.append(images.cpu().numpy())
        test_labels.append(labels.cpu().numpy())
    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)
    train_images = train_images.transpose(0, 2, 3, 1)
    test_images = test_images.transpose(0, 2, 3, 1)
    quanv_circuit = create_quanv_circuit()
    q_train_images, q_test_images = pipeline.preprocess_quanv(train_images, test_images, quanv_circuit, load_from_drive=True)
    q_train_images = torch.tensor(q_train_images, dtype=torch.float32).clone().detach().permute(0, 3, 1, 2).to(DEVICE)
    q_test_images = torch.tensor(q_test_images, dtype=torch.float32).clone().detach().permute(0, 3, 1, 2).to(DEVICE)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).clone().detach().view(-1, 1).to(DEVICE)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).clone().detach().view(-1, 1).to(DEVICE)
    train_dataset = TensorDataset(q_train_images, train_labels)
    test_dataset = TensorDataset(q_test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    model = QuanvModel().to(DEVICE)
    criterion = torch.nn.BCELoss()
    history, best_metrics = train_model(model, train_loader, test_loader, torch.device(DEVICE), criterion, N_EPOCHS, LEARNING_RATE, save_dir)
    print_best_metrics("QuanvNN", best_metrics)
    plot_training_history(history, save_dir, "Quanv Training History")
    plot_quanv_visualization(train_images, q_train_images.cpu(), save_dir)

def run_basic_qnn_experiment():
    print("\n--- Running Basic QNN Experiment ---")
    save_dir = os.path.join(SAVE_PATH, "basic_qnn")
    pipeline = CIFAR10DataPipeline(n_train=N_TRAIN, n_test=N_TEST, save_path=save_dir)
    train_loader, test_loader = get_cq_dataloaders(pipeline, batch_size=BATCH_SIZE, device=torch.device(DEVICE))
    quantum_circuit = create_quantum_circuit()
    model = BasicQNN(N_QUBITS, quantum_circuit).to(DEVICE)
    criterion = torch.nn.BCELoss() 

    history, best_metrics = train_model(model, train_loader, test_loader, torch.device(DEVICE), criterion, N_EPOCHS, LEARNING_RATE, save_dir)
    
    print_best_metrics("BasicQNN", best_metrics)
    plot_training_history(history, save_dir, "Basic QNN Training History")

def run_circuit14_experiment():
    print("\n--- Running Hybrid Circuit 14 Experiment ---")
    save_dir = os.path.join(SAVE_PATH, "circuit14")
    pipeline = CIFAR10DataPipeline(n_train=N_TRAIN, n_test=N_TEST, save_path=save_dir)
    train_loader, test_loader = get_cq_dataloaders(pipeline, batch_size=BATCH_SIZE, device=torch.device(DEVICE))
    
    model = QuantumLayer(N_QUBITS, N_LAYERS, device_name=QUANTUM_DEVICE).to(DEVICE)
    criterion = torch.nn.BCELoss() 

    history, best_metrics = train_model(model, train_loader, test_loader, torch.device(DEVICE), criterion, N_EPOCHS, LEARNING_RATE, save_dir)
    
    print_best_metrics("QuantumLayer (Circuit14)", best_metrics)
    plot_training_history(history, save_dir, "Circuit14 Training History")
def run_pure_qnn_circuit14_experiment():
    print("\n--- Running Pure QNN Circuit 14 Experiment ---")
    save_dir = os.path.join(SAVE_PATH, "pure_qnn_circuit14")


    N_QUBITS_PURE = 6 # Cần 2^6 = 64 để mã hóa ảnh 8x8
    N_LAYERS_PURE = 4
    
    pipeline = CIFAR10DataPipeline(n_train=N_TRAIN, n_test=N_TEST, save_path=SAVE_PATH)
    train_loader, test_loader = get_cq_dataloaders(pipeline, batch_size=BATCH_SIZE, device=torch.device(DEVICE))

    model = PureQuantumCircuit14(
        n_qubits=N_QUBITS_PURE, 
        n_layers=N_LAYERS_PURE, 
        device_name=QUANTUM_DEVICE
    ).to(DEVICE)
    
    criterion = torch.nn.BCELoss()
    
    history, best_metrics = train_model(
        model, train_loader, test_loader, 
        torch.device(DEVICE), criterion, N_EPOCHS, 
        LEARNING_RATE, save_dir
    )
    
    print_best_metrics("Pure QNN Circuit 14", best_metrics)
    plot_training_history(history, save_dir, "Pure QNN Circuit 14 Training History")

def run_transfer_learning_experiment():
    print("\n--- Running Transfer Learning Experiment ---")
    save_dir = os.path.join(SAVE_PATH, "transfer_learning")
    pipeline = CIFAR10DataPipeline(n_train=N_TRAIN, n_test=N_TEST, save_path=SAVE_PATH)
    train_loader, test_loader = get_cq_dataloaders(pipeline, batch_size=BATCH_SIZE, device=torch.device(DEVICE))
    model = CQTransferLearningModel(N_QUBITS, N_LAYERS).to(DEVICE)
    criterion = torch.nn.BCELoss() 

    history, best_metrics = train_model(model, train_loader, test_loader, torch.device(DEVICE), criterion, N_EPOCHS, LEARNING_RATE, save_dir)
    
    print_best_metrics("CQ Transfer Learning", best_metrics)
    plot_training_history(history, save_dir, "Transfer Learning Training History")

def run_leak_experiment(model_type="basic_qnn"):
    pipeline = CIFAR10DataPipeline(n_train=N_LEAK_TRAIN, n_test=N_LEAK_TEST, save_path=SAVE_PATH)
    train_loader, test_loader = get_cq_dataloaders(pipeline, batch_size=LEAK_BATCH_SIZE, device=torch.device(DEVICE))
    out_of_domain_loader = create_out_of_domain_loader(pipeline, batch_size=LEAK_BATCH_SIZE, n_samples=3000)
    if model_type == "basic_qnn":
        quantum_circuit = create_quantum_circuit()
        victim_model = BasicQNN(N_LEAK_QUBITS, quantum_circuit).to(DEVICE)
        victim_model.load_state_dict(torch.load(os.path.join(SAVE_PATH + "/basic_qnn", "best_model.pth")))
    elif model_type == "circuit14":
        victim_model = QuantumLayer(N_LEAK_QUBITS, N_LEAK_LAYERS, QUANTUM_DEVICE).to(DEVICE)
        victim_model.load_state_dict(torch.load(os.path.join(SAVE_PATH + "/circuit14", "best_model.pth")))
    elif model_type == "transfer_learning":
        victim_model = CQTransferLearningModel(N_LEAK_QUBITS, N_LEAK_LAYERS).to(DEVICE)
        victim_model.load_state_dict(torch.load(os.path.join(SAVE_PATH + "/transfer_learning", "best_model.pth")))
    elif model_type == "pure_circuit14":
        print("Loading PureQuantumCircuit14 as the victim model...")
        # Cần các hằng số đúng cho mô hình này
        N_QUBITS_PURE = 6 
        N_LAYERS_PURE = 4
        victim_model = PureQuantumCircuit14(N_QUBITS_PURE, N_LAYERS_PURE, device_name=QUANTUM_DEVICE).to(DEVICE)
        victim_model.load_state_dict(torch.load(os.path.join(SAVE_PATH, "pure_qnn_circuit14", "best_model.pth")))
    else:
        raise ValueError("Invalid model type")
    victim_model.eval()
    cloud_leak = CloudLeak(
        victim_model,
        n_qubits=N_LEAK_QUBITS,
        query_budget=LEAK_QUERY_BUDGET,
        device=DEVICE,
        save_path=QUANTUM_LEAK_SAVE_PATH,
        circuit_device=QUANTUM_DEVICE
    )
    quantum_leak = QuantumLeak(
        victim_model,
        n_qubits=N_LEAK_QUBITS,
        query_budget=LEAK_QUERY_BUDGET,
        n_committee=LEAK_N_COMMITTEE,
        device=DEVICE,
        save_path=QUANTUM_LEAK_SAVE_PATH,
        circuit_device=QUANTUM_DEVICE
    )
    query_dataset = cloud_leak.query_victim(train_loader, n_rounds=3)
    query_outputs = np.concatenate([item[1][:, 1] for item in query_dataset])
    print(f"Query dataset noise check - Mean probability: {query_outputs.mean():.4f}, Std: {query_outputs.std():.4f}")
    print("Training CloudLeak Single-N...")
    single_n_model, single_n_history = cloud_leak.train(
        query_dataset, out_of_domain_loader, n_epochs=2, batch_size=LEAK_BATCH_SIZE, architecture='L2', loss_type='nll'
    )
    single_n_metrics = cloud_leak.evaluate(single_n_model, test_loader)
    print(f"CloudLeak Single-N Metrics: Accuracy: {single_n_metrics['accuracy']:.2f}%, "
          f"Precision: {single_n_metrics['precision']:.2f}, Recall: {single_n_metrics['recall']:.2f}, F1: {single_n_metrics['f1']:.2f}")
    print("Training CloudLeak Single-H...")
    single_h_model, single_h_history = cloud_leak.train(
        query_dataset, out_of_domain_loader, n_epochs=2, batch_size=LEAK_BATCH_SIZE, architecture='L2', loss_type='huber'
    )
    single_h_metrics = cloud_leak.evaluate(single_h_model, test_loader)
    print(f"CloudLeak Single-H Metrics: Accuracy: {single_h_metrics['accuracy']:.2f}%, "
          f"Precision: {single_h_metrics['precision']:.2f}, Recall: {single_h_metrics['recall']:.2f}, F1: {single_h_metrics['f1']:.2f}")
    print("Training QuantumLeak Ens-N...")
    ens_n_models, ens_n_history = quantum_leak.train(
        query_dataset, out_of_domain_loader, n_epochs=2, batch_size=LEAK_BATCH_SIZE, architecture='L2', loss_type='nll'
    )
    ens_n_metrics = quantum_leak.evaluate(test_loader)
    print(f"QuantumLeak Ens-N Metrics: Accuracy: {ens_n_metrics['accuracy']:.2f}%, "
          f"Precision: {ens_n_metrics['precision']:.2f}, Recall: {ens_n_metrics['recall']:.2f}, F1: {ens_n_metrics['f1']:.2f}")
    print("Training QuantumLeak Ens-H...")
    ens_h_models, ens_h_history = quantum_leak.train(
        query_dataset, out_of_domain_loader, n_epochs=2, batch_size=LEAK_BATCH_SIZE, architecture='L2', loss_type='huber'
    )
    ens_h_metrics = quantum_leak.evaluate(test_loader)
    print(f"QuantumLeak Ens-H Metrics: Accuracy: {ens_h_metrics['accuracy']:.2f}%, "
          f"Precision: {ens_h_metrics['precision']:.2f}, Recall: {ens_h_metrics['recall']:.2f}, F1: {ens_h_metrics['f1']:.2f}")
    victim_criterion = HuberLoss(delta=0.5)
    victim_metrics = evaluate_model_with_metrics(victim_model, test_loader, torch.device(DEVICE), victim_criterion)
    print(f"Victim QNN Metrics: Accuracy: {victim_metrics['accuracy']:.2f}%, "
          f"Precision: {victim_metrics['precision']:.2f}, Recall: {victim_metrics['recall']:.2f}, F1: {victim_metrics['f1']:.2f}")
    table2_data = {
        'Scheme': ['Single-N', 'Single-H', 'Ens-N', 'Ens-H'],
        'Accuracy': [single_n_metrics['accuracy'], single_h_metrics['accuracy'], ens_n_metrics['accuracy'], ens_h_metrics['accuracy']],
        'Precision': [single_n_metrics['precision'], single_h_metrics['precision'], ens_n_metrics['precision'], ens_h_metrics['precision']],
        'Recall': [single_n_metrics['recall'], single_h_metrics['recall'], ens_n_metrics['recall'], ens_h_metrics['recall']],
        'F1': [single_n_metrics['f1'], single_h_metrics['f1'], ens_n_metrics['f1'], ens_h_metrics['f1']]
    }
    table2_df = pd.DataFrame(table2_data)
    table2_df.to_csv(os.path.join(QUANTUM_LEAK_SAVE_PATH, 'table2.csv'), index=False)
    print("\nTable II:")
    print(table2_df)
    table3_data = {
        'Method': ['CloudLeak', 'QuantumLeak'],
        'In-domain Images': [6000, 6000],
        'Out-of-domain Images': [512, 0],
        'Query Rounds': [3, 3],
        'VQC Architecture': ['L2', 'L2']
    }
    table3_df = pd.DataFrame(table3_data)
    table3_df.to_csv(os.path.join(QUANTUM_LEAK_SAVE_PATH, 'table3.csv'), index=False)
    print("\nTable III:")
    print(table3_df)
    table4_data = {
        'Model': ['Victim QNN', 'CloudLeak Single-N', 'CloudLeak Single-H', 'QuantumLeak Ens-N', 'QuantumLeak Ens-H'],
        'Accuracy (%)': [victim_metrics['accuracy'], single_n_metrics['accuracy'], single_h_metrics['accuracy'], ens_n_metrics['accuracy'], ens_h_metrics['accuracy']],
        'Precision': [victim_metrics['precision'], single_n_metrics['precision'], single_h_metrics['precision'], ens_n_metrics['precision'], ens_h_metrics['precision']],
        'Recall': [victim_metrics['recall'], single_n_metrics['recall'], single_h_metrics['recall'], ens_n_metrics['recall'], ens_h_metrics['recall']],
        'F1': [victim_metrics['f1'], single_n_metrics['f1'], single_h_metrics['f1'], ens_n_metrics['f1'], ens_h_metrics['f1']],
        'Notes': [
            'Original model, no query noise',
            'Trained with noisy queries (SPAM: 0.54%, 1Q: 0.177%, 2Q: 2.87%, Crosstalk: 20%)',
            'Trained with noisy queries, Huber loss',
            'Ensemble of 5 models, noisy queries, BCE loss',
            'Ensemble of 5 models, noisy queries, Huber loss'
        ]
    }
    table4_df = pd.DataFrame(table4_data)
    table4_df.to_csv(os.path.join(QUANTUM_LEAK_SAVE_PATH, 'table4.csv'), index=False)
    print("\nTable IV: Comparison of Victim QNN, CloudLeak, and QuantumLeak")
    print(table4_df)
    print("\nRunning ablation study for Figure 5, 6, 7, 8...")
    results = quantum_leak.ablation_study(
        train_loader, test_loader, out_of_domain_loader,
        query_budgets=[1500, 3000, 6000],
        architectures=['L1', 'L2', 'L3', 'A1', 'A2'],
        committee_numbers=[3, 5, 7],
        epochs=N_LEAK_EPOCHS
    )
    results_df = pd.read_csv(os.path.join(QUANTUM_LEAK_SAVE_PATH, 'ablation_results.csv'))
    quantum_leak.plot_ablation_results(results_df)
    plot_model_comparison(
        [victim_metrics['accuracy'], single_h_metrics['accuracy'], ens_h_metrics['accuracy']],
        ['Victim QNN', 'CloudLeak Single-H', 'QuantumLeak Ens-H'],
        QUANTUM_LEAK_SAVE_PATH,
        f"{model_type.capitalize()} Model Comparison"
    )

if __name__ == "__main__":
    # run_circuit14_experiment()
    # run_pure_qnn_circuit14_experiment()
    # run_basic_qnn_experiment()
    # run_transfer_learning_experiment()
    # run_quanv_experiment()
    run_leak_experiment("basic_qnn")
    run_leak_experiment("circuit14")
    run_leak_experiment("transfer_learning")
    run_leak_experiment("pure_circuit14")