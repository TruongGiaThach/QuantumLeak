import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from attacks.cloud_leak import CloudLeak
from attacks.quantum_leak import QuantumLeak
import numpy as np
from utils.training import evaluate_model_with_metrics
import torch

def ablation_study(victim_model, in_domain_loader, test_loader, out_of_domain_loader, query_budgets=[1500, 3000, 6000], 
                   architectures=['L1', 'L2', 'L3', 'A1', 'A2'], committee_numbers=[3, 5, 7], epochs=30,
                   n_qubits=4, device="cuda", save_path="./results", circuit_device="default.mixed"):
    """
    Thực hiện ablation study để đánh giá tác động của query_budget, architecture và committee_number.
    Tận dụng mô hình pre-trained và adversarial samples đã lưu.
    Args:
        in_domain_loader: DataLoader chứa ảnh in-domain.
        test_loader: DataLoader chứa dữ liệu kiểm thử.
        out_of_domain_loader: DataLoader chứa 3000 ảnh out-of-domain.
        query_budgets: Danh sách ngân sách truy vấn.
        architectures: Danh sách kiến trúc VQC.
        committee_numbers: Danh sách số lượng thành viên ensemble.
        epochs: Số epoch huấn luyện.
    Returns:
        results: Danh sách kết quả ablation study.
    """
    results = []
    

    # Chuẩn bị public_loader cho pre-training
    public_inputs = np.concatenate([inputs.cpu().numpy() for inputs, _ in out_of_domain_loader])[:3000]
    public_targets = np.concatenate([targets.cpu().numpy() for _, targets in out_of_domain_loader])[:3000]
    public_dataset = TensorDataset(
        torch.tensor(public_inputs, dtype=torch.float32),
        torch.tensor(public_targets, dtype=torch.float32)
    )
    public_loader = DataLoader(public_dataset, batch_size=8, shuffle=True)

    for query_budget in query_budgets:
        for architecture in architectures:
            for n_committee in committee_numbers:
                for scheme in ['Single-N', 'Single-H', 'Ens-N', 'Ens-H']:
                    loss_type = 'nll' if 'N' in scheme else 'huber'
                    add_noise = True  # Mặc định có nhiễu, có thể mở rộng để thử nghiệm không nhiễu
                    if 'Single' in scheme:
                        cloud_leak = CloudLeak(
                            victim_model, n_qubits, query_budget, device, save_path, circuit_device
                        )
                        print(f"Training CloudLeak {scheme} (architecture={architecture}, query_budget={query_budget})...")
                        model, _ = cloud_leak.train(
                            in_domain_loader, out_of_domain_loader, n_epochs=20,
                            batch_size=8, architecture=architecture, loss_type=loss_type, add_noise=add_noise
                        )
                        metrics = cloud_leak.evaluate(model, test_loader)
                    else:
                        print(f"Training QuantumLeak {scheme} (architecture={architecture}, committee={n_committee}, query_budget={query_budget})...")
                        quantum_leak = QuantumLeak(
                            victim_model, n_qubits, query_budget, device, save_path, circuit_device
                        )
                        ensemble_models, _ = quantum_leak.train(
                            in_domain_loader, out_of_domain_loader, n_epochs=epochs,
                            batch_size=8, architecture=architecture, loss_type=loss_type, add_noise=add_noise
                        )
                        metrics = quantum_leak.evaluate(ensemble_models, test_loader)

                    results.append({
                        'Query Budget': query_budget,
                        'Architecture': architecture,
                        'Committee': n_committee,
                        'Scheme': scheme,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1': metrics['f1']
                    })
                    print(f"Query Budget: {query_budget}, Architecture: {architecture}, Committee: {n_committee}, "
                          f"Scheme: {scheme}, Accuracy: {metrics['accuracy']:.2f}%, "
                          f"Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1']:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_path, f'ablation_results.csv'), index=False)
    print(f"Kết quả ablation study đã lưu tại: {save_path}/ablation_results.csv")
    return results

def plot_ablation_results(results_df, save_path):
    plt.figure(figsize=(10, 6))
    df_fig4 = results_df[(results_df['Query Budget'] == 6000) & (results_df['Architecture'] == 'L2') & (results_df['Committee'] == 5)]
    schemes = df_fig4['Scheme'].values
    accuracies = df_fig4['Accuracy'].values
    plt.bar(schemes, accuracies, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Attack Scheme')
    plt.ylabel('Accuracy (%)')
    plt.title('Figure 4: Comparison of Attack Schemes')
    plt.savefig(os.path.join(save_path, 'figure4.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    df_fig5 = results_df[(results_df['Query Budget'] == 6000) & (results_df['Scheme'] == 'Ens-H') & (results_df['Committee'] == 5)]
    architectures = df_fig5['Architecture'].values
    accuracies = df_fig5['Accuracy'].values
    plt.bar(architectures, accuracies, color='purple')
    plt.xlabel('VQC Architecture')
    plt.ylabel('Accuracy (%)')
    plt.title('Figure 5: Impact of VQC Ansatz')
    plt.savefig(os.path.join(save_path, 'figure5.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    df_fig6 = results_df[(results_df['Query Budget'] == 6000) & (results_df['Architecture'] == 'L2') & (results_df['Scheme'] == 'Ens-H')]
    committees = df_fig6['Committee'].values
    accuracies = df_fig6['Accuracy'].values
    plt.plot(committees, accuracies, marker='o', label='Ens-H')
    plt.xlabel('Number of Committee Members')
    plt.ylabel('Accuracy (%)')
    plt.title('Figure 6: Impact of Committee Number')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'figure6.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    for arch in ['L1', 'L2', 'L3']:
        df_fig7 = results_df[(results_df['Architecture'] == arch) & (results_df['Scheme'] == 'Ens-H') & (results_df['Committee'] == 5)]
        query_budgets = df_fig7['Query Budget'].values
        accuracies = df_fig7['Accuracy'].values
        plt.plot(query_budgets, accuracies, marker='o', label=f'{arch}')
    plt.xlabel('Query Budget')
    plt.ylabel('Accuracy (%)')
    plt.title('Figure 7: Performance with Different VQC Layers')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'figure7.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    for arch in ['L1', 'L2', 'L3']:
        df_fig8 = results_df[(results_df['Architecture'] == arch) & (results_df['Scheme'] == 'Ens-H') & (results_df['Committee'] == 5)]
        query_budgets = df_fig8['Query Budget'].values
        accuracies = df_fig8['Accuracy'].values
        plt.plot(query_budgets, accuracies, marker='o', label=f'{arch}')
    plt.xlabel('Query Budget')
    plt.ylabel('Accuracy (%)')
    plt.title('Figure 8: Performance with Query Budgets')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'figure8.png'))
    plt.close()