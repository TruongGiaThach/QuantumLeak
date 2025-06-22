import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from models.qnn_leak import QNN
from utils.visualization import plot_model_comparison
from utils.training import evaluate_model_with_metrics
from abc import ABC, abstractmethod

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, outputs, targets):
        error = outputs - targets
        is_small_error = torch.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * torch.abs(error) - 0.5 * self.delta ** 2
        loss = torch.where(is_small_error, squared_loss, linear_loss)
        return loss.mean()

class ModelExtraction(ABC):
    def __init__(self, victim_model, n_qubits=4, query_budget=6000, device="cuda", save_path="./results", circuit_device="default.mixed"):
        self.victim_model = victim_model
        self.n_qubits = n_qubits
        self.query_budget = query_budget
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.circuit_device = circuit_device
        self.qnn_zoo = self._create_qnn_zoo()

    def _create_quantum_circuit(self, n_layers):
        dev = qml.device(self.circuit_device, wires=self.n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            for layer in range(n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer * self.n_qubits + i], wires=i)
                for i in range(self.n_qubits):
                    qml.CZ(wires=[i, (i + 1) % self.n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def _create_a1_circuit(self):
        dev = qml.device(self.circuit_device, wires=self.n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            param_idx = 0
            for _ in range(2):
                for i in range(self.n_qubits):
                    qml.RX(weights[param_idx], wires=i)
                    param_idx += 1
                    qml.RY(weights[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(weights[param_idx], wires=i)
                    param_idx += 1
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def _create_a2_circuit(self):
        dev = qml.device(self.circuit_device, wires=self.n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            param_idx = 0
            for _ in range(2):
                qml.RX(weights[param_idx], wires=0)
                param_idx += 1
                qml.RY(weights[param_idx], wires=1)
                param_idx += 1
                qml.RZ(weights[param_idx], wires=2)
                param_idx += 1
                qml.RX(weights[param_idx], wires=3)
                param_idx += 1
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def _create_qnn_zoo(self):
        qnn_zoo = []
        for n_layers in [1, 2, 3]:
            circuit = self._create_quantum_circuit(n_layers)
            model = QNN(self.n_qubits, circuit, n_layers).to(self.device)
            model.q_params = nn.Parameter(
                torch.tensor(np.random.normal(0, np.pi, (n_layers * self.n_qubits,)),
                             dtype=torch.float32, device=self.device, requires_grad=True)
            )
            qnn_zoo.append((f'L{n_layers}', model))
        circuit_a1 = self._create_a1_circuit()
        model_a1 = QNN(self.n_qubits, circuit_a1, n_layers=6).to(self.device)
        model_a1.q_params = nn.Parameter(
            torch.tensor(np.random.normal(0, np.pi, (12 * 2,)),
                         dtype=torch.float32, device=self.device, requires_grad=True)
        )
        qnn_zoo.append(('A1', model_a1))
        circuit_a2 = self._create_a2_circuit()
        model_a2 = QNN(self.n_qubits, circuit_a2, n_layers=4).to(self.device)
        model_a2.q_params = nn.Parameter(
            torch.tensor(np.random.normal(0, np.pi, (4 * 2,)),
                         dtype=torch.float32, device=self.device, requires_grad=True)
        )
        qnn_zoo.append(('A2', model_a2))
        return qnn_zoo

    def qnnaas_predict(self, model, images, device):
        model.eval()
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            prob_class_1 = outputs
            prob_class_0 = 1 - prob_class_1
            prob_vectors = torch.cat([prob_class_0, prob_class_1], dim=1)
        return prob_vectors

    def query_victim(self, data_loader, n_rounds=3):
        self.victim_model.eval()
        query_dataset = []
        samples_per_round = self.query_budget // n_rounds
        spam_noise = 0.0054
        gate_1q_noise = 0.00177
        gate_2q_noise = 0.0287
        crosstalk_noise = 0.2
        with torch.no_grad():
            for round in range(n_rounds):
                samples_collected = 0
                for inputs, _ in data_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.qnnaas_predict(self.victim_model, inputs, self.device)
                    noise_scale = spam_noise + gate_1q_noise + gate_2q_noise + crosstalk_noise
                    noise = torch.randn_like(outputs) * noise_scale
                    noisy_outputs = outputs + noise
                    noisy_outputs = torch.clamp(noisy_outputs, 0, 1)
                    query_dataset.append((inputs.cpu().numpy(), noisy_outputs.cpu().numpy()))
                    samples_collected += inputs.size(0)
                    if samples_collected >= samples_per_round:
                        break
                print(f"Completed query round {round + 1}/{n_rounds}")
        return query_dataset

    def generate_adversarial_samples(self, substitute_model, inputs, epsilon=0.1):
        """
        Tạo mẫu đối kháng dựa trên mô hình thay thế (substitute_model).
        Phương pháp: Tìm các mẫu gần ranh giới quyết định và thêm nhiễu.
        Hàm này không dùng nhãn.
        """
        substitute_model.eval()
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Sử dụng mô hình thay thế để tìm các điểm "mong manh"
            outputs = substitute_model(inputs_tensor) # Giả sử model trả về logits
            probs = torch.sigmoid(outputs).squeeze()

            # Tìm các mẫu gần ranh giới quyết định (0.5)
            boundary_mask = (probs >= 0.4) & (probs <= 0.6)
            
            # Tạo một bản sao để không thay đổi tensor gốc
            adv_inputs_tensor = inputs_tensor.clone()

            if boundary_mask.any():
                # Chỉ thêm nhiễu vào các mẫu gần ranh giới
                boundary_inputs = adv_inputs_tensor[boundary_mask]
                noise = epsilon * torch.randn_like(boundary_inputs)
                perturbed_inputs = boundary_inputs + noise
                # Giữ các giá trị trong khoảng hợp lệ của ảnh
                perturbed_inputs = torch.clamp(perturbed_inputs, -1, 1) 
                adv_inputs_tensor[boundary_mask] = perturbed_inputs
            
        return adv_inputs_tensor.cpu().numpy()

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

class CloudLeak(ModelExtraction):
    def pretrain(self, public_loader, n_epochs=10, batch_size=8, architecture='L2', save=True):
        criterion = nn.BCEWithLogitsLoss()
        architecture_map = {name: model for name, model in self.qnn_zoo}
        model = architecture_map[architecture]
        model.q_params = nn.Parameter(
            torch.normal(mean=0, std=np.pi, size=(model.n_layers * self.n_qubits,), device=self.device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = {'pretrain_loss': [], 'pretrain_accuracy': []}
        for epoch in tqdm(range(n_epochs), desc="Pretraining CloudLeak"):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in public_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.view(-1, 1).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) >= 0.5).float().squeeze()
                total += targets.size(0)
                correct += (predicted == targets.squeeze()).sum().item()
            avg_loss = running_loss / len(public_loader)
            accuracy = 100 * correct / total
            history['pretrain_loss'].append(avg_loss)
            history['pretrain_accuracy'].append(accuracy)
            print(f"Pretrain Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        if save:
            save_path = os.path.join(self.save_path, f'pretrained_cloudleak_{architecture}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Pretrained model saved to {save_path}")
        return model, history

    def load_pretrained_model(self, architecture='L2'):
        architecture_map = {name: model for name, model in self.qnn_zoo}
        if architecture not in architecture_map:
            raise ValueError(f"Architecture {architecture} not found in QNN zoo.")
        model = architecture_map[architecture]
        pretrained_path = os.path.join(self.save_path, f'pretrained_cloudleak_{architecture}.pth')
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
            print(f"Loaded pretrained model from {pretrained_path}")
            return model
        else:
            print(f"No pretrained model found at {pretrained_path}")
            return None

    def train(self, query_dataset, out_of_domain_loader, n_epochs=20, batch_size=8, architecture='L2', loss_type='nll'):
        """
        Huấn luyện CloudLeak
        """
        
        # === GIAI ĐOẠN 1: TIỀN HUẤN LUYỆN ===
        print("--- CloudLeak: Giai đoạn 1: Tiền huấn luyện ---")
        model = self.load_pretrained_model(architecture)
        if model is None:
            # Dùng out_of_domain_loader làm public_loader cho pre-training
            print("Chưa có mô hình pre-trained, bắt đầu pre-training...")
            self.pretrain(out_of_domain_loader, n_epochs=10, batch_size=batch_size, architecture=architecture)
            model = self.load_pretrained_model(architecture)
            if model is None: 
                raise RuntimeError("Pre-training thất bại, không thể tiếp tục.")

        # === GIAI ĐOẠN 2: XÂY DỰNG TẬP DỮ LIỆU TRUY VẤN ĐỐI KHÁNG ===
        print("\n--- CloudLeak: Giai đoạn 2: Xây dựng tập dữ liệu truy vấn đối kháng ---")
        
        adversarial_set_path = os.path.join(self.save_path, f'cloudleak_adversarial_set_{architecture}.npy')
        
        if os.path.exists(adversarial_set_path):
            print(f"Đang tải tập đối kháng đã tạo từ: {adversarial_set_path}")
            adversarial_query_set_np = np.load(adversarial_set_path)
        else:
            print("Chưa có tập đối kháng, bắt đầu tạo...")
            # 2.1. Chuẩn bị nguồn tài nguyên (resource pool)
            in_domain_pool = np.concatenate([item[0] for item in query_dataset])
            out_domain_pool_list = [item[0].cpu().numpy() for i, item in enumerate(out_of_domain_loader) if i * batch_size < 512]
            out_domain_pool = np.concatenate(out_domain_pool_list)[:512]
            resource_pool = np.concatenate([in_domain_pool, out_domain_pool])
            print(f"Tổng nguồn tài nguyên: {resource_pool.shape[0]} ảnh.")

            # 2.2. Tạo mẫu đối kháng từ toàn bộ resource pool
            adversarial_query_set_np = self.generate_adversarial_samples(model, resource_pool)
            
            # Lưu lại để dùng cho các lần chạy sau
            np.save(adversarial_set_path, adversarial_query_set_np)
            print(f"Đã lưu tập đối kháng tại: {adversarial_set_path}")
        
        adversarial_query_set_tensor = torch.tensor(adversarial_query_set_np, dtype=torch.float32).to(self.device)
        print(f"Kích thước tập truy vấn đối kháng: {adversarial_query_set_np.shape[0]} mẫu.")
        
        # === GIAI ĐOẠN 3: TRUY VẤN & HUẤN LUYỆN CUỐI CÙNG ===
        print("\n--- CloudLeak: Giai đoạn 3: Truy vấn nạn nhân và Huấn luyện ---")

        # 3.1. Truy vấn nạn nhân để lấy nhãn cho tập đối kháng
        # Để tránh quá tải bộ nhớ, truy vấn theo từng batch
        labeled_adversarial_dataset = []
        adv_loader = DataLoader(TensorDataset(adversarial_query_set_tensor), batch_size=batch_size)
        
        for adv_batch_tensor, in tqdm(adv_loader, desc="Truy vấn nạn nhân với các mẫu đối kháng"):
            outputs_prob = self.qnnaas_predict(self.victim_model, adv_batch_tensor, self.device)
            # Lưu lại cặp (ảnh đối kháng, nhãn từ victim)
            labeled_adversarial_dataset.append(
                (adv_batch_tensor.cpu().numpy(), outputs_prob.cpu().numpy())
            )

        # Gộp kết quả
        final_inputs = np.concatenate([item[0] for item in labeled_adversarial_dataset])
        # Lấy xác suất của lớp 1 làm mục tiêu
        final_outputs = np.concatenate([item[1][:, 1:2] for item in labeled_adversarial_dataset])
        
        # 3.2. Fine-tune mô hình trên dữ liệu đã thu thập
        final_dataset = TensorDataset(
            torch.tensor(final_inputs, dtype=torch.float32),
            torch.tensor(final_outputs, dtype=torch.float32).view(-1, 1)
        )
        final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = HuberLoss(delta=0.5) if loss_type == 'huber' else nn.BCEWithLogitsLoss()
        history = {'train_loss': []}

        print(f"Bắt đầu Fine-tuning cuối cùng cho CloudLeak ({architecture})...")
        for epoch in tqdm(range(n_epochs), desc=f"Fine-tuning CloudLeak"):
            model.train()
            running_loss = 0.0
            for inputs, targets in final_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs) # model trả về logits
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            history['train_loss'].append(running_loss / len(final_loader))
        
        print("Hoàn thành huấn luyện CloudLeak.")
        return model, history
    def evaluate(self, model, test_loader):
        metrics = evaluate_model_with_metrics(model, test_loader, self.device)
        return metrics

class QuantumLeak(ModelExtraction):
    def __init__(self, victim_model, n_qubits=4, query_budget=6000, n_committee=3, device="cuda", save_path="./results", circuit_device="default.mixed"):
        super().__init__(victim_model, n_qubits, query_budget, device, save_path, circuit_device)
        self.n_committee = n_committee
        self.ensemble_models = []

    def train(self, query_dataset, out_of_domain_loader, n_epochs=30, batch_size=8, architecture='L2', loss_type='huber'):
        return self.train_ensemble(query_dataset, n_epochs, batch_size, architecture, loss_type, self.n_committee)

    def train_ensemble(self, query_dataset, n_epochs=30, batch_size=8, architecture='L2', loss_type='huber', n_committee=None):
        n_committee = n_committee or self.n_committee
        criterion = HuberLoss(delta=0.5) if loss_type == 'huber' else nn.BCEWithLogitsLoss()
        dataset_size = len(query_dataset)
        subset_size = dataset_size // n_committee
        ensemble_models = []
        history = {'train_loss': [], 'train_accuracy': []}
        architecture_map = {name: model for name, model in self.qnn_zoo}
        if architecture not in architecture_map:
            raise ValueError(f"Architecture {architecture} not found in QNN zoo.")
        base_model = architecture_map[architecture]
        base_circuit = base_model.quantum_circuit
        vqc_layers = base_model.n_layers
        for i in range(n_committee):
            model = QNN(self.n_qubits, base_circuit, vqc_layers).to(self.device)
            param_size = 12 * 2 if architecture == 'A1' else (4 * 2 if architecture == 'A2' else vqc_layers * self.n_qubits)
            model.q_params = nn.Parameter(
                torch.normal(mean=0, std=np.pi, size=(param_size,)).to(self.device)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            subset_indices = np.random.choice(dataset_size, subset_size, replace=True)
            subset_data = [query_dataset[idx] for idx in subset_indices]
            inputs = np.concatenate([item[0] for item in subset_data])
            outputs = np.concatenate([item[1][:, 1:2] for item in subset_data])
            subset_dataset = TensorDataset(
                torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(outputs, dtype=torch.float32).view(-1, 1)
            )
            subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(n_epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, targets in subset_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    targets = targets.view(-1, 1).float()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) >= 0.5).float().squeeze()
                    target_labels = (targets >= 0.5).float().squeeze()
                    total += targets.size(0)
                    correct += (predicted == target_labels).sum().item()
                avg_loss = running_loss / len(subset_loader)
                accuracy = 100 * correct / total
                history['train_loss'].append(avg_loss)
                history['train_accuracy'].append(accuracy)
                print(f"Committee {i+1}, Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            ensemble_models.append(model)
        self.ensemble_models = ensemble_models
        return ensemble_models, history

    def evaluate(self, test_loader):
        metrics = evaluate_model_with_metrics(self.ensemble_models, test_loader, self.device, is_ensemble=True)
        return metrics

    def ablation_study(self, data_loader, test_loader, out_of_domain_loader, query_budgets=[1500, 3000, 6000], architectures=['L1', 'L2', 'L3', 'A1', 'A2'], committee_numbers=[3, 5, 7], epochs=30):
        results = []
        cloud_leak = CloudLeak(
            self.victim_model, self.n_qubits, self.query_budget, self.device, self.save_path, self.circuit_device
        )
        public_inputs, public_targets = [], []
        for inputs, targets in out_of_domain_loader:
            public_inputs.append(inputs.cpu().numpy())
            public_targets.append(targets.cpu().numpy())
        public_inputs = np.concatenate(public_inputs)[:3000]
        public_targets = np.concatenate(public_targets)[:3000]
        public_dataset = TensorDataset(
            torch.tensor(public_inputs, dtype=torch.float32),
            torch.tensor(public_targets, dtype=torch.float32)
        )
        public_loader = DataLoader(public_dataset, batch_size=8, shuffle=True)
        cloud_leak.pretrain(public_loader, n_epochs=10, batch_size=8, architecture='L2')
        for query_budget in query_budgets:
            self.query_budget = query_budget
            query_dataset = self.query_victim(data_loader, n_rounds=3)
            for architecture in architectures:
                for n_committee in committee_numbers:
                    for scheme in ['Single-N', 'Single-H', 'Ens-N', 'Ens-H']:
                        if 'Single' in scheme:
                            loss_type = 'nll' if scheme == 'Single-N' else 'huber'
                            model, _ = cloud_leak.train(
                                query_dataset, out_of_domain_loader, n_epochs=20,
                                batch_size=8, architecture=architecture, loss_type=loss_type
                            )
                            metrics = cloud_leak.evaluate(model, test_loader)
                        else:
                            loss_type = 'nll' if scheme == 'Ens-N' else 'huber'
                            ensemble_models, _ = self.train_ensemble(
                                query_dataset, n_epochs=epochs, batch_size=8, architecture=architecture,
                                loss_type=loss_type, n_committee=n_committee
                            )
                            metrics = self.evaluate(test_loader)
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
        results_df.to_csv(os.path.join(self.save_path, 'ablation_results.csv'), index=False)
        return results

    def plot_ablation_results(self, results_df):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        df_fig4 = results_df[(results_df['Query Budget'] == 6000) & (results_df['Architecture'] == 'L2') & (results_df['Committee'] == 5)]
        schemes = df_fig4['Scheme'].values
        accuracies = df_fig4['Accuracy'].values
        plt.bar(schemes, accuracies, color=['blue', 'green', 'orange', 'red'])
        plt.xlabel('Attack Scheme')
        plt.ylabel('Accuracy (%)')
        plt.title('Figure 4: Comparison of Attack Schemes')
        plt.savefig(os.path.join(self.save_path, 'figure4.png'))
        plt.close()
        plt.figure(figsize=(10, 6))
        df_fig5 = results_df[(results_df['Query Budget'] == 6000) & (results_df['Scheme'] == 'Ens-H') & (results_df['Committee'] == 5)]
        architectures = df_fig5['Architecture'].values
        accuracies = df_fig5['Accuracy'].values
        plt.bar(architectures, accuracies, color='purple')
        plt.xlabel('VQC Architecture')
        plt.ylabel('Accuracy (%)')
        plt.title('Figure 5: Impact of VQC Ansatz')
        plt.savefig(os.path.join(self.save_path, 'figure5.png'))
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
        plt.savefig(os.path.join(self.save_path, 'figure6.png'))
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
        plt.savefig(os.path.join(self.save_path, 'figure7.png'))
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
        plt.savefig(os.path.join(self.save_path, 'figure8.png'))
        plt.close()