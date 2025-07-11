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
from quantum_circuits.substitution_circuit import create_substitute_circuit

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

    def _create_L_circuit(self, n_layers):
        """Tạo mạch Circuit14."""
        circuit_qnode, params_shape = create_substitute_circuit(
            self.n_qubits, n_layers, self.circuit_device
        )
        return circuit_qnode, params_shape

    def _create_a1_circuit(self):
        dev = qml.device(self.circuit_device, wires=self.n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights, crx_weights=None):
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            param_idx = 0
            for _ in range(2):
                for i in range(self.n_qubits):
                    qml.RX(weights[param_idx], wires=i); param_idx += 1
                    qml.RY(weights[param_idx], wires=i); param_idx += 1
                    qml.RZ(weights[param_idx], wires=i); param_idx += 1
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        params_shape = {
            'weights': (2, self.n_qubits, 2), # 2 lớp, n_qubits, 2 cổng (RY, RZ)
            'crx_weights': None # Mạch này không dùng crx_weights có tham số
        }
        return qml.transforms.broadcast_expand(circuit), params_shape

    def _create_a2_circuit(self):
        dev = qml.device(self.circuit_device, wires=self.n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs,  weights, crx_weights=None):
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            param_idx = 0
            for _ in range(2):
                qml.RX(weights[param_idx], wires=0); param_idx += 1
                qml.RY(weights[param_idx], wires=1); param_idx += 1
                qml.RZ(weights[param_idx], wires=2); param_idx += 1
                qml.RX(weights[param_idx], wires=3); param_idx += 1
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        params_shape = {
            'weights': (2, self.n_qubits, 3), # 2 lớp, n_qubits, 3 cổng (RZ,RY,RZ)
            'crx_weights': (2, self.n_qubits) # 2 lớp, n_qubits cổng CRX
        }
        return qml.transforms.broadcast_expand(circuit), params_shape

    
    def get_substitute_qnn(self, architecture: str):
        """
        Tạo và trả về một INSTANCE QNN MỚI dựa trên tên kiến trúc.
        """
        if architecture.startswith('L'):
            n_layers = int(architecture[1:])
            circuit, params_shape = self._create_L_circuit(n_layers)
        elif architecture == 'A1':
            circuit, params_shape = self._create_a1_circuit()
        elif architecture == 'A2':
            circuit, params_shape = self._create_a2_circuit()
        else:
            raise ValueError(f"Kiến trúc không xác định: {architecture}")

        # Tất cả model đều được tạo từ cùng một lớp, chỉ khác nhau về mạch và shape của tham số
        model = QNN(
            n_qubits=self.n_qubits,
            quantum_circuit=circuit,
            params_shape=params_shape,
            device=self.device
        ).to(self.device)
        
        return model

    def qnnaas_predict(self, model, images, device):
        model.eval()
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            prob_class_1 = outputs
            prob_class_0 = 1 - prob_class_1
            prob_vectors = torch.cat([prob_class_0, prob_class_1], dim=1)
        return prob_vectors

    def query_victim(self, data_loader, n_rounds=3, add_noise=True):
        """
        Truy vấn Victim Model với tùy chọn thêm nhiễu NISQ.
        Args:
            data_loader: DataLoader chứa ảnh (chỉ ảnh, không nhãn).
            n_rounds: Số vòng truy vấn.
            add_noise: Có thêm nhiễu NISQ hay không.
        Returns:
            query_dataset: Danh sách các cặp (inputs, noisy_outputs).
        """
        self.victim_model.eval()
        query_dataset = []
        samples_per_round = self.query_budget // n_rounds
        spam_noise = 0.0034     
        gate_1q_noise = 0.00197 
        gate_2q_noise = 0.0244 
        crosstalk_factor = 1.2 
        if not add_noise:
            spam_error = gate_1q_error = gate_2q_error = 0.0
            crosstalk_factor = 1.0
        with torch.no_grad():
            for round in range(n_rounds):
                samples_collected = 0
                for batch in tqdm(data_loader, desc=f"Query round {round + 1}/{n_rounds}"):
                    inputs = batch[0].to(self.device)
                    outputs = self.qnnaas_predict(self.victim_model, inputs, self.device)
                    if add_noise:
                        base_noise_scale = spam_error + gate_1q_error + gate_2q_error
                        total_noise_scale = base_noise_scale * crosstalk_factor
                        # Tạo nhiễu ngẫu nhiên theo phân phối chuẩn
                        noise = torch.randn_like(outputs) * total_noise_scale
                        noisy_outputs = outputs + noise
                        noisy_outputs = torch.clamp(noisy_outputs, 0, 1)
                    else:
                        noisy_outputs = outputs
                    query_dataset.append((inputs.cpu().numpy(), noisy_outputs.cpu().numpy()))
                    samples_collected += inputs.size(0)
                    if samples_collected >= samples_per_round:
                        break
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
            outputs = substitute_model(inputs_tensor) # model trả về logits
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
    def pretrain(self, public_loader, n_epochs=10, batch_size=8, architecture='L2',save=True):
        criterion = nn.BCEWithLogitsLoss()
        model = self.get_substitute_qnn(architecture)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = {'pretrain_loss': [], 'pretrain_accuracy': []}
        pbar = tqdm(range(n_epochs), desc=f"Pre-training CloudLeak ({architecture})")
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
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2f}%")
        if save:
            save_path = os.path.join(self.save_path, f'pretrained_cloudleak_{architecture}_q{self.query_budget}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Pretrained model saved to {save_path}")
        return model, history

    def load_pretrained_model(self, architecture='L2'):
        model = self.get_substitute_qnn(architecture)
        pretrained_path = os.path.join(self.save_path, f'pretrained_cloudleak_{architecture}_q{self.query_budget}.pth')
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
            print(f"Loaded pretrained model from {pretrained_path}")
            return model
        else:
            print(f"No pretrained model found at {pretrained_path}")
            return None

    def train(self, in_domain_source_loader, out_of_domain_loader, n_epochs=20, batch_size=8, architecture='L2', loss_type='nll', add_noise=True):
        """
        Huấn luyện CloudLeak
        Args:
            in_domain_source_loader: DataLoader chứa 6000 ảnh in-domain (chỉ ảnh).
            out_of_domain_loader: DataLoader chứa 3000 ảnh out-of-domain (có nhãn).
            n_epochs: Số epoch để fine-tune.
            batch_size: Kích thước batch.
            architecture: Kiến trúc VQC ('L1', 'L2', 'L3', 'A1', 'A2').
            loss_type: Loại loss ('nll' hoặc 'huber').
            add_noise: Có thêm nhiễu NISQ khi truy vấn Victim Model hay không.
        Returns:
            model: Mô hình đã fine-tune.
            history: Lịch sử huấn luyện.
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
        
        adversarial_set_path = os.path.join(self.save_path, f'cloudleak_adversarial_set_{architecture}_q{self.query_budget}.npy')
        
        if os.path.exists(adversarial_set_path):
            print(f"Đang tải tập đối kháng đã tạo từ: {adversarial_set_path}")
            adversarial_query_set_np = np.load(adversarial_set_path)
        else:
            print("Chưa có tập đối kháng, bắt đầu tạo...")
            # 2.1. Chuẩn bị nguồn tài nguyên (resource pool)
            in_domain_pool = np.concatenate([batch[0].cpu().numpy() for batch in in_domain_source_loader])
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
        
        labeled_adversarial_dataset = self.query_victim(adv_loader, n_rounds=1, add_noise=add_noise)

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

        print(f"---Bắt đầu Fine-tuning cuối cùng cho CloudLeak ({architecture})...")
        pbar = tqdm(range(n_epochs), desc=f"Fine-tuning CloudLeak")
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
            avg_loss = running_loss / len(final_loader)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
        
        print(f"Hoàn thành huấn luyện CloudLeak ({'noisy' if add_noise else 'clean'} queries).")
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

    def train_ensemble(self, query_dataset, n_epochs=30, batch_size=8, architecture='L2', loss_type='huber', n_committee=None , learn_rate=0.001):
        n_committee = n_committee or self.n_committee
        criterion = HuberLoss(delta=0.5) if loss_type == 'huber' else nn.BCEWithLogitsLoss()
        dataset_size = len(query_dataset)
        subset_size = dataset_size // n_committee
        ensemble_models = []
        history = {'train_loss': [], 'train_accuracy': []}
        
        total_steps = n_committee * n_epochs
        pbar = tqdm(total=total_steps, desc="Training QuantumLeak Ensemble")
        for i in range(n_committee):
            model = self.get_substitute_qnn(architecture)
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
            subset_indices = np.random.choice(dataset_size, subset_size, replace=True)
            # Extract subset data from query_dataset (list of tuples)
            inputs = np.concatenate([query_dataset[idx][0] for idx in subset_indices])
            outputs = np.concatenate([query_dataset[idx][1][:, 1:2] for idx in subset_indices])
            
            # Create TensorDataset for the subset
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
                pbar.update(1)
                pbar.set_postfix(committee=f"{i+1}/{n_committee}", epoch=f"{epoch+1}/{n_epochs}", loss=f"{avg_loss:.4f}", accuracy=f"{accuracy:.2f}%")
            ensemble_models.append(model)
            pbar.close()
        self.ensemble_models = ensemble_models
        return ensemble_models, history

    def evaluate(self, test_loader):
        metrics = evaluate_model_with_metrics(self.ensemble_models, test_loader, self.device, is_ensemble=True)
        return metrics

    