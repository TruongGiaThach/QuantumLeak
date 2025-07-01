 
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from joblib import Parallel, delayed
import pennylane as qml
from configs.config import SAVE_PATH, N_TRAIN, N_TEST, PREPROCESS

class CIFAR10DataPipeline:
    def __init__(self, n_train=N_TRAIN, n_test=N_TEST, save_path=SAVE_PATH):
        self.n_train = n_train
        self.n_test = n_test
        self.save_path = save_path
        self.q_train_images = None
        self.q_test_images = None

    def load_data(self, qanv=True):
        cifar_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
        train_images, train_labels = cifar_dataset.data, np.array(cifar_dataset.targets)
        cifar_dataset_test = datasets.CIFAR10(root='./data', train=False, download=True)
        test_images, test_labels = cifar_dataset_test.data, np.array(cifar_dataset_test.targets)
        train_images = train_images[:self.n_train]
        train_labels = train_labels[:self.n_train]
        test_images = test_images[:self.n_test]
        test_labels = test_labels[:self.n_test]
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        if qanv:
            train_images = np.expand_dims(train_images, axis=-1)
            test_images = np.expand_dims(test_images, axis=-1)
        return train_images, train_labels, test_images, test_labels

    def quanv(self, image, circuit):
        out = np.zeros((16, 16, 4))
        for j in range(0, 32, 2):
            for k in range(0, 32, 2):
                q_results = circuit([
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ])
                for c in range(4):
                    out[j // 2, k // 2, c] = q_results[c]
        return out

    def preprocess_quanv(self, train_images, test_images, circuit, load_from_drive=False):
        train_save_path = os.path.join(self.save_path, "q_train_images.npy")
        test_save_path = os.path.join(self.save_path, "q_test_images.npy")
        if load_from_drive and os.path.exists(train_save_path) and os.path.exists(test_save_path):
            print(f"Loading preprocessed data from {train_save_path}")
            self.q_train_images = np.load(train_save_path)
            self.q_test_images = np.load(test_save_path)
        else:
            if load_from_drive:
                print("Preprocessed files not found. Running quanvolution...")
            if PREPROCESS:
                print("Quantum pre-processing of train images:")
                self.q_train_images = Parallel(n_jobs=-1)(
                    delayed(self.quanv)(img, circuit) for img in tqdm(train_images, desc="Train")
                )
                self.q_train_images = np.asarray(self.q_train_images)
                print("Quantum pre-processing of test images:")
                self.q_test_images = Parallel(n_jobs=-1)(
                    delayed(self.quanv)(img, circuit) for img in tqdm(test_images, desc="Test")
                )
                self.q_test_images = np.asarray(self.q_test_images)
                np.save(train_save_path, self.q_train_images)
                np.save(test_save_path, self.q_test_images)
        return self.q_train_images, self.q_test_images

def get_cq_dataloaders(pipeline, batch_size=8, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    train_images, train_labels, test_images, test_labels = pipeline.load_data(qanv=False)
    train_idx = np.isin(train_labels, [0, 1])
    test_idx = np.isin(test_labels, [0, 1])
    train_images = train_images[train_idx]
    train_labels = train_labels[train_idx]
    test_images = test_images[test_idx]
    test_labels = test_labels[test_idx]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_images_torch = torch.stack([transform(img) for img in train_images]).float()
    test_images_torch = torch.stack([transform(img) for img in test_images]).float()
    train_labels_torch = torch.tensor(train_labels, dtype=torch.long)
    test_labels_torch = torch.tensor(test_labels, dtype=torch.long)
    train_images_torch = train_images_torch.to(device)
    train_labels_torch = train_labels_torch.to(device)
    test_images_torch = test_images_torch.to(device)
    test_labels_torch = test_labels_torch.to(device)
    train_dataset = TensorDataset(train_images_torch, train_labels_torch)
    test_dataset = TensorDataset(test_images_torch, test_labels_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f"Train loader batches: {len(train_loader)}, Test loader batches: {len(test_loader)}")
    return train_loader, test_loader

def create_out_of_domain_loader(pipeline, batch_size=8, n_samples=3000):
    train_images, train_labels, _, _ = pipeline.load_data(qanv=False)
    out_classes = [2, 3, 5, 6]
    out_idx = np.isin(train_labels, out_classes)
    out_images = train_images[out_idx][:n_samples]
    out_labels = train_labels[out_idx][:n_samples]
    if len(out_images) < n_samples:
        raise ValueError(f"Requested {n_samples} out-of-domain samples, but only {len(out_images)} available.")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    out_images_torch = torch.stack([transform(img) for img in out_images]).float()
    out_labels_torch = (torch.tensor(out_labels, dtype=torch.long) >= 2).long()
    out_dataset = TensorDataset(out_images_torch, out_labels_torch)
    out_loader = DataLoader(out_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Out-of-domain loader: {len(out_dataset)} samples, {len(out_loader)} batches")
    return out_loader