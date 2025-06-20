 
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def plot_training_history(history, save_path, title="Training History"):
    plt.figure(figsize=(12, 4))
    if 'train_loss' in history:
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    if 'test_accuracy' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['test_accuracy'], label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_")}.png'))
    plt.close()

def plot_quanv_visualization(train_images, q_train_images, save_path, n_samples=4, n_channels=4):
    q_train_images_np = q_train_images.cpu().numpy().transpose(0, 2, 3, 1)
    fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
    for k in range(n_samples):
        axes[0, 0].set_ylabel("Input")
        if k != 0:
            axes[0, k].yaxis.set_visible(False)
        axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")
        for c in range(n_channels):
            axes[c + 1, 0].set_ylabel(f"Output [ch. {c}]")
            if k != 0:
                axes[c + 1, k].yaxis.set_visible(False)
            axes[c + 1, k].imshow(q_train_images_np[k, :, :, c], cmap="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'quanv_visualization.png'))
    plt.close()

def plot_model_comparison(accuracies, labels, save_path, title="Model Comparison"):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracies, color=['blue', 'orange', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.savefig(os.path.join(save_path, 'model_comparison.png'))
    plt.close()