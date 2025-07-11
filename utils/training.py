import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from models.qnn_leak import QNN
import os
import json # Thêm import để làm việc với file JSON

def train_model(model, train_loader, val_loader, device, criterion, n_epochs, learning_rate, save_path):
    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(save_path, exist_ok=True)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_accuracy = 0.0
    best_metrics = {}
    best_model_path = os.path.join(save_path, "best_model.pth")
    best_metrics_path = os.path.join(save_path, "best_metrics.json")

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Đánh giá trên tập validation
        val_metrics = evaluate_model_with_metrics(model, val_loader, device, criterion)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Accuracy: {val_metrics['accuracy']:.2f}%, "
              f"Precision: {val_metrics['precision']:.2f}, "
              f"Recall: {val_metrics['recall']:.2f}, "
              f"F1: {val_metrics['f1']:.2f}")

        # --- LOGIC LƯU TRỮ ---
        if val_metrics['accuracy'] > best_val_accuracy:
            print(f"  -> New best model found! Saving to {save_path}")
            best_val_accuracy = val_metrics['accuracy']
            best_metrics = val_metrics
            
            # Lưu trọng số của mô hình tốt nhất
            torch.save(model.state_dict(), best_model_path)
            
            # Lưu các chỉ số của mô hình tốt nhất vào file JSON
            with open(best_metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=4)
                
    print(f"\nTraining finished. Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best metrics saved to {best_metrics_path}")
    
    # Trả về lịch sử và các chỉ số tốt nhất
    return history, best_metrics



def evaluate_model_with_metrics(models, data_loader, device, criterion=None, is_ensemble=False):
    if not isinstance(models, (list, tuple)):
        models = [models]
        
    for model in models:
        model.to(device)
        model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0 if criterion is not None else None
    

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().squeeze()
            
            if is_ensemble:
                ensemble_preds_tensor = []
                for model in models:
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).squeeze()
                    predicted = (probs >= 0.5).float()
                    ensemble_preds_tensor.append(predicted)
                
                stacked_preds = torch.stack(ensemble_preds_tensor, dim=1)
                predicted, _ = torch.mode(stacked_preds, dim=1)
            else: # Single model
                outputs = models[0](inputs)
                probs = torch.sigmoid(outputs).squeeze()
                
                predicted = (probs >= 0.5).float()
                
                if criterion is not None:
                    # Tính loss chỉ cho single models
                    loss = criterion(outputs, labels.view(-1, 1)).item()
                    total_loss += loss * len(labels)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if total_loss is not None:
        metrics['loss'] = total_loss / len(all_labels) if len(all_labels) > 0 else 0
        
    return metrics