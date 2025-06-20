import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def train_model(model, train_loader, val_loader, device, criterion, n_epochs, learning_rate, save_path):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model_path = f"{save_path}/best_model.pth"
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
        train_losses.append(avg_train_loss)
        val_metrics = evaluate_model_with_metrics(model, val_loader, device, criterion)
        val_accuracies.append(val_metrics['accuracy'])
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Accuracy: {val_metrics['accuracy']:.2f}%, "
              f"Precision: {val_metrics['precision']:.2f}, "
              f"Recall: {val_metrics['recall']:.2f}, "
              f"F1: {val_metrics['f1']:.2f}")
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            torch.save(model.state_dict(), best_model_path)
    return train_losses, val_accuracies

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
                ensemble_outputs = []
                for model in models:
                    outputs = model(inputs)
                    ensemble_outputs.append(outputs)
                outputs = torch.stack(ensemble_outputs).mean(dim=0)
            else:
                outputs = models[0](inputs)
            probs = torch.sigmoid(outputs).squeeze()
            predicted = (probs >= 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if criterion is not None:
                loss = criterion(outputs, labels.view(-1, 1)).item()
                total_loss += loss * len(labels)
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
    if criterion is not None:
        metrics['loss'] = total_loss / len(all_labels)
    return metrics