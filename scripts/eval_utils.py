import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_metrics(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y, t in loader:
            out = model(x.to(device), y.to(device))[:, -1, 0]
            pred = torch.sigmoid(out).cpu().numpy() > 0.5
            preds.extend(pred)
            labels.extend(t.numpy())

    print(f"Accuracy: {accuracy_score(labels, preds):.4f}, "
          f"Precision: {precision_score(labels, preds):.4f}, "
          f"Recall: {recall_score(labels, preds):.4f}, "
          f"F1: {f1_score(labels, preds):.4f}")
