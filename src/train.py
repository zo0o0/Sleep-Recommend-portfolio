import torch
from torch.utils.data import DataLoader
from model import SleepTransformer
from dataset import SleepDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

def run_valid(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            out = model(x)
            loss = criterion(out, y)
            probs = torch.sigmoid(out)
            preds = (probs >= 0.5).float()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return (total_loss / total_samples,
            accuracy_score(all_targets, all_preds),
            f1_score(all_targets, all_preds),
            cohen_kappa_score(all_targets, all_preds))


def run_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size, stride = 30, 30
    batch_size = 64
    input_size = 13
    hidden_dim = 64
    num_heads, num_layers = 4, 2
    num_epochs = 5

    data_dir = "../../data"
    train_dataset = SleepDataset(data_dir, 'train', window_size, stride)
    valid_dataset = SleepDataset(data_dir, 'valid', window_size, stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = SleepTransformer(input_size, hidden_dim, num_heads, num_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')

    # loss 값 저장
    train_losses = []
    val_losses = []

    # accuracy 값 저장
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_samples = 0.0, 0
        all_preds, all_targets = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            out = model(x)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

            probs = torch.sigmoid(out)
            preds = (probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        avg_train_loss = total_loss / total_samples
        train_acc = accuracy_score(all_targets, all_preds)
        train_losses.append(avg_train_loss)
        train_accuracy.append(train_acc)

        val_loss, val_acc, val_f1, val_kappa = run_valid(model, valid_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Kappa: {val_kappa:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "saved_model/best_model.pt")
            print("[!] Model saved")

    # loss 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train & Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_plot.png')
    plt.show()

    # accuracy 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train & Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_plot.png')
    plt.show()

if __name__ == '__main__':
    run_train()