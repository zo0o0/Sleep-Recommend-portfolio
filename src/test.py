import os
import torch
from model import SleepTransformer
from dataset import SleepDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from torch import nn
import subprocess
import sys

# 하이퍼파라미터 (train.py와 동일하게 맞춰야 함)
window_size = 30
stride = 30
input_size = 13
hidden_dim = 64
num_heads = 4
num_layers = 2
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 모델 파일 확인
python_exec = sys.executable
train_path = os.path.join(os.path.dirname(__file__), "train.py")
model_path = "saved_model/best_model.pt"

if not os.path.exists("saved_model/best_model.pt"):
    print("[!] best_model.pt not found. Running train.py first...")
    subprocess.run([python_exec, train_path], check=True)

# 2. 모델 정의 및 로드
model = SleepTransformer(input_size, hidden_dim, num_heads, num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 3. 데이터 로드
data_dir = "../../data/"
test_data = SleepDataset(data_dir, 'test', window_size, stride)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 4. 평가 함수
def run_test(model, data_loader, criterion, device):
    print('\n[Test Set Evaluation]')
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for sequence, target in data_loader:
            sequence = sequence.to(device)
            target = target.to(device).unsqueeze(1)
            outputs = model(sequence)
            loss = criterion(outputs, target)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy().astype(int))

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test Cohen's Kappa: {kappa:.4f}")
    print("Confusion Matrix:")
    print(cm)

# 5. 테스트 실행
run_test(model, test_loader, nn.BCEWithLogitsLoss(), device)
