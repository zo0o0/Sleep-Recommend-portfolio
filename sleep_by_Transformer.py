import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

##### 잘못된 지수 표현 제거 #####
def clean_scientific_notation(val):
    try:
        return float(val) # 변환 가능한 경우
    except:
        return np.nan # 변환 불가능한 경우 NaN으로 처리

##### 전처리 함수 #####
def preprocess(file_name):
    df = pd.read_csv(file_name)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%y-%m-%d %H:%M:%S:%f', errors='coerce')
    df.set_index('Timestamp', inplace=True)

    numeric_feature = ['Acc X', 'Acc Y', 'Acc Z', 'Acc Linear X', 'Acc Linear Y', 'Acc Linear Z',
                       'Gyro X', 'Gyro Y', 'Gyro Z', 'Proximity', 'Step', 'Light']
    categorical_feature = ['Screen', 'Sleep']

    numeric_df = df[numeric_feature].copy()
    numeric_df = numeric_df.apply(lambda col: col.map(clean_scientific_notation))
    categorical_df = df[categorical_feature].copy()

    numeric_df = numeric_df.resample(rule='50ms', origin='start').mean()
    categorical_df = categorical_df.resample(rule='50ms', origin='start').max()

    resample_numeric_df = numeric_df.resample(rule='1s', origin='start').mean()
    resample_categorical_df = categorical_df.resample(rule='1s', origin='start').max()

    resample_numeric_df.interpolate(method='linear', inplace=True)
    resample_categorical_df.ffill(inplace=True)

    resample_numeric_df = resample_numeric_df.astype('Float32')
    resample_categorical_df = resample_categorical_df.astype('Float32')

    z_scaler = StandardScaler()
    z_scaler.fit(resample_numeric_df)
    z_scaled_data = z_scaler.transform(resample_numeric_df)
    z_scaled_data = pd.DataFrame(z_scaled_data, columns=numeric_feature, index=resample_numeric_df.index)

    z_scaled_data['Screen'] = resample_categorical_df['Screen'].values

    X_all = z_scaled_data
    y_all = resample_categorical_df['Sleep']

    return X_all, y_all

##### 데이터셋 클래스 #####
class SleepDataset(Dataset):
    def __init__(self, data_dir, mode, window_size, stride):
        self.window_size = window_size
        self.stride = stride

        all_dirs = sorted(glob.glob(os.path.join(data_dir, mode, '*')))
        self.X = []
        self.y = []

        for dir in all_dirs:
            csv_files = sorted(glob.glob(os.path.join(dir, '*.csv')))
            for file_name in csv_files:
                X_all, y_all = preprocess(file_name)

                for i in range(0, len(X_all) - window_size + 1, stride):
                    x_window = X_all.iloc[i:i + window_size].values
                    y_label = y_all.iloc[i + window_size - 1]
                    self.X.append(x_window)
                    self.y.append(y_label)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __getitem__(self, index):
        x = self.X[index].clone().detach().float()
        y = self.y[index].clone().detach().float()
        return x, y

    def __len__(self):
        return len(self.X)

##### Positional Encoding #####
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # (batch_size, seq_len, d_model)
        return x

##### Transformer 모델 #####
class SleepTransformer(nn.Module):
    def __init__(self, input_size, hidden_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)  # (B, T, H)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]  # 마지막 타임스텝 사용
        return self.fc(out)

##### Training 함수 #####
def run_train(num_epochs, model, data_loader, criterion, optimizer, device):
    print('Start training..')
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, (sequence, target) in enumerate(data_loader):
            sequence = sequence.to(device)
            target = target.to(device).unsqueeze(1)
            outputs = model(sequence)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

##### Testing 함수 #####
def run_test(model, data_loader, criterion, device):
    print('\nStart test..')
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for step, (sequence, target) in enumerate(data_loader):
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

##### 파라미터 #####
window_size = 30
stride = 30
batch_size = 64
input_size = 13
hidden_dim = 64
num_heads = 4
num_layers = 2
num_epochs = 5  # 50~100정도 필요
#pytorch lighting

##### 데이터 로딩 #####
data_dir = '../../data/'
train_data = SleepDataset(data_dir, 'train', window_size, stride)
test_data = SleepDataset(data_dir, 'test', window_size, stride)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True) # test_loader batch_size = 1 이어야

##### 모델 정의 #####
torch.manual_seed(7777)
model = SleepTransformer(input_size, hidden_dim, num_heads, num_layers).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

##### 학습 및 평가 #####
run_train(num_epochs, model, train_loader, criterion, optimizer, device)
run_test(model, test_loader, criterion, device)
