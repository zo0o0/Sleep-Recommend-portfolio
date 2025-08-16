import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_scientific_notation(val):
    try:
        return float(val)
    except:
        return np.nan

def my_mode(x):
    if x.isna().all():
        return np.nan
    return x.value_counts().idxmax()

def preprocess(file_name):
    df = pd.read_csv(file_name)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%y-%m-%d %H:%M:%S:%f', errors='coerce')
    df.set_index('Timestamp', inplace=True)

    numeric_feature = ['Acc X', 'Acc Y', 'Acc Z', 'Acc Linear X', 'Acc Linear Y', 'Acc Linear Z',
                       'Gyro X', 'Gyro Y', 'Gyro Z', 'Proximity', 'Step', 'Light']
    categorical_feature = ['Screen', 'Sleep']

    numeric_df = df[numeric_feature].copy().apply(lambda col: col.map(clean_scientific_notation))
    categorical_df = df[categorical_feature].copy()

    numeric_df = numeric_df.resample('50ms').mean().resample('1s').mean()

    categorical_df = categorical_df.resample('50ms').mean() # 50ms 간격으로 표기
    categorical_df.ffill(inplace=True) # 결측치는 앞의 값으로 채우기
    categorical_df = categorical_df.resample('1s').mean()
    categorical_df = (categorical_df >= 0.5).astype(float) # 평균이 0.5보다 크면 1, 작으면 0

    numeric_df.interpolate(method='linear', inplace=True)
    categorical_df.ffill(inplace=True)

    z_scaler = StandardScaler()
    z_scaler.fit(numeric_df)
    z_scaled_data = pd.DataFrame(z_scaler.transform(numeric_df), columns=numeric_feature, index=numeric_df.index)

    z_scaled_data['Screen'] = categorical_df['Screen'].values

    return z_scaled_data, categorical_df['Sleep']
