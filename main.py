import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# データの読み込み
data = pd.read_csv('input/cancer_data.csv')

# diagnosis列のBとMの数を数え上げ
diagnosis_counts = data['diagnosis'].value_counts()
print("\n=== Diagnosis Count ===")
print(diagnosis_counts)

# データエンジニアリング

# 欠損値の取得
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns
print(kesson_table(data)) # 欠損数0

# 訓練とテストに分割(8:2)
exclude_cols = ['id', 'diagnosis'] # ここは好きに入れたくない特徴量等あれば追加しちゃってください
X = data.drop(columns=exclude_cols)
y = data['diagnosis']

# diagnosisをラベルエンコーディング（B→0, M→1）
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2, 
    stratify=y_encoded,
    random_state=42
)

# X_train, y_trainを用いてモデル構築

# LightGBM
def create_lgb_model():
    params = {
        "learning_rate": 0.07, 
        "shrinkage_rate": 0.12
    }
    return params
lgb_params = create_lgb_model()
train_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_test, y_test, reference=train_data)

lgb_model = lgb.train(lgb_params, train_data)

# 評価
