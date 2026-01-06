import numpy as np
import pandas as pd
from seaborn_analyzer import regplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

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
seed = 42
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
    random_state=seed
)
# regression_heat_plotは2〜4次元のデータのみ対応しているため、主要な特徴量を選択
USE_EXPLANATORY = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']

# X_train, y_trainを用いてモデル構築

# LightGBM
# データセットを生成する
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'objective': 'binary',  # 最小化させるべき損失関数
    'metric': 'auc',  # 二値分類用の評価指標（AUCスコア）
    'random_state': seed,  # 乱数シード
    'boosting_type': 'gbdt',  # boosting_type
    'n_estimators': 10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
    'verbosity': -1,  # 警告メッセージを抑制
}

# モデル作成
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, 
                  num_boost_round=1000, 
                  callbacks=[lgb.log_evaluation(50), lgb.early_stopping(100)]
                  )

# 保存
model.save_model('model.txt')

# テストデータを予測する
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# AUC (Area Under the Curve) を計算する
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)

# ROC曲線をプロット
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.savefig('output/roc_curve.png', dpi=300, bbox_inches='tight')
print("ROC曲線を output/roc_curve.png に保存しました")
plt.close()

# 特徴量の重要度出力
print(model.feature_importance())

# 特徴量の重要度をプロット
plt.figure(figsize=(10, 6))
lgb.plot_importance(model)
plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
print("特徴量重要度を output/feature_importance.png に保存しました")
plt.close()