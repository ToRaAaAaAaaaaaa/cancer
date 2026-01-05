# Cancer Project

がん診断データを用いた機械学習プロジェクト

---

## 📋 目次
- [GPU環境のセットアップ](#gpu環境のセットアップ)
- [CPU環境のセットアップ](#cpu環境のセットアップ)
- [GPU動作確認](#gpu動作確認)
- [トラブルシューティング](#トラブルシューティング)

---

## 🚀 GPU環境のセットアップ

GPU（NVIDIA）を搭載したPCでの推奨セットアップ方法です。

### Windows PowerShell

#### 1. uvのインストール

PowerShellを管理者権限で開き、以下のコマンドを実行：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後、PowerShellを再起動して確認：

```powershell
uv --version
```

#### 2. プロジェクトのセットアップ

```powershell
cd path\to\cancer
```

#### 3. GPU対応パッケージのインストール

```powershell
uv venv
uv pip install -r requirements-gpu.txt
```

#### 4. GPU動作確認

```powershell
# PyTorchでGPU確認
uv run python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# XGBoostでGPU使用例
uv run python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')"
```

#### 5. スクリプト実行

```powershell
uv run main.py
```

### Linux/WSL

#### 1. uvのインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

#### 2. セットアップとインストール

```bash
cd /path/to/cancer
uv venv
uv pip install -r requirements-gpu.txt
```

#### 3. GPU確認

```bash
# NVIDIA GPUの確認
nvidia-smi

# PyTorchでGPU確認
uv run python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

---

## 💻 CPU環境のセットアップ

GPUがない、またはCPUのみで実行したい場合のセットアップです。

### Windows PowerShell

#### 1. uvのインストール

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. セットアップ

```powershell
cd path\to\cancer
uv venv
uv pip install -r requirements.txt
```

#### 3. スクリプト実行

```powershell
uv run main.py
```

### Linux/macOS

#### 1. uvのインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # または ~/.bashrc, ~/.zshrc など
```

#### 2. セットアップ

```bash
cd /path/to/cancer
uv venv
uv pip install -r requirements.txt
```

#### 3. スクリプト実行

```bash
uv run main.py
```

---

## 🔍 GPU動作確認

### PyTorchでGPU使用

```python
import torch

# GPU利用可能か確認
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# テンソルをGPUに配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 3).to(device)
print(f"Tensor device: {x.device}")
```

### XGBoostでGPU使用

```python
import xgboost as xgb

# GPU使用時のパラメータ
params = {
    'tree_method': 'hist',  # CPUの場合
    # 'tree_method': 'gpu_hist',  # GPUの場合（CUDA対応ビルドが必要）
    'device': 'cuda',  # GPUを使用
}

# モデル訓練
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
```

### LightGBMでGPU使用

```python
import lightgbm as lgb

# GPU使用時のパラメータ
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)
```

---

## 📦 インストールされるパッケージ

### 共通パッケージ（CPU/GPU両方）
- numpy: 数値計算
- pandas: データ処理
- scikit-learn: 機械学習
- xgboost: 勾配ブースティング
- lightgbm: 軽量勾配ブースティング
- matplotlib: データ可視化
- seaborn: 統計的データ可視化
- optuna: ハイパーパラメータ最適化
- jupyter: Jupyter Notebook環境
- ipykernel: Jupyterカーネル
- scipy: 科学計算
- joblib: 並列処理
- imbalanced-learn: 不均衡データ対応

### GPU環境のみ追加
- torch: PyTorch（CUDA 12.4対応）
- torchvision: PyTorch画像処理ライブラリ

---

## 🛠️ トラブルシューティング

### パッケージが見つからない場合

グローバルの`pip`ではなく、必ず`uv pip`を使用：

```bash
# ❌ 間違い（グローバル環境を見る）
pip list

# ✅ 正しい（uv環境を見る）
uv pip list
```

### インストール確認スクリプト

**Windows PowerShell:**
```powershell
Get-Content requirements.txt | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '\S' } | ForEach-Object {
    $pkgName = ($_ -split '[><=!]')[0].Trim()
    if (uv pip show $pkgName 2>$null) {
        Write-Host "✓ $pkgName" -ForegroundColor Green
    } else {
        Write-Host "✗ $pkgName (missing)" -ForegroundColor Red
    }
}
```

**Linux/macOS:**
```bash
while read line; do
    if [[ "$line" =~ ^#.*$ ]] || [[ -z "$line" ]]; then
        continue
    fi
    pkg_name=$(echo "$line" | sed 's/[><=!].*//' | xargs)
    if uv pip show "$pkg_name" > /dev/null 2>&1; then
        echo "✓ $pkg_name"
    else
        echo "✗ $pkg_name (missing)"
    fi
done < requirements.txt
```

### 環境のリセット

仮想環境を削除して再作成：

**Windows:**
```powershell
Remove-Item -Recurse -Force .venv
uv venv
uv pip install -r requirements.txt  # またはrequirements-gpu.txt
```

**Linux/macOS:**
```bash
rm -rf .venv
uv venv
uv pip install -r requirements.txt  # またはrequirements-gpu.txt
```

### GPUが認識されない場合

1. NVIDIAドライバーが最新か確認：
   ```bash
   nvidia-smi
   ```

2. CUDA Toolkitのバージョン確認：
   ```bash
   nvcc --version
   ```

3. PyTorchのCUDAバージョンが合っているか確認：
   ```python
   import torch
   print(torch.version.cuda)
   ```

4. 再インストール：
   ```bash
   uv pip uninstall torch torchvision
   uv pip install -r requirements-gpu.txt
   ```

---

## 📝 使用方法

### 基本的な実行

```bash
uv run main.py
```

### Jupyter Notebookの起動

```bash
uv run jupyter notebook
```

### Jupyter Labの起動

```bash
uv run jupyter lab
```

---

## 🎯 プロジェクト概要

このプロジェクトは、がん診断データ（良性/悪性の分類）を用いた機械学習モデルの構築を行います。

- **データ**: 569件の患者データ（良性357件、悪性212件）
- **タスク**: 2値分類（がんの良性・悪性判定）
- **重要指標**: Recall（再現率）を最重視（がんの見逃しを最小化）
