# SupertonicTTS 学習PC セットアップガイド

新しいPCでSupertonicTTSの学習環境を構築するための完全ガイドです。

## 📋 システム要件

### 必須要件
- **OS**: Ubuntu 22.04 LTS / Windows 11
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CUDA**: 12.1以上
- **GPU Driver**: 最新版推奨
- **Python**: 3.12以上
- **RAM**: 32GB以上推奨
- **ストレージ**: 最低300GB空き容量
  - LJSpeech データセット: 2.6GB
  - Assets (モデル): ~500MB
  - チェックポイント: 50-100GB
  - 出力ログ: 10-50GB

### 推奨スペック
- **CPU**: 8コア以上
- **RAM**: 64GB
- **ストレージ**: NVMe SSD 500GB以上

---

## 🚀 セットアップ手順

### Step 1: NVIDIA Driver & CUDA インストール

#### Ubuntu 22.04の場合

**1.1 NVIDIA Driverインストール**
```bash
# システム更新
sudo apt update && sudo apt upgrade -y

# 推奨ドライバー確認
ubuntu-drivers devices

# 自動インストール（推奨）
sudo ubuntu-drivers autoinstall

# 再起動
sudo reboot

# 確認
nvidia-smi
```

**期待される出力**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
|  0%   45C    P8    25W / 450W |    123MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**1.2 CUDA Toolkitインストール**
```bash
# CUDA 12.1のインストール
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-1 -y

# 環境変数設定
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 確認
nvcc --version
```

#### Windows 11の場合

**1.1 NVIDIA Driverインストール**
1. [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)にアクセス
2. GeForce RTX 4090を選択してダウンロード
3. インストーラーを実行
4. 再起動後、コマンドプロンプトで確認:
   ```cmd
   nvidia-smi
   ```

**1.2 CUDA Toolkitインストール**
1. [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)をダウンロード
2. インストーラーを実行
3. 環境変数が自動設定されることを確認
4. PowerShellで確認:
   ```powershell
   nvcc --version
   ```

---

### Step 2: Python環境構築

#### Ubuntu 22.04の場合

**2.1 Python 3.12インストール**
```bash
# deadsnakes PPA追加
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Python 3.12インストール
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# デフォルトPython設定（オプション）
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# 確認
python3 --version
```

**2.2 uvインストール**
```bash
# curlでインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# または pip経由
pip install uv

# PATHに追加
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 確認
uv --version
```

#### Windows 11の場合

**2.1 Python 3.12インストール**
1. [Python.org](https://www.python.org/downloads/)から Python 3.12をダウンロード
2. インストーラー実行時に **"Add Python to PATH"** をチェック
3. インストール完了後、PowerShellで確認:
   ```powershell
   python --version
   ```

**2.2 uvインストール**
```powershell
# pip経由でインストール
pip install uv

# 確認
uv --version
```

---

### Step 3: リポジトリセットアップ

**3.1 Gitがインストールされていることを確認**
```bash
# Ubuntu
sudo apt install git -y

# Windows
# Git for Windows をインストール: https://git-scm.com/download/win
```

**3.2 リポジトリクローン**
```bash
# 作業ディレクトリに移動
cd ~/projects  # Ubuntuの場合
# または
cd C:\Users\YourName\projects  # Windowsの場合

# リポジトリクローン
git clone https://github.com/ayutaz/supertonic.git
cd supertonic

# ブランチ確認（devブランチを使用）
git checkout dev
```

**3.3 Assets（ONNXモデル）ダウンロード** 🔴 **重要！**
```bash
# Hugging Faceからモデルとプリセットをダウンロード
git clone https://huggingface.co/Supertone/supertonic assets

# または、Git LFSがインストールされている場合:
# brew install git-lfs && git lfs install  # macOS
# sudo apt install git-lfs && git lfs install  # Ubuntu

# ディレクトリ構造確認
ls assets/onnx/
# 期待される出力:
# duration_predictor.onnx  text_encoder.onnx  vector_estimator.onnx
# vocoder.onnx  unicode_indexer.json  tts.json
```

**⚠️ 注意**: `assets/` ディレクトリがないと学習が開始できません！

---

### Step 4: 依存関係インストール

**4.1 trainingディレクトリに移動**
```bash
cd training
```

**4.2 依存関係インストール**
```bash
# uvで全依存関係をインストール（PyTorch含む）
uv sync

# 所要時間: 5-10分（初回のみ）
```

**4.3 PyTorch CUDA認識確認** 🔴 **重要！**
```bash
# Python起動して確認
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**期待される出力**:
```
PyTorch version: 2.2.0+cu121
CUDA available: True
CUDA version: 12.1
Device count: 1
Device name: NVIDIA GeForce RTX 4090
```

**トラブルシューティング**:
- `CUDA available: False` の場合:
  1. NVIDIA Driver を再インストール
  2. CUDA Toolkit を再インストール
  3. PyTorchを CUDA対応版で再インストール:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

---

### Step 5: データセット準備

**5.1 データディレクトリ作成**
```bash
# プロジェクトルートに戻る
cd ..

# ディレクトリ作成
mkdir -p data/raw/ljspeech
cd data/raw/ljspeech
```

**5.2 LJSpeechダウンロード**
```bash
# ダウンロード（2.6GB、5-15分）
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# または、ブラウザでダウンロード:
# http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# 解凍（3-5分）
tar -xvjf LJSpeech-1.1.tar.bz2

# プロジェクトルートに戻る
cd ../../..
```

**Windowsの場合**:
- ブラウザで http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 をダウンロード
- 7-ZipまたはWinRARで解凍
- `data/raw/ljspeech/` に配置

**ディレクトリ構造確認**:
```
data/raw/ljspeech/LJSpeech-1.1/
├── wavs/
│   ├── LJ001-0001.wav
│   ├── LJ001-0002.wav
│   └── ... (13,100 files)
├── metadata.csv
└── README
```

**5.3 メタデータ変換**
```bash
# プロジェクトルートで実行
python prepare_ljspeech.py
```

**期待される出力**:
```
[INFO] Reading metadata from data/raw/ljspeech/LJSpeech-1.1/metadata.csv
[INFO] Loaded 13100 samples

[INFO] Splitting dataset:
  Train: 12445 samples (95%)
  Val:   327 samples (2.5%)
  Test:  328 samples (2.5%)

[OK] Created data/processed/ljspeech_train.txt
     12445 samples
     First sample: wavs/LJ001-0001.wav | Printing, in the only sense...

[OK] Created data/processed/ljspeech_val.txt
     327 samples

[OK] Created data/processed/ljspeech_test.txt
     328 samples

[SUCCESS] Dataset preparation complete!
```

**5.4 データセット検証**
```bash
# 整合性チェック（2-3分）
python verify_dataset.py \
  data/raw/ljspeech/LJSpeech-1.1 \
  data/processed/ljspeech_train.txt \
  --check-audio \
  --num-samples 10
```

**期待される出力**:
```
============================================================
DATASET VERIFICATION
============================================================
Data directory:   data/raw/ljspeech/LJSpeech-1.1
Metadata file:    data/processed/ljspeech_train.txt
Audio check:      Enabled
Samples to check: 10
============================================================

[INFO] Total samples in metadata: 12445

[OK] Sample 0:
     File:     LJ001-0001.wav
     Duration: 7.02s
     SR:       22050Hz
     Text:     Printing, in the only sense with which we are at...

...

============================================================
VERIFICATION SUMMARY
============================================================
Total samples:    12445
Missing files:    0
Corrupted files:  0
Avg duration:     5.12s
Total checked:    10 samples
============================================================

[SUCCESS] All checks passed!

You can now start training:

cd training
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs \
  --ae-epochs 200 \
  --ttl-epochs 150 \
  --dp-epochs 100 \
  --batch-size 16
```

---

### Step 6: 学習開始（RTX 4090向け設定）

**6.1 学習実行**
```bash
cd training

# 推奨設定: batch_size=32（RTX 4090）
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs/ljspeech \
  --batch-size 32 \
  --ae-epochs 200 \
  --ttl-epochs 150 \
  --dp-epochs 100 \
  --learning-rate 2e-4
```

**段階ごとに実行する場合**:
```bash
# Stage 1のみ: Autoencoder（2-3日）
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs/ljspeech \
  --stage 1 \
  --ae-epochs 200 \
  --batch-size 32

# Stage 2のみ: TTL（1-2日）
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs/ljspeech \
  --stage 2 \
  --ttl-epochs 150 \
  --batch-size 32

# Stage 3のみ: Duration Predictor（1日）
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs/ljspeech \
  --stage 3 \
  --dp-epochs 100 \
  --batch-size 32
```

**6.2 学習パラメータ（RTX 4090最適化）**

| パラメータ | 値 | 説明 |
|-----------|---|------|
| `--batch-size` | 32 | RTX 4090推奨値（メモリ: ~20GB使用） |
| `--ae-epochs` | 200 | Autoencoder エポック数 |
| `--ttl-epochs` | 150 | TTL エポック数 |
| `--dp-epochs` | 100 | Duration Predictor エポック数 |
| `--learning-rate` | 2e-4 | 学習率 |

**メモリが不足する場合**:
- `--batch-size 32` → `16` → `8` に削減

---

### Step 7: 学習監視

**7.1 TensorBoard起動**

別のターミナルを開いて:
```bash
cd ~/projects/supertonic  # プロジェクトルート
tensorboard --logdir outputs/ljspeech/logs
```

**7.2 ブラウザでアクセス**
```
http://localhost:6006
```

**7.3 監視するメトリクス**

**Stage 1 (Autoencoder)**:
- `train/loss`: 総損失（下降すべき）
- `train/stft`: STFT損失
- `train/mel`: Mel-Spectrogram損失
- `train/adversarial`: 敵対的損失
- `train/discriminator`: Discriminator損失
- `val/stft_loss`, `val/mel_loss`: 検証損失

**Stage 2 (TTL)**:
- `train/flow_matching_loss`: Flow Matching損失（下降すべき）
- `val/flow_matching_loss`: 検証損失

**Stage 3 (DP)**:
- `train/mse_loss`: MSE損失（下降すべき）
- `train/mean_predicted_duration`: 予測平均長
- `train/mean_ground_truth_duration`: 真値平均長
- `val/mse_loss`: 検証損失

**7.4 学習進捗確認**

ログを監視:
```bash
tail -f outputs/ljspeech/logs/train.log
```

GPU使用率確認:
```bash
watch -n 1 nvidia-smi
```

期待される GPU 使用率:
- GPU Utilization: 95-100%
- Memory Usage: 18-22GB / 24GB

---

## 📈 推定学習時間（RTX 4090）

| Stage | エポック数 | 予想時間 | チェックポイントサイズ |
|-------|----------|---------|---------------------|
| Stage 1: Autoencoder | 200 | 2-3日 | ~500MB |
| Stage 2: TTL | 150 | 1-2日 | ~300MB |
| Stage 3: DP | 100 | 1日 | ~200MB |
| **合計** | - | **4-6日** | **~1GB** |

---

## 🔧 トラブルシューティング

### 1. CUDA not available

**症状**:
```python
import torch
torch.cuda.is_available()  # False
```

**解決策**:
1. NVIDIA Driverを再インストール
   ```bash
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```

2. CUDA Toolkitを再インストール

3. PyTorchをCUDA対応版で再インストール
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### 2. OOM (Out of Memory) エラー

**症状**:
```
RuntimeError: CUDA out of memory.
```

**解決策**:
1. バッチサイズを削減
   ```bash
   --batch-size 32  # → 16 → 8
   ```

2. Segment lengthを確認（デフォルト2.0秒）

3. GPU メモリをクリア
   ```bash
   uv run python -c "import torch; torch.cuda.empty_cache()"
   ```

### 3. Assets not found エラー

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: '../assets/onnx/tts.json'
```

**解決策**:
```bash
# プロジェクトルートで実行
git clone https://huggingface.co/Supertone/supertonic assets

# 確認
ls assets/onnx/
```

### 4. Import エラー

**症状**:
```
ModuleNotFoundError: No module named 'torch'
```

**解決策**:
```bash
cd training
uv sync
```

### 5. データセットが見つからない

**症状**:
```
FileNotFoundError: data/processed/ljspeech_train.txt not found
```

**解決策**:
```bash
# メタデータを再作成
python prepare_ljspeech.py
```

### 6. 学習が進まない

**症状**:
- Loss が下がらない
- GPU 使用率が低い

**解決策**:
1. 学習率を調整
   ```bash
   --learning-rate 1e-4  # または 5e-5
   ```

2. データの前処理を確認
   ```bash
   python verify_dataset.py ... --check-audio --num-samples 100
   ```

3. チェックポイントから再開
   ```bash
   # trainers/*_trainer.py 内で resume 機能を使用
   ```

---

## ✅ セットアップチェックリスト

完了したら ✅ をつけてください:

### 環境構築
- [ ] NVIDIA Driver インストール済み (`nvidia-smi` で確認)
- [ ] CUDA Toolkit インストール済み (`nvcc --version` で確認)
- [ ] Python 3.12+ インストール済み (`python --version` で確認)
- [ ] uv インストール済み (`uv --version` で確認)

### リポジトリ
- [ ] リポジトリクローン済み (`git clone https://github.com/ayutaz/supertonic.git`)
- [ ] devブランチにチェックアウト済み (`git checkout dev`)
- [ ] Assets ダウンロード済み (`git clone https://huggingface.co/Supertone/supertonic assets`)
- [ ] `assets/onnx/tts.json` 存在確認済み

### 依存関係
- [ ] `cd training && uv sync` 実行済み
- [ ] PyTorch GPU 認識確認済み (`torch.cuda.is_available() == True`)
- [ ] GPU名確認済み (`torch.cuda.get_device_name(0)` == "NVIDIA GeForce RTX 4090")

### データセット
- [ ] LJSpeech ダウンロード済み (2.6GB)
- [ ] `data/raw/ljspeech/LJSpeech-1.1/` 存在確認
- [ ] `python prepare_ljspeech.py` 実行済み
- [ ] `data/processed/ljspeech_train.txt` 存在確認 (12445 samples)
- [ ] `python verify_dataset.py ...` 実行済み (All checks passed)

### 学習準備
- [ ] TensorBoard インストール確認 (`tensorboard --version`)
- [ ] 学習コマンド準備完了
- [ ] ストレージ空き容量確認 (最低300GB)

---

## 📚 関連ドキュメント

- **CLAUDE.md**: プロジェクト概要と学習パイプライン実装状況
- **training/TRAINING_PIPELINE.md**: 学習パイプライン詳細ガイド
- **training/PROGRESS_SUMMARY.md**: 実装完了報告
- **README.md**: Supertonic 推論ガイド

---

## 🎯 次のステップ

セットアップ完了後:

1. **学習開始**
   ```bash
   cd training
   uv run python scripts/train_pipeline.py --config ... --batch-size 32
   ```

2. **TensorBoard起動**
   ```bash
   tensorboard --logdir ../outputs/ljspeech/logs
   ```

3. **学習監視**
   - TensorBoard: http://localhost:6006
   - GPU使用率: `watch -n 1 nvidia-smi`
   - ログ: `tail -f ../outputs/ljspeech/logs/train.log`

4. **4-6日後**: 学習完了、音声生成テスト

---

**作成日**: 2025年1月
**最終更新**: 2025年1月
**対象GPU**: NVIDIA RTX 4090 (24GB)
**推定セットアップ時間**: 1～1.5時間
