# SupertonicTTS 統合学習パイプライン

3段階学習を自動化する統合学習スクリプトです。

## アーキテクチャ

SupertonicTTSの学習は以下の3段階で構成されます:

```
Stage 1: Speech Autoencoder
    音声 → 24次元潜在表現 → 音声
    目的: 高品質な音声圧縮・復元

Stage 2: TTL (Text-to-Latent)
    テキスト + 音声スタイル → 潜在表現
    目的: テキストから音声潜在表現を生成 (Flow Matching)
    Frozen: Stage 1のAutoencoder Encoder

Stage 3: Duration Predictor
    テキスト + 音声スタイル → 発話長 (秒)
    目的: 発話全体の長さを予測
    Frozen: Stage 1のAutoencoder Encoder
```

## 前提条件

### データセット準備

1. **Autoencoder学習用** (Stage 1)
   - 音声ファイルのみ（テキスト不要）
   - 推奨: 10-100時間の音声データ
   - メタデータフォーマット (TXT):
     ```
     path/to/audio1.wav
     path/to/audio2.wav
     ...
     ```

2. **TTL/DP学習用** (Stage 2, 3)
   - 音声とテキストのペア
   - 推奨: 10-100時間の音声データ + テキスト
   - メタデータフォーマット (TXT):
     ```
     path/to/audio1.wav|This is the transcription.
     path/to/audio2.wav|Another example text.
     ...
     ```

### 環境構築

```bash
cd training
uv sync  # 依存関係インストール
```

## 使い方

### 1. 完全なパイプライン実行

3段階すべてを自動実行:

```bash
cd training

uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_train.txt \
  --output-dir outputs \
  --ae-epochs 100 \
  --ttl-epochs 100 \
  --dp-epochs 100 \
  --batch-size 16 \
  --learning-rate 2e-4
```

### 2. 特定の段階のみ実行

**Stage 1のみ (Autoencoder)**:

```bash
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_train.txt \
  --output-dir outputs \
  --stage 1 \
  --ae-epochs 100 \
  --batch-size 16
```

**Stage 2のみ (TTL)**:

```bash
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_train.txt \
  --output-dir outputs \
  --stage 2 \
  --ttl-epochs 100 \
  --batch-size 16
```

**Stage 3のみ (DP)**:

```bash
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_train.txt \
  --output-dir outputs \
  --stage 3 \
  --dp-epochs 100 \
  --batch-size 16
```

### 3. 段階をスキップ

既に学習済みの段階をスキップ:

```bash
# Autoencoderをスキップして、TTLとDPのみ学習
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_train.txt \
  --output-dir outputs \
  --skip-stages autoencoder \
  --ttl-epochs 100 \
  --dp-epochs 100
```

## コマンドラインオプション

### 必須オプション

- `--config`: tts.json設定ファイルパス
- `--data-dir`: データディレクトリ
- `--metadata`: メタデータファイルパス（複数指定可）

### オプション

**出力設定:**
- `--output-dir`: 出力ディレクトリ (デフォルト: `outputs`)

**デバイス:**
- `--device`: 使用デバイス (`cuda` or `cpu`)

**学習設定:**
- `--ae-epochs`: Autoencoderエポック数 (デフォルト: 100)
- `--ttl-epochs`: TTLエポック数 (デフォルト: 100)
- `--dp-epochs`: DPエポック数 (デフォルト: 100)
- `--batch-size`: バッチサイズ (デフォルト: 16)
- `--learning-rate`: 学習率 (デフォルト: 2e-4)

**パイプライン制御:**
- `--stage`: 実行する段階 (1=AE, 2=TTL, 3=DP)
- `--skip-stages`: スキップする段階 (例: `autoencoder ttl`)

## 出力

### ディレクトリ構造

```
outputs/
├── checkpoints/
│   ├── autoencoder/
│   │   ├── autoencoder_step5000.pt
│   │   ├── autoencoder_step10000.pt
│   │   └── autoencoder_final.pt
│   ├── ttl/
│   │   └── ttl_final.pt
│   └── dp/
│       └── dp_final.pt
└── logs/
    ├── autoencoder/
    ├── ttl/
    └── dp/
```

### チェックポイント形式

各チェックポイントには以下が含まれます:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "global_step": global_step,
    "epoch": epoch,
    "scaler_state_dict": scaler.state_dict(),  # (if AMP)
    "scheduler_state_dict": scheduler.state_dict(),  # (if scheduler)
}
```

## 学習監視

### TensorBoard

TensorBoardでログを確認:

```bash
tensorboard --logdir outputs/logs
```

ブラウザで `http://localhost:6006` を開く

### ログメトリクス

**Stage 1 (Autoencoder):**
- `train/loss`: 総損失
- `train/stft`: STFT損失
- `train/mel`: Mel-Spectrogram損失
- `train/adversarial`: 敵対的損失
- `train/feature_matching`: Feature Matching損失
- `train/discriminator`: Discriminator損失
- `val/stft_loss`, `val/mel_loss`: 検証損失

**Stage 2 (TTL):**
- `train/flow_matching_loss`: Flow Matching損失
- `val/flow_matching_loss`: 検証損失

**Stage 3 (DP):**
- `train/mse_loss`: MSE損失
- `train/mean_predicted_duration`: 予測平均長
- `train/mean_ground_truth_duration`: 真値平均長
- `val/mse_loss`: 検証損失

## 推奨設定

### GPU別推奨設定

| GPU | Batch Size | Segment Length | Epochs (AE) | 学習時間 (10h data) |
|-----|-----------|----------------|-------------|-------------------|
| RTX 3060 (12GB) | 8-16 | 2.0s | 100 | 3-7日 |
| RTX 4070 (12GB) | 16-24 | 2.0s | 100 | 2-5日 |
| RTX 4090 (24GB) | 32-48 | 2.0-3.0s | 100 | 1-3日 |
| A100 (40GB) | 64-96 | 3.0-4.0s | 100 | 0.5-1.5日 |

### データセット別推奨エポック数

| データサイズ | AE Epochs | TTL Epochs | DP Epochs |
|-----------|-----------|-----------|----------|
| 10時間 | 200-300 | 150-200 | 100-150 |
| 50時間 | 100-150 | 80-120 | 50-80 |
| 100時間 | 80-100 | 50-80 | 30-50 |

## トラブルシューティング

### OOM (Out of Memory) エラー

1. `--batch-size` を減らす (16 → 8 → 4)
2. Segment lengthを減らす（train_pipeline.py内で調整、デフォルト2.0秒）
3. Mixed Precisionを有効化（デフォルトで有効）

### 学習が進まない

1. 学習率を調整 (`--learning-rate 1e-4` など)
2. Gradient clippingを確認（trainer内でデフォルト1.0）
3. データの前処理を確認（正規化、無音削除）

### チェックポイントから再開

```bash
# 各段階の個別スクリプトを使用
cd training

# Autoencoder
uv run python scripts/train_autoencoder.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_train.txt \
  --resume outputs/checkpoints/autoencoder/autoencoder_step10000.pt
```

## 次のステップ

学習完了後:

1. **評価**: 音声品質評価 (MOS, RTF測定)
2. **ONNXエクスポート**: PyTorch → ONNX変換
3. **推論**: 学習済みモデルで音声生成

詳細は各スクリプトのドキュメントを参照してください。

## 注意事項

### 現在の実装状況

- ✅ **Stage 1 (Autoencoder)**: 完全実装、テスト済み
- ✅ **Stage 2 (TTL)**: 完全実装、テスト済み
- ✅ **Stage 3 (DP)**: 完全実装、テスト済み
- ✅ **TTSDataset**: テキスト-音声ペアのデータセット完全実装
- ✅ **Unicode処理**: NFKD正規化とUnicode→Text IDs変換完全実装

全3段階の学習パイプラインが完全に実装され、すぐに学習を開始できます！

### 今後の改善点

1. Wandb統合（TensorBoardは既に実装済み）
2. 分散学習サポート（マルチGPU対応）
3. 自動ハイパーパラメータチューニング
4. より詳細なメトリクスと可視化
5. 音声品質評価 (MOS, RTF測定)
6. ONNXエクスポート機能
