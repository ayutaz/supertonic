# SupertonicTTS 学習ガイド

本ガイドでは、SupertonicTTSモデルをゼロから学習する手順を詳しく説明します。

## 目次

1. [環境構築](#環境構築)
2. [データ準備](#データ準備)
3. [Phase 1: Speech Autoencoder学習](#phase-1-speech-autoencoder学習)
4. [Phase 2: Text-to-Latent学習](#phase-2-text-to-latent学習)
5. [Phase 3: Duration Predictor学習](#phase-3-duration-predictor学習)
6. [Phase 4: 統合とファインチューニング](#phase-4-統合とファインチューニング)
7. [評価とテスト](#評価とテスト)
8. [トラブルシューティング](#トラブルシューティング)

---

## 環境構築

### 1. 前提条件

- **OS**: Linux (Ubuntu 20.04以上推奨)
- **Python**: 3.12以上
- **GPU**: CUDA対応GPU (推奨: 4x RTX 4090以上)
- **RAM**: 128GB以上
- **ストレージ**: 500GB以上の空き容量

### 2. uvのインストール

```bash
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# パスを通す
source $HOME/.cargo/env
```

### 3. プロジェクトのセットアップ

```bash
# プロジェクトディレクトリに移動
cd training

# 依存関係をインストール
uv sync --extra dev
```

### 4. CUDA環境の確認

```bash
# PyTorchのCUDA対応を確認
uv run python -c "import torch; print(torch.cuda.is_available())"
# True が出力されればOK
```

---

## データ準備

### 1. データセットのダウンロード

#### JSUT (Japanese Speech Corpus for Text-to-Speech)

```bash
# JSUTのダウンロード
mkdir -p data/raw/jsut
cd data/raw/jsut
wget https://sites.google.com/site/shinnosuketakamichi/publication/jsut_ver1.1.zip
unzip jsut_ver1.1.zip
cd ../../../
```

#### JVS (Japanese versatile speech corpus)

```bash
# JVSのダウンロード
mkdir -p data/raw/jvs
cd data/raw/jvs
# JVSの公式サイトから手動でダウンロードしてください
# https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
cd ../../../
```

### 2. メタデータの作成

```bash
# JSUTのメタデータ作成
uv run python scripts/prepare_jsut.py \
  --jsut-dir data/raw/jsut \
  --output data/processed/jsut_metadata.txt

# JVSのメタデータ作成
uv run python scripts/prepare_jvs.py \
  --jvs-dir data/raw/jvs \
  --output data/processed/jvs_metadata.txt
```

### 3. Unicode文字辞書の構築

```bash
# 文字辞書を構築
uv run python scripts/build_char_dict.py \
  --metadata data/processed/jsut_metadata.txt data/processed/jvs_metadata.txt \
  --output resources/metadata/char_dict/ja/char_dict.json
```

---

## Phase 1: Speech Autoencoder学習

Speech Autoencoderは、音声を24次元の潜在表現に圧縮・復元するモデルです。

### 学習設定

- **学習ステップ**: 1.5M steps
- **バッチサイズ**: 64
- **GPU数**: 4 (推奨)
- **所要時間**: 約3-4週間

### 学習コマンド

```bash
# シングルGPU
uv run python scripts/train_autoencoder.py \
  --config configs/autoencoder.yaml \
  --data-dir data/raw \
  --metadata data/processed/jsut_metadata.txt data/processed/jvs_metadata.txt

# マルチGPU (DDP)
uv run torchrun --nproc_per_node=4 scripts/train_autoencoder.py \
  --config configs/autoencoder.yaml \
  --data-dir data/raw \
  --metadata data/processed/jsut_metadata.txt data/processed/jvs_metadata.txt
```

### 重要なハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `latent_dim` | 24 | 潜在次元数 |
| `sample_rate` | 44100 | サンプリングレート |
| `hop_length` | 512 | STFTのホップ長 (86 Hz相当) |
| `n_mels` | 228 | Melフィルターバンク数 |
| `learning_rate` | 2e-4 | 学習率 |
| `stft_loss_weight` | 1.0 | STFT損失の重み |
| `mel_loss_weight` | 45.0 | Mel損失の重み |

### モニタリング

```bash
# TensorBoard
uv run tensorboard --logdir logs/autoencoder

# Weights & Biases
# wandbの設定は configs/base.yaml で行ってください
```

### チェックポイント

チェックポイントは `checkpoints/autoencoder/` に保存されます。

```bash
# ベストモデルの確認
ls -lh checkpoints/autoencoder/best_*.pt
```

---

## Phase 2: Text-to-Latent学習

Text-to-Latent (TTL) は、テキストから音声の潜在表現を生成するFlow Matchingベースのモデルです。

### 学習設定

- **学習ステップ**: 700k steps
- **バッチサイズ**: 64 (effective 384 with Ke=6)
- **GPU数**: 4-8 (推奨)
- **所要時間**: 約6-8週間

### 前処理: 潜在表現の抽出

TTL学習の前に、学習済みAutoencoderで全データの潜在表現を抽出します。

```bash
# 潜在表現を抽出
uv run python scripts/extract_latents.py \
  --autoencoder-checkpoint checkpoints/autoencoder/best_model.pt \
  --data-dir data/raw \
  --metadata data/processed/jsut_metadata.txt data/processed/jvs_metadata.txt \
  --output-dir data/processed/latents
```

### 学習コマンド

```bash
# マルチGPU (DDP)
uv run torchrun --nproc_per_node=4 scripts/train_text_to_latent.py \
  --config configs/text_to_latent.yaml \
  --latent-dir data/processed/latents \
  --metadata data/processed/jsut_metadata.txt data/processed/jvs_metadata.txt \
  --char-dict resources/metadata/char_dict/ja/char_dict.json
```

### 重要なハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `chunk_compress_factor` | 6 | 時間圧縮率 (86 Hz → 14.3 Hz) |
| `n_batch_expand` | 6 | バッチ拡張係数 (Ke) |
| `learning_rate` | 1e-4 | 学習率 |
| `total_step` | 32 | 推論時のNFE (ステップ数) |
| `cfg_scale` | 3.0 | Classifier-Free Guidanceスケール |
| `rotary_base` | 10000 | LARoPEのベース値 |
| `rotary_scale` | 10 | LARoPEのスケール |

### Context-Sharing Batch Expansion

TTLでは、Context-Sharing Batch Expansionを使用して効率的に学習します。

```
実バッチサイズ: 64
拡張係数 (Ke): 6
実効バッチサイズ: 64 × 6 = 384
```

各サンプルを6つの異なるノイズレベルで同時に処理することで、
計算効率を大幅に向上させます。

---

## Phase 3: Duration Predictor学習

Duration Predictorは、テキストから発話時間を予測するO(1)モデルです。

### 学習設定

- **学習ステップ**: 3k steps (非常に短い!)
- **バッチサイズ**: 128
- **GPU数**: 1-2
- **所要時間**: 約1-2日

### 前処理: 時間長の抽出

TTLで生成した音声から実際の時間長を抽出します。

```bash
# 時間長を抽出
uv run python scripts/extract_durations.py \
  --ttl-checkpoint checkpoints/text_to_latent/best_model.pt \
  --autoencoder-checkpoint checkpoints/autoencoder/best_model.pt \
  --metadata data/processed/jsut_metadata.txt data/processed/jvs_metadata.txt \
  --output-dir data/processed/durations
```

### 学習コマンド

```bash
# シングルGPU
uv run python scripts/train_duration_predictor.py \
  --config configs/duration_predictor.yaml \
  --duration-dir data/processed/durations \
  --metadata data/processed/jsut_metadata.txt data/processed/jvs_metadata.txt \
  --char-dict resources/metadata/char_dict/ja/char_dict.json
```

### 重要なハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `learning_rate` | 5e-4 | 学習率 |
| `n_style` | 8 | スタイルトークン数 (少数) |
| `hdim` | 128 | 隠れ層次元数 |

---

## Phase 4: 統合とファインチューニング

3つのコンポーネントを統合し、エンドツーエンドでファインチューニングします。

### 統合

```bash
# 統合スクリプト
uv run python scripts/integrate_models.py \
  --autoencoder-checkpoint checkpoints/autoencoder/best_model.pt \
  --ttl-checkpoint checkpoints/text_to_latent/best_model.pt \
  --dp-checkpoint checkpoints/duration_predictor/best_model.pt \
  --output checkpoints/integrated/supertonic_v1.0.pt
```

### ファインチューニング (オプション)

```bash
# エンドツーエンドでファインチューニング
uv run python scripts/finetune_integrated.py \
  --checkpoint checkpoints/integrated/supertonic_v1.0.pt \
  --config configs/finetune.yaml \
  --data-dir data/raw \
  --metadata data/processed/jsut_metadata.txt data/processed/jvs_metadata.txt
```

---

## 評価とテスト

### 推論テスト

```bash
# 単一音声の生成
uv run python scripts/inference.py \
  --checkpoint checkpoints/integrated/supertonic_v1.0.pt \
  --text "こんにちは、世界！" \
  --voice-style assets/voice_styles/F1.json \
  --output output.wav

# バッチ生成
uv run python scripts/inference.py \
  --checkpoint checkpoints/integrated/supertonic_v1.0.pt \
  --batch \
  --voice-style assets/voice_styles/M1.json assets/voice_styles/F1.json \
  --text "テスト文章1" "テスト文章2" \
  --output-dir outputs/
```

### 評価指標

```bash
# 客観評価
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/integrated/supertonic_v1.0.pt \
  --test-data data/processed/test_metadata.txt \
  --metrics mel_error duration_error cer

# 主観評価 (MOS) - Gradioアプリ
uv run python demo/gradio_app.py \
  --checkpoint checkpoints/integrated/supertonic_v1.0.pt
```

### ONNX変換

```bash
# ONNXにエクスポート
uv run python scripts/export_onnx.py \
  --checkpoint checkpoints/integrated/supertonic_v1.0.pt \
  --output-dir onnx_models/ \
  --opset-version 17

# ONNX推論テスト
uv run python scripts/test_onnx.py \
  --onnx-dir onnx_models/ \
  --text "ONNX推論テスト" \
  --voice-style assets/voice_styles/M1.json \
  --output test_onnx.wav
```

---

## トラブルシューティング

### 1. GPU メモリ不足 (OOM)

**症状**: `CUDA out of memory` エラー

**解決策**:
```yaml
# configs/*.yaml でバッチサイズを減らす
training:
  batch_size: 32  # 64 → 32
  gradient_accumulation_steps: 2  # 実効バッチサイズを維持
```

### 2. 学習が収束しない

**症状**: 損失が減少しない、または発散する

**解決策**:
- 学習率を下げる: `lr: 5e-5` (1e-4 → 5e-5)
- Gradient Clippingを確認: `gradient_clip: 1.0`
- Warmup Stepsを増やす: `warmup_steps: 10000`

### 3. 音質が悪い

**症状**: 生成された音声にノイズや歪みがある

**チェックポイント**:
1. **Autoencoder**: Mel損失が十分に低いか確認
2. **TTL**: NFE (推論ステップ数) を増やす (32 → 64)
3. **Duration Predictor**: 時間長予測が正確か確認

### 4. 学習が遅い

**最適化案**:
- Mixed Precision Training: `use_amp: true` (base.yaml)
- DataLoaderの最適化: `num_workers: 16`, `pin_memory: true`
- Flash Attention: PyTorch 2.0以上で自動有効化

### 5. データセットが小さい

**対策**:
- データ拡張を有効化:
  ```yaml
  augmentation:
    time_stretch:
      enabled: true
    pitch_shift:
      enabled: true
  ```
- 事前学習モデルの利用 (今後のリリース)

---

## 参考資料

- [論文解説ドキュメント](../../論文解説.md)
- [SupertonicTTS論文](https://arxiv.org/abs/2503.23108)
- [プロジェクトREADME](../README.md)

---

## サポート

質問やバグ報告は、GitHubのIssuesでお願いします。

Happy Training! 🎉
