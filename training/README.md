# SupertonicTTS Training Implementation

SupertonicTTSの学習コード実装プロジェクトです。ゼロショット音声合成モデルをフルスクラッチで学習します。

## 概要

このプロジェクトは、論文 "Supertonic: A Text-To-Speech Foundation Model Capable of Adapting to Novel Voices from Short Prompts" の完全な学習実装を提供します。

### 主要コンポーネント

1. **Speech Autoencoder (AE)**: 音声を24次元の潜在表現に圧縮
2. **Text-to-Latent (TTL)**: Flow Matchingベースの生成モデル
3. **Duration Predictor (DP)**: O(1)の発話レベル時間長予測

### アーキテクチャハイライト

- パラメータ数: 44M (AE: 0.5M, TTL: 18.5M, DP: 25M)
- サンプリングレート: 44.1kHz
- 潜在次元: 24-dim @ 86 Hz
- 時間圧縮: 6倍 (86 Hz → 14.3 Hz)
- NFE (推論ステップ数): 32 (推奨)

## セットアップ

### 前提条件

- Python 3.12以上
- uv (Pythonパッケージマネージャー)
- CUDA対応GPU (推奨: 4x RTX 4090以上)
- 最低500GB以上のストレージ

### 環境構築

```bash
# uvのインストール (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトディレクトリに移動
cd training

# 依存関係のインストール
uv sync

# 開発依存関係も含める場合
uv sync --extra dev

# デモ用依存関係
uv sync --extra demo
```

## プロジェクト構造

```
training/
├── configs/           # 設定ファイル (YAML)
├── data/             # データ処理関連
│   ├── unicode_processor.py
│   ├── preprocessing.py
│   └── datasets.py
├── models/           # モデル実装
│   ├── autoencoder/
│   ├── text_to_latent/
│   ├── duration_predictor/
│   └── common/
├── losses/           # 損失関数
├── trainers/         # 学習ループ
├── inference/        # 推論パイプライン
├── evaluation/       # 評価指標
├── export/           # ONNX変換
├── scripts/          # 学習・推論スクリプト
├── demo/             # Gradioデモ
└── docs/             # ドキュメント
```

## 使用方法

### データ準備

```bash
# JSUTデータセットのダウンロードと前処理
uv run scripts/prepare_jsut.py

# JVSデータセットのダウンロードと前処理
uv run scripts/prepare_jvs.py
```

### 学習

```bash
# Speech Autoencoderの学習
uv run scripts/train_autoencoder.py --config configs/autoencoder.yaml

# Text-to-Latentの学習
uv run scripts/train_text_to_latent.py --config configs/text_to_latent.yaml

# Duration Predictorの学習
uv run scripts/train_duration_predictor.py --config configs/duration_predictor.yaml
```

### 推論

```bash
# 音声合成
uv run scripts/inference.py \
  --text "合成したいテキスト" \
  --voice-style path/to/voice_style.json \
  --output output.wav
```

### ONNX変換

```bash
# モデルをONNXに変換
uv run scripts/export_onnx.py \
  --checkpoint checkpoints/best_model.pt \
  --output-dir onnx_models/
```

## 開発ガイド

詳細な実装ガイドは以下を参照してください:

- [学習ガイド](docs/training_guide.md) - 学習プロセスの詳細
- [アーキテクチャドキュメント](../論文解説.md) - モデルアーキテクチャの詳細解析

## 学習スケジュール

全体の学習には約15-22週間を見込んでいます:

| フェーズ | 期間 | 内容 |
|---------|------|------|
| Phase 1 | 2-3週 | 環境構築・データ準備 |
| Phase 2 | 3-4週 | Speech Autoencoder実装・学習 |
| Phase 3 | 6-8週 | Text-to-Latent実装・学習 |
| Phase 4 | 1-2週 | Duration Predictor実装・学習 |
| Phase 5 | 2-3週 | 統合・ファインチューニング |
| Phase 6 | 1-2週 | 最適化・デプロイ |

## リソース要件

### 最小構成
- GPU: 4x RTX 4090 (24GB VRAM each)
- RAM: 128GB
- ストレージ: 500GB SSD

### 推奨構成
- GPU: 8x A100 (80GB VRAM each)
- RAM: 256GB
- ストレージ: 2TB NVMe SSD

## ライセンス

MIT License

## 参考文献

```bibtex
@article{supertonic2024,
  title={Supertonic: A Text-To-Speech Foundation Model Capable of Adapting to Novel Voices from Short Prompts},
  author={...},
  journal={arXiv preprint arXiv:2503.23108},
  year={2024}
}
```

## 謝辞

本実装は論文 "Supertonic: A Text-To-Speech Foundation Model" に基づいています。
