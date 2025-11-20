# SupertonicTTS 再現実装プラン

## プロジェクト概要

SupertonicTTSの完全な再現実装を行い、日本語音声合成モデルを学習する。
論文とONNX実装（tts.json）を基に、PyTorchでゼロから実装する。

### 目標

- **最終成果物**: 日本語対応のゼロショットTTSモデル
- **学習データ**: JSUT (10h) + JVS (30h) = 40-50時間
- **推定期間**: 15-22週間
- **リソース**: 4x RTX 4090以上のGPU

---

## マイルストーン

| マイルストーン | 期間 | 成果物 | 完了条件 |
|--------------|------|--------|---------|
| **M0: 初期セットアップ** | Week 0 | プロジェクト基盤、共通モジュール | ✅ 完了 |
| **M1: Speech Autoencoder** | Week 1-4 | 学習済みAEモデル | Mel損失 < 1.0 |
| **M2: Text-to-Latent** | Week 5-12 | 学習済みTTLモデル | Flow Matching収束 |
| **M3: Duration Predictor** | Week 13-14 | 学習済みDPモデル | Duration誤差 < 10% |
| **M4: 統合** | Week 15-17 | 統合モデル | エンドツーエンド動作 |
| **M5: 最適化** | Week 18-20 | ONNX変換、デモ | 推論速度 < 1s (RTF) |
| **M6: リリース** | Week 21-22 | ドキュメント、公開 | GitHub公開 |

---

## Phase 0: 初期セットアップ ✅ 完了

### タスク
- [x] ディレクトリ構造作成
- [x] pyproject.toml、.python-version、README作成
- [x] 設定ファイル作成（4つのYAML）
- [x] 共通モジュール実装（ConvNeXt、Attention、LARoPE、基本レイヤー）
- [x] データ処理基盤実装（Unicode処理、音声前処理、Dataset）
- [x] 学習ガイド作成

### 成果物
- `training/` ディレクトリ完全構築
- 共通モジュール（models/common/）
- データ処理基盤（data/）
- ドキュメント（docs/）

---

## Phase 1: Speech Autoencoder実装

**期間**: 3-4週間
**目標**: 音声を24次元の潜在表現に圧縮・復元

### 1.1 モデル実装

#### Encoder (models/autoencoder/encoder.py)
- [ ] `MelSpectrogramProcessor`: Mel-Spectrogram抽出
  - n_fft=2048, hop_length=512, n_mels=228
  - Delta, Delta-Delta特徴量
- [ ] `AcousticEncoder`: ConvNeXtベースのエンコーダー
  - 初期Conv (kernel=7)
  - ConvNeXt Stack (10 layers, dilation=[1,1,1,1,1,1,1,1,1,1])
  - Projection to latent (24-dim)

#### Decoder (models/autoencoder/decoder.py)
- [ ] `AcousticDecoder`: ConvNeXtベースのデコーダー
  - ConvNeXt Stack (10 layers, dilation=[1,2,4,1,2,4,1,1,1,1])
  - Decoder Head (3層MLP)
- [ ] `Vocoder`: Mel-Spectrogram → Waveform
  - Griffin-Lim または Neural Vocoder

#### Discriminator (models/autoencoder/discriminator.py)
- [ ] `MultiScaleDiscriminator`: 複数スケールの識別器
  - 3スケール (full, 1/2, 1/4)
  - PatchGAN構造

#### 統合 (models/autoencoder/autoencoder.py)
- [ ] `SpeechAutoencoder`: Encoder + Decoder統合
  - Forward: wav → latent → reconstructed_wav
  - Encode/Decode分離メソッド

### 1.2 損失関数実装 (losses/autoencoder_losses.py)

- [ ] `MultiScaleSTFTLoss`: Multi-Scale STFT Loss
  - 5スケール [2048, 1024, 512, 256, 128]
  - Magnitude + Phase損失
- [ ] `MelSpectrogramLoss`: Mel-Spectrogram L1 Loss
  - 重み: 45.0
- [ ] `FeatureMatchingLoss`: Discriminatorの中間層マッチング
  - 重み: 2.0
- [ ] `AdversarialLoss`: GAN損失
  - Hinge Loss
  - Generator重み: 1.0
  - Discriminator重み: 1.0

### 1.3 学習ループ (trainers/autoencoder_trainer.py)

- [ ] `AutoencoderTrainer`: 学習ループ実装
  - Generator学習ステップ
  - Discriminator学習ステップ（10k steps後開始）
  - Validation
  - Checkpointing (5k stepsごと)
  - Logging (WandB, TensorBoard)
  - Mixed Precision Training (AMP)
  - EMA（オプション）

### 1.4 スクリプト (scripts/)

- [ ] `train_autoencoder.py`: 学習メインスクリプト
  - 引数解析
  - データローダー構築
  - モデル初期化
  - Trainer実行
- [ ] `evaluate_autoencoder.py`: 評価スクリプト
  - Mel損失、STFT損失計算
  - サンプル音声生成
- [ ] `inference_autoencoder.py`: 推論スクリプト
  - 音声 → 潜在表現 → 復元音声

### 1.5 テストとデバッグ

- [ ] ユニットテスト作成
  - Encoder/Decoderの形状確認
  - 損失関数の動作確認
- [ ] 小規模データでの動作確認
  - 1エポックの学習
  - Overfittingテスト
- [ ] 本格学習開始
  - 1.5M stepsまで学習

### 成果物
- 学習済みSpeech Autoencoderモデル
- Mel損失 < 1.0達成
- 復元音声サンプル

---

## Phase 2: Text-to-Latent実装

**期間**: 6-8週間
**目標**: テキストから音声潜在表現を生成（Flow Matching）

### 2.1 前処理

- [ ] `scripts/extract_latents.py`: 潜在表現抽出
  - 学習済みAutoencoderで全データを処理
  - 潜在表現を`data/processed/latents/`に保存

### 2.2 モデル実装

#### Text Encoder (models/text_to_latent/text_encoder.py)
- [ ] `CharacterEmbedder`: 文字埋め込み
  - Unicode文字 → 256-dim embedding
- [ ] `TextEncoder`: テキスト表現抽出
  - ConvNeXt (6 layers)
  - Attention Encoder (4 layers, 4 heads)
  - Projection Out (256-dim)

#### Style Encoder (models/text_to_latent/style_encoder.py)
- [ ] `StyleEncoder`: スタイル表現抽出
  - Projection In (temporal compression 6x)
  - ConvNeXt (6 layers)
  - Style Token Layer (50 tokens)
  - Output: 256-dim style embedding

#### Speech-Prompted Text Encoder (models/text_to_latent/speech_prompted_text_encoder.py)
- [ ] `SpeechPromptedTextEncoder`: テキスト+スタイル融合
  - Cross-modal attention (2 heads)
  - テキスト表現にスタイルを注入

#### Vector Estimator (models/text_to_latent/vector_estimator.py)
- [ ] `TimeEncoder`: 時刻埋め込み
  - Sinusoidal embedding (64-dim)
- [ ] `VectorEstimatorBlock`: メインブロック
  - Time Conditioning (FiLM)
  - Style Conditioning (FiLM)
  - Text Conditioning (Cross-Attention with LARoPE)
  - ConvNeXt blocks (3種類: dilation [1,2,4,8], [1], [1])
- [ ] `VectorEstimator`: Vector Field全体
  - Projection In (temporal compression 6x)
  - 4 Main Blocks
  - Last ConvNeXt (4 layers)
  - Projection Out (temporal decompression)

#### Unconditional Masker (models/text_to_latent/uncond_masker.py)
- [ ] `UnconditionalMasker`: CFG用マスク
  - テキスト+スタイル両方マスク: 4%
  - テキストのみマスク: 1%
  - ガウシアンノイズ注入 (std=0.1)

#### Batch Expander (models/text_to_latent/batch_expander.py)
- [ ] `ContextSharingBatchExpander`: バッチ拡張
  - Ke=6の拡張
  - 各サンプルを6つの異なる時刻tで処理

#### 統合 (models/text_to_latent/text_to_latent.py)
- [ ] `TextToLatent`: TTL全体統合
  - Text Encoder + Style Encoder
  - Speech-Prompted Text Encoder
  - Vector Estimator
  - Flow Matching推論実装（Euler法）

### 2.3 損失関数実装 (losses/ttl_losses.py)

- [ ] `FlowMatchingLoss`: Flow Matching損失
  - MSE between predicted velocity and target velocity
  - v_target = (x_1 - x_0) / (1 - t)
- [ ] `StyleRegularizationLoss`: スタイル正則化
  - スタイル埋め込みのL2正則化
  - 重み: 0.01

### 2.4 学習ループ (trainers/ttl_trainer.py)

- [ ] `TextToLatentTrainer`: 学習ループ実装
  - Batch Expansion適用
  - Flow Matching損失計算
  - CFG用のUnconditionalマスク
  - Validation（サンプル生成）
  - Checkpointing (5k stepsごと)
  - EMA (decay=0.9999)
  - Logging

### 2.5 スクリプト (scripts/)

- [ ] `extract_latents.py`: 潜在表現抽出（前処理）
- [ ] `train_text_to_latent.py`: 学習メインスクリプト
- [ ] `evaluate_text_to_latent.py`: 評価スクリプト
  - Flow Matching損失
  - サンプル音声生成（多様性確認）
- [ ] `inference_text_to_latent.py`: 推論スクリプト
  - テキスト → 潜在表現生成

### 2.6 テストとデバッグ

- [ ] ユニットテスト
  - 各モジュールの形状確認
  - Batch Expansionの動作確認
  - LARoPEの適用確認
- [ ] 小規模学習
  - Overfittingテスト
  - CFGの効果確認
- [ ] 本格学習
  - 700k stepsまで学習

### 成果物
- 学習済みText-to-Latentモデル
- Flow Matching損失収束
- 多様な音声生成サンプル

---

## Phase 3: Duration Predictor実装

**期間**: 1-2週間
**目標**: テキストから発話時間をO(1)で予測

### 3.1 前処理

- [ ] `scripts/extract_durations.py`: 時間長抽出
  - TTLで生成した音声から実際の時間長を抽出
  - `data/processed/durations/`に保存

### 3.2 モデル実装

#### Sentence Encoder (models/duration_predictor/sentence_encoder.py)
- [ ] `SentenceEncoder`: 文レベルのテキスト表現
  - Character Embedder (64-dim)
  - ConvNeXt (6 layers)
  - Attention Encoder (2 layers, 2 heads)
  - Projection Out (64-dim)

#### Style Encoder (models/duration_predictor/style_encoder.py)
- [ ] `DPStyleEncoder`: スタイル表現抽出（簡略版）
  - Projection In (temporal compression 6x)
  - ConvNeXt (4 layers)
  - Style Token Layer (8 tokens, simplified)
  - Output: 16-dim style embedding

#### Predictor (models/duration_predictor/predictor.py)
- [ ] `DurationPredictor`: 時間長予測
  - MLP (2 layers, hidden=128)
  - Input: sentence_emb (64) + style_emb (16)
  - Output: duration (scalar)

#### 統合 (models/duration_predictor/duration_predictor.py)
- [ ] `DurationPredictorModel`: DP全体統合

### 3.3 損失関数実装 (losses/dp_losses.py)

- [ ] `DurationLoss`: MSE損失
  - 予測時間長と実際の時間長の差
- [ ] `L1DurationLoss`: L1損失（補助）
  - 重み: 0.5

### 3.4 学習ループ (trainers/dp_trainer.py)

- [ ] `DurationPredictorTrainer`: 学習ループ
  - Duration損失計算
  - Validation
  - Checkpointing
  - Logging

### 3.5 スクリプト (scripts/)

- [ ] `extract_durations.py`: 時間長抽出（前処理）
- [ ] `train_duration_predictor.py`: 学習メインスクリプト
- [ ] `evaluate_duration_predictor.py`: 評価スクリプト
  - Duration誤差計算
  - 相関係数分析

### 3.6 テストとデバッグ

- [ ] ユニットテスト
- [ ] 学習（3k steps、非常に短い）

### 成果物
- 学習済みDuration Predictorモデル
- Duration誤差 < 10%

---

## Phase 4: 統合とファインチューニング

**期間**: 2-3週間
**目標**: 3つのコンポーネントを統合してエンドツーエンドTTS

### 4.1 統合実装

#### Pipeline (inference/pipeline.py)
- [ ] `SupertonicTTSPipeline`: エンドツーエンド推論
  - テキスト入力
  - Duration Predictor → 時間長予測
  - Text-to-Latent → 潜在表現生成
  - Autoencoder Decoder → 音声波形
  - 速度調整、CFGスケール調整

#### Text Processor (inference/text_processor.py)
- [ ] `TextProcessor`: テキスト前処理
  - 正規化、クリーニング
  - 文分割
  - Unicode変換

#### 統合スクリプト (scripts/integrate_models.py)
- [ ] 3つのチェックポイントを読み込み
- [ ] 統合モデルとして保存

### 4.2 ファインチューニング（オプション）

- [ ] `scripts/finetune_integrated.py`: エンドツーエンド学習
  - 3つのモデルを同時に最適化
  - 学習率を低く設定
  - 数千ステップのみ

### 4.3 テストとデバッグ

- [ ] エンドツーエンドテスト
  - 様々なテキストで音声生成
  - 音質確認
  - 速度確認

### 成果物
- 統合TTSモデル
- エンドツーエンド推論パイプライン

---

## Phase 5: 評価と最適化

**期間**: 1-2週間
**目標**: 評価、最適化、ONNX変換

### 5.1 評価実装

#### 評価スクリプト (scripts/evaluate.py)
- [ ] 客観評価
  - Mel-Spectrogram Error
  - Duration Error
  - Character Error Rate (ASR)
- [ ] 主観評価準備
  - サンプル生成
  - MOSテスト用インターフェース

### 5.2 ONNX変換 (export/)

#### ONNX Exporter (export/onnx_exporter.py)
- [ ] `ONNXExporter`: PyTorch → ONNX変換
  - 各コンポーネント個別に変換
  - Dynamic axes設定
  - Opset version 17

#### 量子化（オプション）(export/quantization.py)
- [ ] Dynamic Quantization
- [ ] Static Quantization

#### スクリプト (scripts/export_onnx.py)
- [ ] ONNX変換メインスクリプト
- [ ] 変換後の動作確認

#### テストスクリプト (scripts/test_onnx.py)
- [ ] ONNX推論テスト
- [ ] PyTorchとの出力比較

### 5.3 デモ実装

#### Gradio App (demo/gradio_app.py)
- [ ] `GradioApp`: Webインターフェース
  - テキスト入力
  - 音声スタイル選択
  - パラメータ調整（速度、NFE、CFGスケール）
  - 音声生成・再生
  - ダウンロード

### 成果物
- 評価レポート
- ONNXモデル
- Gradioデモアプリ

---

## Phase 6: ドキュメントとリリース

**期間**: 1-2週間
**目標**: ドキュメント整備、公開準備

### 6.1 ドキュメント

- [ ] README更新
  - 実装結果
  - 使用方法
  - サンプル結果
- [ ] API Documentation
  - モジュールのdocstring整備
  - Sphinx documentation
- [ ] トラブルシューティングガイド更新
- [ ] 学習ガイド最終化

### 6.2 リポジトリ整備

- [ ] Licenseファイル確認
- [ ] CONTRIBUTINGガイド
- [ ] Issue templates
- [ ] CI/CD設定（GitHub Actions）

### 6.3 リリース準備

- [ ] バージョンタグ付け
- [ ] GitHub Release作成
- [ ] モデル公開（Hugging Face）
- [ ] デモサイト公開（Hugging Face Spaces）

### 成果物
- 完全なドキュメント
- GitHub公開リポジトリ
- 公開デモサイト

---

## リスク管理

### 技術的リスク

| リスク | 影響度 | 対策 |
|-------|--------|------|
| GPU不足でOOM | 高 | バッチサイズ削減、Gradient Accumulation |
| データ不足で品質低下 | 中 | データ拡張、Transfer Learning検討 |
| 学習が収束しない | 高 | 学習率調整、Warmup、Gradient Clipping |
| ONNXエクスポート失敗 | 中 | Dynamic Shape対応、Opset調整 |

### スケジュールリスク

| リスク | 影響度 | 対策 |
|-------|--------|------|
| TTL学習が長引く | 高 | Early Stoppingの検討 |
| バグ修正に時間がかかる | 中 | ユニットテスト充実、小規模検証 |

---

## 次のアクション

### 最優先タスク（Phase 1開始）

1. **モデル実装開始**
   - [ ] `models/autoencoder/encoder.py` 実装
   - [ ] `models/autoencoder/decoder.py` 実装
   - [ ] `models/autoencoder/discriminator.py` 実装

2. **損失関数実装**
   - [ ] `losses/autoencoder_losses.py` 実装

3. **学習ループ実装**
   - [ ] `trainers/autoencoder_trainer.py` 実装

4. **学習スクリプト作成**
   - [ ] `scripts/train_autoencoder.py` 実装

---

## 進捗追跡

進捗は以下で追跡:
- [ ] このドキュメントのチェックボックス
- [ ] GitHub Issues/Projects
- [ ] 週次レビュー

---

## 参考資料

- [論文解説](../../論文解説.md)
- [学習ガイド](./training_guide.md)
- [SupertonicTTS論文](https://arxiv.org/abs/2503.23108)
- [tts.json設定](../../assets/onnx/tts.json)
