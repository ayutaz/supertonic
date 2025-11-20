# SupertonicTTS Training - 実装完了報告

このドキュメントは、SupertonicTTS学習パイプラインの実装状況をまとめたものです。

## 実装完了日
2025年1月某日

## 完了した実装

### ✅ Phase 0: Speech Autoencoder (音声オートエンコーダー)

**実装内容:**
- MelSpectrogramProcessor: メルスペクトログラム変換
- AcousticEncoder: 音声 → 24次元潜在表現
- AcousticDecoder + Vocoder: 潜在表現 → 音声
- MultiScaleDiscriminator: 3スケール (full, 1/2, 1/4) でGAN識別
- AutoencoderTrainer: 完全な学習ループ + チェックポイント管理
- 損失関数:
  - MultiScaleSTFTLoss: 複数FFTサイズでのスペクトログラム損失
  - MelSpectrogramLoss: メルスペクトログラム損失
  - FeatureMatchingLoss: Discriminator中間層特徴のマッチング
  - AdversarialLoss: GAN学習での敵対的損失 (Hinge Loss)
  - DiscriminatorLoss: Discriminator学習損失

**テスト:**
- test_autoencoder.py: 6つの包括的なテスト
- 全てのコンポーネントが正常動作を確認

**使用方法:**
```bash
cd training
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_audio.txt \
  --stage 1 \
  --ae-epochs 100 \
  --batch-size 16
```

---

### ✅ Phase 1: Common Modules (共通モジュール)

**実装内容:**
- **Attention:**
  - LARoPE (Length-Aware Rotary Position Embedding): テキスト-音声アライメント最適化
  - MultiHeadAttention: マルチヘッドアテンション
- **ConvNeXt:**
  - ConvNeXtBlock: Depthwise convolution + inverted bottleneck + layer scale
  - ConvNeXtStack: 複数ConvNeXtブロックのスタック
  - InitialConvNeXt: 初期畳み込みレイヤー
- **Layers:**
  - CausalConv1d: 因果的畳み込み (自己回帰生成用)
  - LayerScale: レイヤースケーリング
  - FiLM: Feature-wise Linear Modulation (条件付け)
  - TimeEmbedding: タイムステップエンベディング
  - StyleTokenLayer: スタイルトークンレイヤー (key-value/prototype)
- **Utils:**
  - get_activation: アクティベーション関数取得
  - sequence_mask: シーケンスマスク生成
  - init_weights: 重み初期化

**テスト:**
- test_common.py: 14の包括的なテスト
- 全ての共通モジュールが正常動作を確認

---

### ✅ Phase 2: TTL (Text-to-Latent, テキスト→潜在表現)

**実装内容:**
- TextEncoder: テキスト → テキストエンベディング (LARoPE使用)
- StyleEncoder: 参照音声 → スタイルエンベディング
- CrossAttentionLARoPE: LARoPE対応クロスアテンション
- VectorField: Flow Matchingによる潜在表現生成
- TTLModel: 完全なTTLモデル統合
- TTLTrainer: 完全な学習ループ + 冷凍Autoencoder
- 損失関数:
  - FlowMatchingLoss: Conditional Flow Matching損失
  - TTLLoss: TTL総合損失

**テスト:**
- test_ttl.py: 7つの包括的なテスト
- 全てのTTLコンポーネントが正常動作を確認

**使用方法:**
```bash
cd training
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_tts.txt \
  --stage 2 \
  --ttl-epochs 100 \
  --batch-size 16
```

**重要な機能:**
- **Frozen Autoencoder**: Stage 1で学習したAutoencoderを固定して使用
- **Flow Matching**: ノイズからクリーンな潜在表現を反復的に生成
- **LARoPE**: テキスト-音声の異なる長さを正規化し、アライメント精度を向上

---

### ✅ Phase 3: Duration Predictor (発話長予測)

**実装内容:**
- SentenceEncoder: テキストエンコーダー (文レベル)
- StyleEncoderDP: 参照音声からスタイル抽出
- DurationPredictor: 発話全体の長さを予測 (O(1)計算量)
- DPTrainer: 完全な学習ループ + 冷凍Autoencoder
- 損失関数:
  - DPLoss: MSE損失 (予測長 vs 真値長)

**革新的な特徴:**
- **発話レベル予測**: 従来のO(N)音素単位予測ではなく、O(1)の発話全体予測
- **速度調整対応**: 学習後に速度パラメータで柔軟に調整可能

**テスト:**
- test_dp.py: 5つの包括的なテスト
- 速度調整機能の動作確認含む

**使用方法:**
```bash
cd training
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/data \
  --metadata metadata_tts.txt \
  --stage 3 \
  --dp-epochs 100 \
  --batch-size 16
```

---

### ✅ Dataset & Data Processing (データセットとデータ処理)

**実装内容:**
- **TTSDataset**: テキスト-音声ペアデータセット
  - JSON/TXT形式メタデータ対応
  - Unicode処理統合
  - メルスペクトログラム抽出統合
  - 長さフィルタリング対応
- **AudioDataset**: 音声のみデータセット (Stage 1 Autoencoder用)
  - セグメント分割対応
  - 正規化、無音削除対応
- **UnicodeProcessor**: Unicode処理
  - NFKD正規化 (Compatibility Decomposition)
  - Unicode値 → Text IDs変換
  - unicode_indexer.json対応
  - バッチ処理 + マスク生成
- **collate_fn**: バッチ処理
  - 可変長シーケンスのパディング
  - マスク生成
  - DataLoader統合

**テスト:**
- test_dataset.py: 6つの包括的なテスト
  - TXT/JSON両形式のメタデータ読み込み
  - Unicode処理
  - バッチ処理
  - DataLoader統合
  - AudioDataset

**使用方法:**
```python
# TTSDataset (Stage 2/3用)
from data.datasets import TTSDataset, collate_fn
from data.unicode import UnicodeProcessor
from data.preprocessing import MelSpectrogramExtractor

unicode_processor = UnicodeProcessor(unicode_indexer_path="unicode_indexer.json")
mel_extractor = MelSpectrogramExtractor(sample_rate=44100, ...)

dataset = TTSDataset(
    data_dir="/path/to/data",
    metadata_file="metadata.txt",  # or .json
    unicode_processor=unicode_processor,
    mel_extractor=mel_extractor,
)

# AudioDataset (Stage 1用)
from data.datasets import AudioDataset

dataset = AudioDataset(
    data_dir="/path/to/data",
    audio_files=["audio1.wav", "audio2.wav", ...],
    mel_extractor=mel_extractor,
    segment_length=44100,  # 1秒セグメント
)
```

---

### ✅ 統合学習パイプライン (Integrated Training Pipeline)

**実装内容:**
- **train_pipeline.py**: 3段階学習の自動化スクリプト
  - Stage 1: Autoencoder学習
  - Stage 2: TTL学習 (冷凍Autoencoder使用)
  - Stage 3: DP学習 (冷凍Autoencoder使用)
  - チェックポイント自動管理
  - TensorBoard統合
  - 段階スキップ機能
  - 個別段階実行機能

**主な機能:**
1. **完全な自動化**: 3段階を順次実行
2. **チェックポイント管理**: 自動保存・読み込み
3. **段階制御**: 特定段階のみ実行、スキップ可能
4. **自動検出**: Stage 1チェックポイント自動検出
5. **検証**: 各エポック後に検証実行

**使用方法:**

**完全パイプライン実行:**
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

**特定段階のみ実行:**
```bash
# Stage 1のみ
uv run python scripts/train_pipeline.py ... --stage 1 --ae-epochs 100

# Stage 2のみ
uv run python scripts/train_pipeline.py ... --stage 2 --ttl-epochs 100

# Stage 3のみ
uv run python scripts/train_pipeline.py ... --stage 3 --dp-epochs 100
```

**段階をスキップ:**
```bash
# Stage 1をスキップして、Stage 2と3のみ実行
uv run python scripts/train_pipeline.py \
  ... \
  --skip-stages autoencoder \
  --ttl-epochs 100 \
  --dp-epochs 100
```

---

### ✅ テストスイート (Test Suite)

**実装内容:**
- **test_autoencoder.py**: Phase 0 (6テスト)
- **test_common.py**: Phase 1 (14テスト)
- **test_ttl.py**: Phase 2 (7テスト)
- **test_dp.py**: Phase 3 (5テスト)
- **test_dataset.py**: Dataset (6テスト)
- **run_all_tests.py**: 統合テストランナー

**統合テスト結果:**
```
Total: 5 phases | Passed: 5 | Failed: 0
Phase 0: 6/6 tests passed
Phase 1: 14/14 tests passed
Phase 2: 7/7 tests passed
Phase 3: 5/5 tests passed
Dataset: 6/6 tests passed
```

**使用方法:**
```bash
cd training
uv run python tests/run_all_tests.py
```

---

## ドキュメント

完備されたドキュメント:
- ✅ **TRAINING_PIPELINE.md**: 完全な学習パイプラインガイド
  - 使用方法
  - コマンドラインオプション
  - GPU別推奨設定
  - データセット別推奨エポック数
  - トラブルシューティング
  - 出力ディレクトリ構造
  - 実装状況
- ✅ **PROGRESS_SUMMARY.md** (このファイル): 実装完了報告

---

## 出力ディレクトリ構造

```
outputs/
├── checkpoints/
│   ├── autoencoder/
│   │   ├── autoencoder_step5000.pt
│   │   ├── autoencoder_step10000.pt
│   │   └── autoencoder_final.pt
│   ├── ttl/
│   │   ├── ttl_step1000.pt
│   │   └── ttl_final.pt
│   └── dp/
│       ├── dp_step1000.pt
│       └── dp_final.pt
└── logs/
    ├── autoencoder/
    │   └── events.out.tfevents...
    ├── ttl/
    │   └── events.out.tfevents...
    └── dp/
        └── events.out.tfevents...
```

**TensorBoardで監視:**
```bash
tensorboard --logdir outputs/logs
```

---

## 今後の拡張機能 (オプション)

以下の機能は完全に動作する学習パイプラインには含まれていませんが、将来的な改善として検討できます:

1. **Wandb統合** (TensorBoardは既に実装済み)
2. **分散学習サポート** (マルチGPU対応)
3. **自動ハイパーパラメータチューニング**
4. **音声品質評価スクリプト** (MOS, RTF測定)
5. **ONNXエクスポート機能** (PyTorch → ONNX変換)
6. **自動データ拡張**
7. **Resume機能** (チェックポイントから再開)

---

## GPU別推奨設定

| GPU | Batch Size | Segment Length | Epochs (AE) | 学習時間 (10h data) |
|-----|-----------|----------------|-------------|----------------------|
| RTX 3060 (12GB) | 8-16 | 2.0s | 100 | 3-7日 |
| RTX 4070 (12GB) | 16-24 | 2.0s | 100 | 2-5日 |
| RTX 4090 (24GB) | 32-48 | 2.0-3.0s | 100 | 1-3日 |
| A100 (40GB) | 64-96 | 3.0-4.0s | 100 | 0.5-1.5日 |

---

## データセット別推奨エポック数

| データサイズ | AE Epochs | TTL Epochs | DP Epochs |
|-----------|-----------|-----------|-----------|
| 10時間 | 200-300 | 150-200 | 100-150 |
| 50時間 | 100-150 | 80-120 | 50-80 |
| 100時間 | 80-100 | 50-80 | 30-50 |

---

## トラブルシューティング

### OOM (Out of Memory) エラー

1. `--batch-size` を減らす (16 → 8 → 4)
2. Segment lengthを減らす (デフォルト2.0秒)
3. Mixed Precisionを有効化 (デフォルトで有効)

### 学習が進まない

1. 学習率を調整 (`--learning-rate 1e-4` など)
2. Gradient clippingを確認 (デフォルト1.0)
3. データの前処理を確認 (正規化、無音削除)

### チェックポイントが見つからない

Stage 2/3を実行する前に、必ずStage 1を実行してください:
```bash
# Stage 1を先に実行
uv run python scripts/train_pipeline.py ... --stage 1 --ae-epochs 100

# その後Stage 2/3を実行
uv run python scripts/train_pipeline.py ... --stage 2 --ttl-epochs 100
```

---

## まとめ

**完了した実装:**
- ✅ Phase 0: Speech Autoencoder (6テスト)
- ✅ Phase 1: Common Modules (14テスト)
- ✅ Phase 2: TTL (7テスト)
- ✅ Phase 3: Duration Predictor (5テスト)
- ✅ Dataset & Data Processing (6テスト)
- ✅ 統合学習パイプライン
- ✅ 完全なドキュメント

**テスト結果:**
- **38テスト全て合格** (100%成功率)

**すぐに学習を開始できます！**

```bash
cd training
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir /path/to/your/data \
  --metadata your_metadata.txt \
  --output-dir outputs \
  --ae-epochs 100 \
  --ttl-epochs 100 \
  --dp-epochs 100 \
  --batch-size 16 \
  --learning-rate 2e-4
```

学習の進捗はTensorBoardで監視できます:
```bash
tensorboard --logdir outputs/logs
```

---

**作成日**: 2025年1月某日
**最終更新**: 2025年1月某日
