# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**Supertonic**は、デバイス上で動作する超高速なテキスト読み上げ(TTS)システムです。ONNX Runtimeを使用して完全にローカルで実行され、クラウドやAPIコールは不要で、プライバシーを完全に保護します。

### 主な特徴

- **超高速**: M4 Proで最大167倍のリアルタイム速度で音声生成
- **軽量**: わずか66Mパラメータで効率的なデバイス上パフォーマンスを実現
- **多言語サポート**: Python, Node.js, Browser, Java, C++, C#, Go, Swift, iOS, Rustの実装例を提供
- **自然なテキスト処理**: 数字、日付、通貨、略語などを前処理なしで処理

### アーキテクチャ

Supertonicは以下の4つの主要なONNXモデルで構成されています:

1. **Duration Predictor (DP)**: テキストと音声スタイルから音声の長さを予測
2. **Text Encoder**: テキストをエンベディングに変換
3. **Vector Estimator (TTL)**: ノイズの多い潜在表現からクリーンな潜在表現を推定(Flow Matchingベース)
4. **Vocoder**: 潜在表現を音声波形に変換

### ワークフロー

```
テキスト入力 → Unicode処理 → Duration Predictor → Text Encoder
                                                    ↓
音声スタイル ─────────────────────────────┬─────→ Vector Estimator
                                         ↓
                                  ノイズサンプリング
                                         ↓
                              反復的ノイズ除去(N steps)
                                         ↓
                                     Vocoder
                                         ↓
                                    WAVファイル出力
```

## 開発コマンド

### テスト実行

全ての言語実装をテストするには:
```bash
bash test_all.sh
```

テストモードの選択:
- `1`: デフォルト推論のみ
- `2`: バッチ推論のみ
- `3`: 長文推論のみ
- `4`: 全てのテスト

### Python実装

```bash
cd py
uv sync                    # 依存関係のインストール
uv run example_onnx.py     # デフォルト推論
uv run example_onnx.py --total-step 10 --speed 1.2  # 高品質・高速生成
```

主なパラメータ:
- `--total-step`: ノイズ除去ステップ数(デフォルト: 5、高いほど高品質だが遅い)
- `--speed`: 音声速度係数(デフォルト: 1.05、推奨範囲: 0.9-1.5)
- `--batch`: バッチ処理モード
- `--voice-style`: 音声スタイルファイルパス
- `--text`: 合成するテキスト

### C#実装

```bash
cd csharp
dotnet restore             # 依存関係のインストール
dotnet run                 # デフォルト推論
dotnet run -- --total-step 10 --speed 0.9  # パラメータ指定
dotnet build -c Release    # リリースビルド
```

依存関係:
- .NET 9.0 SDK
- Microsoft.ML.OnnxRuntime (1.20.1)
- System.Text.Json (9.0.1)

## モデルとアセットのセットアップ

実行前に、ONNXモデルとプリセット音声をダウンロードする必要があります:

```bash
# Git LFSのインストール(macOS)
brew install git-lfs && git lfs install

# モデルとアセットのダウンロード
git clone https://huggingface.co/Supertone/supertonic assets
```

必要なファイル:
- `assets/onnx/`: ONNXモデルファイル(dp.onnx, text_enc.onnx, vector_est.onnx, vocoder.onnx, unicode_indexer.json, config.json)
- `assets/voice_styles/`: 音声スタイルファイル(M1.json, F1.json, M2.json, F2.json)

## コード構造の理解

### 共通パターン(Python/C#)

全ての実装は以下の共通パターンに従っています:

1. **設定の読み込み**: `config.json`からモデル設定を読み込み
2. **Unicode処理**: `unicode_indexer.json`を使用してテキストをトークンに変換
3. **音声スタイルの読み込み**: JSON形式の音声スタイルベクトル(`style_ttl`, `style_dp`)を読み込み
4. **推論パイプライン**:
   - Duration Predictor: テキストマスクと音声スタイルから音声長を予測
   - Text Encoder: テキストエンベディングを生成
   - Vector Estimator: Flow Matchingを使用して`total_step`回の反復でノイズを除去
   - Vocoder: 最終的な音声波形を生成
5. **後処理**: 音声をトリミングして16-bit WAVとして保存

### 長文処理

長いテキスト(デフォルト300文字以上)は自動的にチャンクに分割されます:
- 段落境界(`\n\n`)で分割
- 文境界(`.`, `!`, `?`)で分割
- カンマ(`,`)で分割
- 空白で分割
- チャンク間に0.3秒の無音を挿入
- 全チャンクを単一のWAVファイルに結合

**注意**: `--batch`モードでは自動チャンキングは無効化されます。

## Unity統合ガイド

### C# (.NET)実装をUnityで使用する方法

Unityでこのプロジェクトを使用する場合、以下の2つのアプローチがあります:

#### 1. ONNX Runtime Unity Pluginを使用(推奨)

**最新情報(2025年)**:
- Unity 6000.0.43f1 (LTS)とONNX Runtime 1.22.1をサポート
- 公式プラグイン: `asus4/onnxruntime-unity`

**インストール手順**:

1. `Packages/manifest.json`にスコープ付きレジストリを追加:
```json
{
  "scopedRegistries": [
    {
      "name": "npm",
      "url": "https://registry.npmjs.com/",
      "scopes": ["com.github.asus4"]
    }
  ],
  "dependencies": {
    "com.github.asus4.onnxruntime": "0.4.2"
  }
}
```

2. `Helper.cs`と`ExampleONNX.cs`のコードを改変:
   - `Microsoft.ML.OnnxRuntime`の代わりに`Unity.Sentis.ONNX`または`asus4.onnxruntime`を使用
   - ファイルI/O操作をUnityの`Resources`または`StreamingAssets`に対応させる

**プラットフォームサポート**:
- CPU: 全プラットフォーム(Windows, macOS, Linux, iOS, Android)
- CoreML: macOS/iOS
- NNAPI: Android
- DirectML: Windows
- CUDA/TensorRT: 対応システム

#### 2. IronPythonでPython実装を使用(非推奨)

**制限事項**:
- IronPythonは.NET 4.5を必要とします
- `numpy`, `onnxruntime`などのネイティブC拡張に依存するライブラリは動作しません
- このため、**Python実装はUnityでは直接使用できません**

**代替案**:
- C#実装を使用し、ONNX Runtime Unity Pluginで実行
- または、外部Pythonプロセスを起動してプロセス間通信(IPC)を使用

### Unity統合の推奨アプローチ

1. **C#コードベースを使用**: `csharp/Helper.cs`と`csharp/ExampleONNX.cs`をベースにする
2. **ONNX Runtime Unity Pluginをインストール**: NPMパッケージとして追加
3. **アセットを配置**: `assets/`フォルダを`StreamingAssets/`に配置
4. **コードを改変**:
   - `File.ReadAllText()`を`UnityEngine.Resources.Load()`または`StreamingAssets`パスに変更
   - `soundfile`での保存を`AudioClip`生成に変更
5. **実行プロバイダーを選択**: プラットフォームに応じてCPU/CoreML/NNAPI/DirectMLを選択

## 技術的な詳細

### パフォーマンス特性

- **RTF (Real-time Factor)**: M4 Pro CPUで0.012(266文字)、RTX4090で0.001
- **文字/秒**: M4 Pro CPUで1263文字/秒、RTX4090で12164文字/秒
- **推論ステップ**: 2ステップ(最速)〜10ステップ(最高品質)の範囲で調整可能

### メモリ要件

- モデルサイズ: 合計約66M パラメータ
- ランタイムメモリ: テキスト長とバッチサイズに依存
- 長文処理: 自動チャンキングによりメモリ使用量を制御

## 日本語対応について

### 現状

**Supertonicは現時点では英語専用のTTSシステムです。**

- 公式ドキュメントやサンプルコードには日本語のテキスト例が一切含まれていません
- Unicode文字レベル処理を採用しているため、理論的には日本語を含む多言語に対応可能な設計ですが、実際の動作は未検証です
- モデルの学習データに日本語が含まれているかは不明で、日本語テキストで適切な音声が生成される保証はありません

### 日本語対応の実現可能性

**技術的には可能ですが、重大な課題があります。**

#### 総合評価

| 評価項目 | スコア | 備考 |
|---------|-------|------|
| 技術的実現可能性 | ⭐⭐⭐☆☆ (3/5) | 学習コード未公開が最大の障壁 |
| リソース要件 | ⭐⭐⭐⭐☆ (4/5) | GPUは比較的安価、データは無料 |
| 開発期間 | ⭐⭐⭐☆☆ (3/5) | 4～7ヶ月は長期プロジェクト |
| 成功確率 | ⭐⭐⭐☆☆ (3/5) | 学習コード再実装のリスクあり |

**総合難易度: 中～高**

#### 主要な技術的課題

1. **学習コードが公開されていない** 🔴 難易度：非常に高
   - リポジトリには推論用のONNXモデルのみが含まれています
   - PyTorchの学習コードは現時点で公開されていません
   - [GitHub Issue #1](https://github.com/supertone-inc/supertonic/issues/1)によると、トレーニングコードの公開予定は不明です
   - ONNXモデルからPyTorchモデルへの変換は推論には可能ですが、学習には不十分です

2. **学習コードの完全な再実装が必要** 🔴 難易度：非常に高
   - 論文（[arXiv:2503.23108](https://arxiv.org/abs/2503.23108)）を基に一から実装する必要があります
   - Flow Matching、Duration Predictor、Reference Encoderなど複雑なコンポーネント
   - 推定工数：2～4ヶ月（経験あるエンジニア1名）

3. **日本語テキスト正規化** 🟡 難易度：中
   - 現在のUnicode処理は英語中心です
   - 日本語特有の処理（漢字、カタカナ、ひらがな）が必要
   - 長文チャンキング機能は英語の文区切りのみに対応（日本語の「。」「！」「？」には未対応）

4. **各モジュールの学習要件**
   - **Vector Estimator (Flow Matching)**: 再学習必須 🔴
   - **Text Encoder**: 再学習必須 🔴
   - **Duration Predictor**: 再学習必須 🔴
   - **Reference Encoder**: 部分的な再学習が必要 🟡
   - **Vocoder**: 再学習不要の可能性あり 🟢
   - **Speech Autoencoder**: 再学習不要の可能性あり 🟢

### 必要なリソース

#### データセット（無料で入手可能）

| データセット | 時間 | 話者数 | 品質 | URL |
|------------|-----|-------|------|-----|
| JSUT Corpus | 10時間 | 1 | 高 | https://sites.google.com/site/shinnosuketakamichi/publication/jsut |
| JVS Corpus | 30時間 | 100 | 高（24kHz） | https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus |
| **合計** | **40時間** | **101** | **高** | Fine-tuningには十分 |

#### GPU要件

| 学習モード | 必要VRAM | 推奨GPU | 学習時間（10時間データ） |
|---------|---------|---------|---------------------|
| Mixed precision (FP16) | 1.5～2GB | RTX 3050 (4GB) | 5～10日 |
| 最適構成 | 4～6GB | RTX 3060 (12GB) | 2～7日 |
| 高速学習 | 12～24GB | RTX 4090 (24GB) | 1～3日 |

**推奨構成**: RTX 3060 (12GB) または RTX 4060 Ti (16GB)

#### 開発期間（1名のエンジニア）

| フェーズ | 期間 | 内容 |
|---------|-----|------|
| 学習コード実装 | 2～4ヶ月 | Flow Matching、各モジュールの実装 |
| データ準備 | 2週間～1ヶ月 | JSUT/JVSのダウンロードと前処理 |
| 学習と調整 | 1～2ヶ月 | 学習、ハイパーパラメータチューニング |
| **合計** | **4～7ヶ月** | - |

### 推奨アプローチ

#### 最も現実的な戦略

**短期的（すぐに日本語TTSが必要な場合）**

既存の日本語対応TTSを使用することを強く推奨します：

1. **StableTTS** ⭐ 最も有望
   - Flow Matching + DiT（Diffusion Transformer）アーキテクチャ
   - **日本語、中国語、英語を単一チェックポイントでサポート**
   - 31Mパラメータ（SupertonicTTSと同規模）
   - **学習コード公開済み**
   - GitHub: [KdaiP/StableTTS](https://github.com/KdaiP/StableTTS)

2. **Kokoro TTS**
   - 82Mパラメータ
   - **日本語、英語、フランス語、韓国語、中国語対応**
   - StyleTTS 2 + ISTFTNetアーキテクチャ
   - 学習コスト明確（$1000、A100 80GB × 1000時間）
   - GitHub: [hexgrad/kokoro](https://github.com/hexgrad/kokoro)

3. **その他の選択肢**
   - **XTTS**: 467Mパラメータ、16言語対応、fine-tuningコード公開
   - **StyleTTS2**: Diffusionベース、fine-tuningガイドあり

**中期的（3～6ヶ月の開発期間が確保できる場合）**

StableTTSをベースに、SupertonicTTSの技術を段階的に統合する混合アプローチ：

1. StableTTSで日本語TTSの基盤を構築
2. SupertonicTTSの技術を部分的に取り入れる：
   - LARoPE（Length-Aware Rotary Position Embedding）
   - Self-Purifying Flow Matching
   - 発話レベルDuration Predictor
3. 段階的に最適化とカスタマイズ

**長期的（7ヶ月以上の開発期間とリソースがある場合）**

論文ベースでSupertonicTTSの学習コードを完全に再実装：

1. 論文（arXiv:2503.23108, 2509.11084, 2509.19091）を詳細に分析
2. 各モジュールを一から実装
3. JSUT + JVSで学習
4. PyTorchからONNXへエクスポート

**代替案：コミュニティとの協力**

- Supertone Inc.に[GitHub Issue](https://github.com/supertone-inc/supertonic/issues)でトレーニングコード公開を依頼
- 学術的な協力の可能性を探る
- コミュニティで共同開発を提案

### 実装ステップ（StableTTSベースの場合）

1. **StableTTSのセットアップ** (1週間)
   ```bash
   git clone https://github.com/KdaiP/StableTTS
   cd StableTTS
   # 環境構築とデモ実行
   ```

2. **データセット準備** (2週間)
   - JSUT CorpusとJVS Corpusのダウンロード
   - StableTTS形式への変換
   - 訓練/検証/テストセット分割

3. **モデル学習** (1～2ヶ月)
   - config.pyでパラメータ設定
   - train.pyで学習開始
   - チェックポイントの評価

4. **SupertonicTTS技術の統合**（オプション、1～2ヶ月）
   - LARoPEの実装と統合
   - Self-Purifying Flow Matchingの適用
   - 発話レベルDuration Predictorへの変更

5. **評価と最適化** (2週間～1ヶ月)
   - 音声品質評価（MOS、RTF測定）
   - ハイパーパラメータチューニング
   - ONNXエクスポート（オプション）

### 参考リソース

#### 論文

- **SupertonicTTS**: [arXiv:2503.23108](https://arxiv.org/abs/2503.23108) - メインアーキテクチャ
- **LARoPE**: [arXiv:2509.11084](https://arxiv.org/abs/2509.11084) - テキスト-音声アライメント
- **Self-Purifying Flow Matching**: [arXiv:2509.19091](https://arxiv.org/abs/2509.19091) - ノイズラベルでの学習

#### 参考プロジェクト

| プロジェクト | 特徴 | GitHub |
|------------|------|--------|
| StableTTS | Flow Matching、日本語対応、学習コード公開 | [KdaiP/StableTTS](https://github.com/KdaiP/StableTTS) |
| Kokoro TTS | 日本語対応済み、82Mパラメータ | [hexgrad/kokoro](https://github.com/hexgrad/kokoro) |
| Matcha-TTS | Conditional Flow Matching、学習コード公開 | [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) |
| F5-TTS | 100K時間多言語データ、ゼロショット | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) |
| VoiceFlow-TTS | Rectified Flow Matching | [X-LANCE/VoiceFlow-TTS](https://github.com/X-LANCE/VoiceFlow-TTS) |

#### データセット

- **JSUT Corpus**: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- **JVS Corpus**: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

#### ツール

- **onnx2torch**: ONNXからPyTorchへの変換（推論のみ）
- **Monotonic Alignment Search**: [supertone-inc/super-monotonic-align](https://github.com/supertone-inc/super-monotonic-align)

### 結論

日本語対応は技術的には実現可能ですが、学習コードが公開されていないことが最大の障壁です。

**推奨事項**:
- **すぐに必要**: StableTTSやKokoro TTSなど既存の日本語対応TTSを使用
- **カスタマイズが必要**: StableTTSをベースにSupertonicTTS技術を段階的に統合
- **長期的な研究**: 論文ベースで学習コードを完全再実装、またはSupertone Inc.にトレーニングコード公開を依頼

現時点では、**StableTTSベースのアプローチ**が成功確率と実装コストのバランスが最も良いと考えられます。

---

## 学習コード実装ガイド（ゼロから実装）

本セクションでは、Supertonicモデルを**ゼロから実装**するための完全なガイドを提供します。リポジトリ内の推論コード分析、論文の詳細解説、具体的な実装方法を含みます。

### 目次

1. [リポジトリコードから抽出した実装情報](#リポジトリコードから抽出した実装情報)
2. [SupertonicTTS論文の完全解説](#supertonictts論文の完全解説)
3. [学習方法の詳細ガイド](#学習方法の詳細ガイド)
4. [特殊技術の実装](#特殊技術の実装)
5. [実装チェックリストとロードマップ](#実装チェックリストとロードマップ)

---

### リポジトリコードから抽出した実装情報

#### 推論パイプライン全体のフロー

```
1. テキスト前処理
   text → NFKD正規化 → Unicode値 → text_ids (via unicode_indexer)

2. Duration Predictor
   text_ids + style_dp + text_mask → duration (秒単位)
   duration = duration / speed  # 速度調整

3. Text Encoder
   text_ids + style_ttl + text_mask → text_emb

4. ノイズサンプリング
   duration → wav_length → latent_length
   noisy_latent ~ N(0, I)  # 標準正規分布

5. Flow Matching (反復的ノイズ除去)
   for step in 0..total_step-1:
       xt = VectorEstimator(xt, text_emb, style_ttl, text_mask, latent_mask, step, total_step)

6. Vocoder
   clean_latent → wav_tts
```

#### モジュール別の入出力仕様

**Duration Predictor** (`py/helper.py:96-99`)
```python
# 入力
text_ids: [batch, max_text_len], dtype=int64
style_dp: [batch, dp_dim1, dp_dim2], dtype=float32
text_mask: [batch, 1, max_text_len], dtype=float32

# 出力
duration: [batch], dtype=float32  # 秒単位
```

**Text Encoder** (`py/helper.py:100-103`)
```python
# 入力
text_ids: [batch, max_text_len], dtype=int64
style_ttl: [batch, ttl_dim1, ttl_dim2], dtype=float32
text_mask: [batch, 1, max_text_len], dtype=float32

# 出力
text_emb: テキストエンベディング（次元はモデル依存）
```

**Vector Estimator** (`py/helper.py:106-119`)
```python
# 入力
noisy_latent: [batch, latent_dim, latent_len], dtype=float32
text_emb: Text Encoderの出力
style_ttl: [batch, ttl_dim1, ttl_dim2], dtype=float32
text_mask: [batch, 1, max_text_len], dtype=float32
latent_mask: [batch, 1, latent_len], dtype=float32
current_step: [batch], dtype=float32  # 0 to total_step-1
total_step: [batch], dtype=float32

# 出力
denoised_latent: [batch, latent_dim, latent_len], dtype=float32
```

**Vocoder** (`py/helper.py:120-121`)
```python
# 入力
latent: [batch, latent_dim, latent_len], dtype=float32

# 出力
wav_tts: [batch, wav_length], dtype=float32
```

#### テキスト処理の詳細

**Unicode正規化** (`py/helper.py:17-20`)
```python
def _preprocess_text(self, text: str) -> str:
    text = normalize("NFKD", text)  # Compatibility Decomposition
    return text
```

**テキストマスクの生成** (`py/helper.py:22-24`)
```python
def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    """
    Returns: mask [B, 1, max_len]
    有効部分=1.0、パディング=0.0
    """
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)
```

#### 潜在空間の次元計算

**重要な数式** (`py/helper.py:72-86`)
```python
# 設定ファイルから取得
base_chunk_size = cfgs["ae"]["base_chunk_size"]
chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
latent_dim = cfgs["ttl"]["latent_dim"]
sample_rate = cfgs["ae"]["sample_rate"]

# 計算式
chunk_size = base_chunk_size * chunk_compress_factor
wav_len = duration * sample_rate
latent_len = ceil(wav_len / chunk_size)
latent_dim_total = latent_dim * chunk_compress_factor
```

**例**（典型的な値）:
```
duration = 3.0秒
sample_rate = 24000Hz
base_chunk_size = 240
chunk_compress_factor = 4
latent_dim = 64

→ wav_len = 3.0 * 24000 = 72000サンプル
→ chunk_size = 240 * 4 = 960
→ latent_len = ceil(72000 / 960) = 75
→ latent_dim_total = 64 * 4 = 256
```

#### スタイルベクトルの構造

**JSONフォーマット** (`py/helper.py:238-266`)
```json
{
    "style_ttl": {
        "dims": [1, ttl_dim1, ttl_dim2],
        "data": [[[...]]]
    },
    "style_dp": {
        "dims": [1, dp_dim1, dp_dim2],
        "data": [[[...]]]
    }
}
```

2つのスタイルベクトルが存在：
- `style_ttl`: Text EncoderとVector Estimatorで使用
- `style_dp`: Duration Predictorで使用

---

### SupertonicTTS論文の完全解説

#### 論文情報

1. **SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System**
   - arXiv: 2503.23108
   - URL: https://arxiv.org/abs/2503.23108

2. **Length-Aware Rotary Position Embedding for Text-Speech Alignment**
   - arXiv: 2509.11084
   - URL: https://arxiv.org/abs/2509.11084

3. **Training Flow Matching Models with Reliable Labels via Self-Purification**
   - arXiv: 2509.19091
   - URL: https://arxiv.org/abs/2509.19091

#### 全体アーキテクチャ

SupertonicTTSは4つのコンポーネントで構成：

1. **Speech Autoencoder**: 音声 ↔ 潜在表現の変換
2. **Duration Predictor**: テキストから発話全体の長さを予測
3. **Text Encoder**: テキストをエンベディングに変換（LARoPE使用）
4. **Vector Estimator**: Flow Matchingで潜在表現を生成

#### Speech Autoencoder

**エンコーダー（学習時のみ）**:
```
音声 [wav_len]
  → ConvNeXtベースのダウンサンプリング
  → 潜在表現 [latent_dim, latent_len]
```

**デコーダー（Vocoder）**:
```
潜在表現 [latent_dim, latent_len]
  → ConvNeXtベースのアップサンプリング
  → 音声 [wav_len]
```

**損失関数**:
```
L_ae = L_reconstruction + λ_stft * L_multi_scale_stft + λ_adv * L_gan

L_reconstruction = ||wav_gt - wav_pred||_1
L_multi_scale_stft = Σ_i ||STFT_i(wav_gt) - STFT_i(wav_pred)||_1
L_gan = GAN損失（Discriminator使用）
```

#### Duration Predictor

**発話レベル予測**の革新：
- 従来: 各音素の長さを個別に予測 → O(N)回の予測
- Supertonic: 発話全体で1回の予測 → O(1)

**アーキテクチャ**:
```
text_ids + style_dp + text_mask
  → Transformer Encoder
  → Global Average Pooling
  → MLP
  → duration (単一スカラー値)
```

**損失関数**:
```
L_dp = MSE(predicted_duration, ground_truth_duration)
ground_truth_duration = wav_length / sample_rate
```

#### Text Encoder with LARoPE

**LARoPE (Length-Aware Rotary Position Embedding)**:

標準RoPEとの違い：
```
# 標準RoPE
pos = [0, 1, 2, ..., seq_len-1]  # 絶対位置

# LARoPE
pos_norm = pos / seq_len  # 正規化位置 [0, 1]
pos_scaled = pos_norm * γ  # γ = 128-256
```

数式：
```
RoPE(x, pos) = rotate(x, θ * pos)
LARoPE(x, pos, seq_len) = rotate(x, θ * (pos/seq_len) * γ)

where θ = base^(-2i/d), base=10000, d=embed_dim
```

**効果**: テキストと音声の異なる長さを正規化し、Cross-Attentionでのアライメント精度を向上

#### Vector Estimator (Flow Matching)

**Flow Matchingの原理**:

目標: ノイズ分布 p(x_0) から データ分布 p(x_1) への変換を学習

```
# Linear interpolation path
x_t = t * x_1 + (1 - t) * x_0,  t ∈ [0, 1]
x_0 ~ N(0, I)  # Gaussian noise
x_1 = Encoder(wav_gt)  # Ground truth latent

# 速度場
v_t(x|x_1) = dx_t/dt = x_1 - x_0

# 学習目標
L_cfm = E_{t,x_0,x_1,c} [||u_θ(x_t, t, c) - (x_1 - x_0)||^2]
```

**推論（Euler法）**:
```python
x = x_0  # 初期ノイズ
dt = 1.0 / total_step

for step in range(total_step):
    t = step / total_step
    v = u_θ(x, t, text_emb, style)
    x = x + v * dt  # Euler更新
```

**ネットワーク構造**: U-Net with Transformer blocks (DiT style)
- Down Blocks: Self-Attention + Cross-Attention (text) + ConvNeXt
- Middle Block: 同様
- Up Blocks: 同様 + Skip Connections

---

### 学習方法の詳細ガイド

#### 段階的学習プロセス

**Stage 1: Speech Autoencoder** (2-4週間)
```
データ: 大規模音声データ（数百～数千時間、ラベル不要）
損失: L_reconstruction + L_multi_scale_stft + L_gan
目標: 高品質な音声再構築
```

**Stage 2: Duration Predictor** (1-2週間)
```
データ: テキスト-音声ペア（10-100時間）
損失: MSE(predicted_duration, ground_truth_duration)
目標: 正確な発話長予測
```

**Stage 3: Text Encoder + Vector Estimator** (3-5週間)
```
データ: 同上
損失: Flow Matching Loss
目標: 高品質な音声潜在表現の生成
```

#### Flow Matching Lossの実装

```python
def flow_matching_loss(model, text_ids, wav_gt, style):
    # 1. Encode audio to latent (frozen autoencoder)
    with torch.no_grad():
        x_1 = frozen_encoder(wav_gt)

    # 2. Sample time uniformly
    batch_size = x_1.shape[0]
    t = torch.rand(batch_size, device=x_1.device)

    # 3. Sample Gaussian noise
    x_0 = torch.randn_like(x_1)

    # 4. Linear interpolation
    t_expanded = t.view(-1, 1, 1)
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0

    # 5. Encode text
    text_emb = text_encoder(text_ids, style.ttl, text_mask)

    # 6. Predict velocity
    v_pred = vector_estimator(
        x_t, t, text_emb, style.ttl,
        text_mask, latent_mask
    )

    # 7. Target velocity
    v_target = x_1 - x_0

    # 8. MSE loss
    loss = F.mse_loss(v_pred, v_target)

    return loss
```

#### ハイパーパラメータ（推定値）

```yaml
# Speech Autoencoder
base_chunk_size: 256
chunk_compress_factor: 4
latent_dim: 128
sample_rate: 16000

# Training
batch_size: 16-32
learning_rate: 1e-4 ~ 2e-4
optimizer: AdamW(betas=(0.9, 0.999), weight_decay=0.01)
scheduler: CosineAnnealingLR
mixed_precision: fp16
gradient_clip: 1.0

# Flow Matching
total_step_train: N/A (time sampled uniformly)
total_step_inference: 2-10 (デフォルト5)
```

---

### 特殊技術の実装

#### LARoPE完全実装

```python
import torch
import torch.nn as nn
import math

class LARoPE(nn.Module):
    def __init__(self, dim, gamma=256.0, base=10000.0):
        super().__init__()
        self.dim = dim
        self.gamma = gamma

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        """
        x: [batch, seq_len, dim]
        Returns: x_rotated [batch, seq_len, dim]
        """
        batch, seq_len_dim, dim = x.shape
        if seq_len is None:
            seq_len = seq_len_dim

        # Normalized positions [0, 1]
        positions = torch.arange(seq_len_dim, device=x.device, dtype=x.dtype)
        positions_norm = positions / seq_len
        positions_scaled = positions_norm * self.gamma

        # Sinusoidal embeddings
        freqs = torch.einsum('i,j->ij', positions_scaled, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos()[None, :, :]
        sin_emb = emb.sin()[None, :, :]

        # Apply rotation
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        x_rotated = torch.cat([
            x1 * cos_emb[..., :self.dim//2] - x2 * sin_emb[..., self.dim//2:],
            x1 * sin_emb[..., :self.dim//2] + x2 * cos_emb[..., self.dim//2:]
        ], dim=-1)

        return x_rotated
```

#### Self-Purifying Flow Matching実装

```python
class SelfPurifyingFlowMatching:
    def __init__(self, cond_model, uncond_model, threshold=0.5):
        self.cond_model = cond_model
        self.uncond_model = uncond_model
        self.threshold = threshold

    def train_unconditional(self, dataloader, optimizer, epochs=10):
        """Step 1: Unconditionalモデルを学習"""
        for epoch in range(epochs):
            for batch in dataloader:
                wav_gt = batch['wav']
                x_1 = self.encoder(wav_gt)

                # Unconditional Flow Matching
                loss = self._flow_matching_loss(
                    self.uncond_model, x_1, condition=None
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def compute_confidence_scores(self, dataloader):
        """Step 2: 信頼度スコア計算"""
        scores = []

        for batch in dataloader:
            text_ids, wav_gt, style = batch['text_ids'], batch['wav'], batch['style']
            x_1 = self.encoder(wav_gt)
            text_emb = self.text_encoder(text_ids, style)

            # Log-likelihood ratio
            log_p_cond = -self._flow_matching_loss(
                self.cond_model, x_1, condition=(text_emb, style)
            )
            log_p_uncond = -self._flow_matching_loss(
                self.uncond_model, x_1, condition=None
            )

            confidence = log_p_cond - log_p_uncond
            scores.extend(confidence.cpu().tolist())

        return scores

    def filter_dataset(self, dataset, scores):
        """Step 3: データセットフィルタリング"""
        scores_norm = (scores - min(scores)) / (max(scores) - min(scores))
        reliable_indices = [i for i, s in enumerate(scores_norm) if s > self.threshold]
        return torch.utils.data.Subset(dataset, reliable_indices)

    def train_conditional(self, dataloader, optimizer, epochs=100, purify_at_epoch=10):
        """Step 4: Conditionalモデル学習（Self-Purification付き）"""
        for epoch in range(epochs):
            if epoch == purify_at_epoch:
                # Purification
                scores = self.compute_confidence_scores(dataloader)
                filtered = self.filter_dataset(dataloader.dataset, scores)
                dataloader = DataLoader(filtered, batch_size=dataloader.batch_size)

            # Training loop
            for batch in dataloader:
                text_ids, wav_gt, style = batch['text_ids'], batch['wav'], batch['style']
                x_1 = self.encoder(wav_gt)
                text_emb = self.text_encoder(text_ids, style)

                loss = self._flow_matching_loss(
                    self.cond_model, x_1, condition=(text_emb, style)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

---

### 実装チェックリストとロードマップ

#### フェーズ1: 環境構築とデータ準備（2週間）

**Week 1: 環境構築**
- [ ] Python 3.8+、PyTorch 2.0+ (CUDA) インストール
- [ ] 必要ライブラリ: `librosa`, `soundfile`, `tensorboard`
- [ ] GPU環境確認（RTX 3060以上推奨）

**Week 2: データ準備**
- [ ] Hugging Faceからassetsダウンロード:
  ```bash
  git clone https://huggingface.co/Supertone/supertonic assets
  ```
- [ ] `assets/onnx/tts.json`から設定抽出
- [ ] `assets/onnx/unicode_indexer.json`読み込み
- [ ] JSUT/JVSデータセットダウンロード

#### フェーズ2: Speech Autoencoder（2-4週間）

**Week 3-4: Autoencoder実装**
- [ ] エンコーダー: ConvNeXt + Downsampling
- [ ] デコーダー: ConvNeXt + Upsampling
- [ ] Multi-scale STFT Loss
- [ ] 学習ループ

**Week 5-6: 学習と評価**
- [ ] 学習実行（音声のみ、テキスト不要）
- [ ] 音声再構築品質の評価
- [ ] チェックポイント保存

#### フェーズ3: Duration Predictor（2-3週間）

**Week 7-8: DP実装**
- [ ] Transformer Encoder
- [ ] Global Pooling + MLP
- [ ] MSE Loss
- [ ] 学習実行

**Week 9: 評価**
- [ ] Duration予測精度の評価
- [ ] 速度パラメータのテスト

#### フェーズ4: Text Encoder（3-4週間）

**Week 10-11: Text Encoder実装**
- [ ] Embedding Layer
- [ ] LARoPE実装
- [ ] Transformer Encoder with LARoPE
- [ ] AdaLN (Style Conditioning)

**Week 12-13: 統合準備**
- [ ] Vector Estimatorとの接続確認
- [ ] 単体テスト

#### フェーズ5: Vector Estimator（4-6週間）

**Week 14-16: Flow Matching実装**
- [ ] Timestep Embedding
- [ ] U-Net DiT Architecture
  - [ ] Down Blocks
  - [ ] Middle Block
  - [ ] Up Blocks
- [ ] Cross-Attention (LARoPE対応)

**Week 17-19: 学習**
- [ ] Flow Matching Loss実装
- [ ] Text Encoder + Vector Estimator同時学習
- [ ] Euler法による推論

#### フェーズ6: 統合と評価（2-4週間）

**Week 20-21: 統合**
- [ ] 全モジュール結合
- [ ] End-to-End推論パイプライン
- [ ] 長文チャンキング（日本語対応）

**Week 22-23: 評価と最適化**
- [ ] 音声品質評価（MOS、RTF）
- [ ] ハイパーパラメータチューニング
- [ ] Mixed Precision Training

#### フェーズ7: ONNXエクスポート（2週間）

**Week 24-25: ONNX変換**
- [ ] Duration Predictor → ONNX
- [ ] Text Encoder → ONNX
- [ ] Vector Estimator → ONNX
- [ ] Vocoder → ONNX
- [ ] ONNX Runtime検証

#### フェーズ8: Self-Purifying（オプション、2週間）

**Week 26-27**
- [ ] Unconditional Model学習
- [ ] 信頼度スコア計算
- [ ] データフィルタリング
- [ ] 再学習

#### 必要なリソース

**GPU**:
- 最小: RTX 3060 (12GB)
- 推奨: RTX 4070 Ti (16GB) or RTX 4090 (24GB)
- 最適: A100 (40GB/80GB)

**ストレージ**:
- データセット: 50-200GB
- チェックポイント: 10-50GB
- 合計: 100-300GB

**開発期間**:
- フルタイム1名: 6-7ヶ月
- パートタイム: 10-12ヶ月

**学習時間** (RTX 4090):
- Autoencoder: 3-7日
- Duration Predictor: 0.5-1日
- Text Encoder + Vector Estimator: 7-14日
- 合計: 20-40日

#### 重要な注意事項

1. **assetsのダウンロードは必須**: 具体的なハイパーパラメータを取得するため
2. **段階的に実装**: 各モジュールを個別に実装・テスト
3. **TensorBoardでモニタリング**: 学習の進捗を常に確認
4. **チェックポイント保存**: 定期的にモデルを保存
5. **論文を詳細に読む**: 実装の不明点は論文で確認

---

## 学習パイプライン実装状況と次のステップ

### 📊 実装完了状況 (2025年1月)

**✅ 全ての学習コードが実装完了し、すぐに学習を開始できる状態です！**

#### 完了した実装

**Phase 0-3: 全モジュール実装完了**
- ✅ **Phase 0: Speech Autoencoder** (6テスト合格)
  - Encoder, Decoder, Vocoder, MultiScaleDiscriminator
  - AutoencoderTrainer完全実装
- ✅ **Phase 1: Common Modules** (14テスト合格)
  - Attention (LARoPE, MultiHeadAttention)
  - ConvNeXt (Block, Stack, InitialConvNeXt)
  - Layers (CausalConv1d, LayerScale, FiLM, TimeEmbedding, StyleTokenLayer)
- ✅ **Phase 2: TTL (Text-to-Latent)** (7テスト合格)
  - TextEncoder, StyleEncoder, CrossAttentionLARoPE, VectorField
  - TTLTrainer完全実装
- ✅ **Phase 3: Duration Predictor** (5テスト合格)
  - SentenceEncoder, StyleEncoderDP, DurationPredictor
  - DPTrainer完全実装

**データパイプライン: 完全実装**
- ✅ **TTSDataset**: テキスト-音声ペアデータセット (JSON/TXT対応)
- ✅ **AudioDataset**: 音声のみデータセット (Stage 1用)
- ✅ **UnicodeProcessor**: NFKD正規化 + Unicode→Text IDs変換
- ✅ **collate_fn**: バッチ処理 + パディング + マスク生成
- ✅ データセットテスト (6テスト合格)

**統合学習パイプライン: 完全実装**
- ✅ **train_pipeline.py**: 3段階学習の自動化
  - Stage 1: Autoencoder学習
  - Stage 2: TTL学習 (冷凍Autoencoder使用)
  - Stage 3: DP学習 (冷凍Autoencoder使用)
  - チェックポイント自動管理
  - TensorBoard統合
  - 段階スキップ/個別実行機能

**テストとドキュメント**
- ✅ **統合テスト**: 38テスト全て合格 (100%成功率)
- ✅ **training/TRAINING_PIPELINE.md**: 完全な学習ガイド
- ✅ **training/PROGRESS_SUMMARY.md**: 実装完了報告

#### 実装ファイル一覧

```
training/
├── models/
│   ├── autoencoder.py          # Phase 0: Encoder, Decoder, Discriminator
│   ├── ttl.py                  # Phase 2: TTL モデル
│   ├── dp.py                   # Phase 3: Duration Predictor
│   └── common/                 # Phase 1: 共通モジュール
│       ├── attention.py        # LARoPE, MultiHeadAttention
│       ├── convnext.py         # ConvNeXt blocks
│       ├── layers.py           # CausalConv1d, FiLM, etc.
│       └── utils.py            # ユーティリティ
├── data/
│   ├── datasets.py             # TTSDataset, AudioDataset, collate_fn
│   ├── unicode.py              # UnicodeProcessor
│   └── preprocessing.py        # MelSpectrogramExtractor
├── trainers/
│   ├── autoencoder_trainer.py  # Stage 1 trainer
│   ├── ttl_trainer.py          # Stage 2 trainer
│   └── dp_trainer.py           # Stage 3 trainer
├── losses/
│   ├── autoencoder_losses.py   # STFT, Mel, Feature Matching, etc.
│   ├── ttl_losses.py           # Flow Matching Loss
│   └── dp_losses.py            # MSE Loss
├── scripts/
│   └── train_pipeline.py       # 統合学習スクリプト
├── tests/
│   ├── test_autoencoder.py     # Phase 0 tests (6)
│   ├── test_common.py          # Phase 1 tests (14)
│   ├── test_ttl.py             # Phase 2 tests (7)
│   ├── test_dp.py              # Phase 3 tests (5)
│   ├── test_dataset.py         # Dataset tests (6)
│   └── run_all_tests.py        # 統合テストランナー
└── docs/
    ├── TRAINING_PIPELINE.md    # 完全な学習ガイド
    └── PROGRESS_SUMMARY.md     # 実装完了報告
```

---

### 🎯 次のステップ: 学習実行（プランA - LJSpeech高速テスト）

**目標**: 30-45分以内に学習を開始し、パイプラインを検証する

**データセット**: LJSpeech
- 24時間の高品質音声
- 単一話者（女性）
- 22.05kHz (自動的に44.1kHzにリサンプリング)
- 13,100クリップ
- ライセンス: Public Domain
- ダウンロードサイズ: 2.6GB

**推定時間**:
- ダウンロード: 10-15分
- 準備: 5分
- 検証: 5分
- **合計: 30-45分で学習開始可能**

---

#### Step 1: LJSpeechダウンロード（10-15分）

```bash
# データディレクトリ作成
mkdir -p data/raw/ljspeech
cd data/raw/ljspeech

# ダウンロード（2.6GB）
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# または、ブラウザでダウンロード:
# http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# 解凍
tar -xvjf LJSpeech-1.1.tar.bz2

# 戻る
cd ../../..
```

**ディレクトリ構造**:
```
data/raw/ljspeech/LJSpeech-1.1/
├── wavs/
│   ├── LJ001-0001.wav
│   ├── LJ001-0002.wav
│   └── ... (13,100 files)
├── metadata.csv
└── README
```

---

#### Step 2: データセット準備スクリプト作成（5分）

`prepare_ljspeech.py`を作成:

```python
#!/usr/bin/env python3
"""
LJSpeechメタデータをSupertonicTTS形式に変換
"""
import csv
from pathlib import Path

def prepare_ljspeech(
    ljspeech_dir: str = "data/raw/ljspeech/LJSpeech-1.1",
    output_dir: str = "data/processed"
):
    """LJSpeechメタデータをTXT形式に変換"""
    ljspeech_path = Path(ljspeech_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # metadata.csv読み込み
    metadata_file = ljspeech_path / "metadata.csv"

    all_data = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            file_id, text, normalized_text = row
            wav_path = f"wavs/{file_id}.wav"
            # 正規化テキストを使用（TTS向き）
            all_data.append((wav_path, normalized_text))

    # 分割: 95% train, 2.5% val, 2.5% test
    n = len(all_data)
    train_size = int(0.95 * n)
    val_size = int(0.025 * n)

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    # メタデータファイル書き込み
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_file = output_path / f"ljspeech_{split}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for wav_path, text in data:
                f.write(f"{wav_path}|{text}\n")
        print(f"Created {output_file} with {len(data)} samples")

    print(f"\nTotal: {n} samples")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    prepare_ljspeech()
```

---

#### Step 3: メタデータ準備（1分）

```bash
# スクリプト実行
python prepare_ljspeech.py
```

**出力**:
```
Created data/processed/ljspeech_train.txt with 12445 samples
Created data/processed/ljspeech_val.txt with 327 samples
Created data/processed/ljspeech_test.txt with 328 samples

Total: 13100 samples
Train: 12445, Val: 327, Test: 328
```

**メタデータフォーマット** (`ljspeech_train.txt`):
```
wavs/LJ001-0001.wav|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition
wavs/LJ001-0002.wav|in being comparatively modern.
wavs/LJ001-0003.wav|For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process
...
```

---

#### Step 4: データセット検証（2分）

`verify_dataset.py`を作成:

```python
#!/usr/bin/env python3
"""
データセット整合性確認
"""
from pathlib import Path
import librosa

def verify_dataset(
    data_dir: str,
    metadata_file: str,
    check_audio: bool = True,
    num_samples: int = 10
):
    """データセットファイルの存在と読み込み可能性を確認"""
    data_path = Path(data_dir)

    print(f"Verifying dataset: {metadata_file}")
    print(f"Data directory: {data_dir}")

    missing_files = []
    corrupted_files = []
    total_duration = 0.0

    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total samples: {len(lines)}")

    for i, line in enumerate(lines):
        parts = line.strip().split("|")
        if len(parts) < 2:
            print(f"Warning: Invalid line {i}: {line}")
            continue

        audio_path = data_path / parts[0]
        text = parts[1]

        # ファイル存在確認
        if not audio_path.exists():
            missing_files.append(str(audio_path))
            continue

        # 音声読み込みテスト（サンプル）
        if check_audio and i < num_samples:
            try:
                y, sr = librosa.load(str(audio_path), sr=None)
                duration = len(y) / sr
                total_duration += duration
                print(f"Sample {i}: {audio_path.name} - {duration:.2f}s, {sr}Hz")
            except Exception as e:
                corrupted_files.append(str(audio_path))
                print(f"Error loading {audio_path}: {e}")

    # サマリー
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    print(f"Total samples: {len(lines)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    if check_audio and total_duration > 0:
        print(f"Average duration: {total_duration / min(num_samples, len(lines)):.2f}s")

    if missing_files:
        print("\nMissing files (first 10):")
        for f in missing_files[:10]:
            print(f"  - {f}")

    if corrupted_files:
        print("\nCorrupted files:")
        for f in corrupted_files:
            print(f"  - {f}")

    return len(missing_files) == 0 and len(corrupted_files) == 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python verify_dataset.py <data_dir> <metadata_file>")
        sys.exit(1)

    data_dir = sys.argv[1]
    metadata_file = sys.argv[2]
    verify_dataset(data_dir, metadata_file)
```

**実行**:
```bash
python verify_dataset.py \
  data/raw/ljspeech/LJSpeech-1.1 \
  data/processed/ljspeech_train.txt
```

**出力例**:
```
Verifying dataset: data/processed/ljspeech_train.txt
Data directory: data/raw/ljspeech/LJSpeech-1.1
Total samples: 12445
Sample 0: LJ001-0001.wav - 7.02s, 22050Hz
Sample 1: LJ001-0002.wav - 1.85s, 22050Hz
...

==================================================
VERIFICATION SUMMARY
==================================================
Total samples: 12445
Missing files: 0
Corrupted files: 0
Average duration: 5.12s
```

---

#### Step 5: 学習開始（即座）

```bash
cd training

# 3段階全て実行
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs/ljspeech \
  --ae-epochs 200 \
  --ttl-epochs 150 \
  --dp-epochs 100 \
  --batch-size 16 \
  --learning-rate 2e-4
```

**または、段階ごとに実行**:

```bash
# Stage 1のみ: Autoencoder
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs/ljspeech \
  --stage 1 \
  --ae-epochs 200 \
  --batch-size 16

# Stage 2のみ: TTL（Stage 1完了後）
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs/ljspeech \
  --stage 2 \
  --ttl-epochs 150 \
  --batch-size 16

# Stage 3のみ: Duration Predictor（Stage 1完了後）
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/ljspeech/LJSpeech-1.1 \
  --metadata ../data/processed/ljspeech_train.txt \
  --output-dir ../outputs/ljspeech \
  --stage 3 \
  --dp-epochs 100 \
  --batch-size 16
```

---

#### Step 6: 学習監視

**TensorBoard起動**:
```bash
tensorboard --logdir ../outputs/ljspeech/logs
```

ブラウザで `http://localhost:6006` を開く

**監視するメトリクス**:

**Stage 1 (Autoencoder)**:
- `train/loss`: 総損失
- `train/stft`: STFT損失
- `train/mel`: Mel-Spectrogram損失
- `train/adversarial`: 敵対的損失
- `train/discriminator`: Discriminator損失
- `val/stft_loss`, `val/mel_loss`: 検証損失

**Stage 2 (TTL)**:
- `train/flow_matching_loss`: Flow Matching損失
- `val/flow_matching_loss`: 検証損失

**Stage 3 (DP)**:
- `train/mse_loss`: MSE損失
- `train/mean_predicted_duration`: 予測平均長
- `train/mean_ground_truth_duration`: 真値平均長
- `val/mse_loss`: 検証損失

---

### 📈 推定学習時間

**LJSpeech（24時間）でのGPU別推定時間**:

| GPU | Batch Size | Stage 1 (200ep) | Stage 2 (150ep) | Stage 3 (100ep) | 合計 |
|-----|-----------|----------------|----------------|----------------|------|
| RTX 3060 (12GB) | 8-12 | 5-7日 | 3-4日 | 2-3日 | 10-14日 |
| RTX 4070 (12GB) | 16 | 3-5日 | 2-3日 | 1-2日 | 6-10日 |
| RTX 4090 (24GB) | 16-24 | 2-3日 | 1-2日 | 1日 | 4-6日 |
| A100 (40GB) | 32-48 | 1-2日 | 0.5-1日 | 0.5日 | 2-3.5日 |

---

### 🚀 今後の展開: 本番学習

LJSpeechでの検証完了後、以下のステップに進みます:

#### オプション1: Hi-Fi TTS（推奨）
- **292時間、10話者、44.1kHz**
- SupertonicTTSと完全一致のサンプルレート
- 卓越した音質（SNR ≥ 32dB）
- ダウンロード: 60GB
- 学習時間（RTX 4090）: 20-30日

```bash
# Hi-Fi TTSダウンロード
mkdir -p data/raw/hifi_tts && cd data/raw/hifi_tts
wget http://www.openslr.org/resources/109/hi_fi_tts_v0.tar.gz
tar -xvzf hi_fi_tts_v0.tar.gz
cd ../../..

# メタデータ準備（prepare_hifitts.pyを作成）
python prepare_hifitts.py

# 学習開始
cd training
uv run python scripts/train_pipeline.py \
  --config ../assets/onnx/tts.json \
  --data-dir ../data/raw/hifi_tts/hi_fi_tts_v0 \
  --metadata ../data/processed/hifitts_train.txt \
  --output-dir ../outputs/hifitts \
  --ae-epochs 100 \
  --ttl-epochs 80 \
  --dp-epochs 50 \
  --batch-size 16
```

#### オプション2: LibriTTS-R（大規模）
- **585時間、2,456話者、24kHz**
- 最大の話者多様性
- ボイスクローニングに最適
- ダウンロード: 200GB
- 学習時間（RTX 4090）: 30-50日

#### オプション3: 日本語対応
CLAUDE.mdの「日本語対応について」セクションを参照:
- JSUT + JVS Corpus（50時間）
- 推定工数: 4-7ヶ月
- またはStableTTS（日本語対応済み）をベースにする

---

### 📚 参考リンク

**ドキュメント**:
- **training/TRAINING_PIPELINE.md**: 完全な学習ガイド
- **training/PROGRESS_SUMMARY.md**: 実装完了報告

**データセット**:
- LJSpeech: http://data.keithito.com/data/speech/
- Hi-Fi TTS: http://www.openslr.org/109/
- LibriTTS-R: http://www.openslr.org/141/

**テスト**:
```bash
cd training
uv run python tests/run_all_tests.py  # 38テスト全て実行
```

---

## ライセンス

- サンプルコード: MIT License
- モデル: OpenRAIL-M License
- ONNX Runtime: MIT License
