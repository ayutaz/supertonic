"""
PyTorch Dataset classes for TTS training
"""

import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset

from .preprocessing import AudioPreprocessor, MelSpectrogramExtractor
from .unicode_processor import UnicodeProcessor


class TTSDataset(Dataset):
    """
    TTS学習用データセット

    音声ファイルとテキストのペアを読み込み、前処理を行う
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Union[str, Path],
        unicode_processor: UnicodeProcessor,
        mel_extractor: MelSpectrogramExtractor,
        sample_rate: int = 44100,
        max_wav_length: Optional[int] = None,
        max_text_length: Optional[int] = None,
        normalize: bool = True,
        trim_silence: bool = False,
    ):
        """
        引数:
            data_dir: データディレクトリ
            metadata_file: メタデータファイル (JSON or TXT)
            unicode_processor: Unicode処理器
            mel_extractor: Mel-Spectrogram抽出器
            sample_rate: サンプリングレート
            max_wav_length: 最大音声長 (サンプル数、Noneの場合は制限なし)
            max_text_length: 最大テキスト長 (文字数、Noneの場合は制限なし)
            normalize: 音声を正規化するか
            trim_silence: 無音部分を削除するか
        """
        self.data_dir = Path(data_dir)
        self.unicode_processor = unicode_processor
        self.mel_extractor = mel_extractor
        self.sample_rate = sample_rate
        self.max_wav_length = max_wav_length
        self.max_text_length = max_text_length

        # Audio preprocessor
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            normalize=normalize,
            trim_silence=trim_silence,
        )

        # Load metadata
        self.metadata = self._load_metadata(metadata_file)

        # Filter by length
        if max_wav_length is not None or max_text_length is not None:
            self.metadata = self._filter_by_length(self.metadata)

    def _load_metadata(self, metadata_file: Union[str, Path]) -> List[Dict[str, str]]:
        """
        メタデータを読み込み

        引数:
            metadata_file: メタデータファイルのパス

        戻り値:
            メタデータのリスト [{"audio_path": ..., "text": ...}, ...]
        """
        metadata_file = Path(metadata_file)

        if metadata_file.suffix == ".json":
            # JSON形式
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            # TXT形式 (audio_path|text)
            metadata = []
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 2:
                        audio_path = parts[0]
                        text = "|".join(parts[1:])  # テキストに'|'が含まれる場合
                        metadata.append({
                            "audio_path": audio_path,
                            "text": text,
                        })

        return metadata

    def _filter_by_length(self, metadata: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        長さでフィルタリング

        引数:
            metadata: メタデータリスト

        戻り値:
            フィルタリング後のメタデータリスト
        """
        filtered = []

        for item in metadata:
            # テキスト長チェック
            if self.max_text_length is not None:
                if len(item["text"]) > self.max_text_length:
                    continue

            # 音声長チェック (実際に読み込まないと分からないため、ここではスキップ)
            # 必要に応じて事前処理で音声長を記録しておく

            filtered.append(item)

        return filtered

    def __len__(self) -> int:
        """データセットのサイズ"""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データを取得

        引数:
            idx: インデックス

        戻り値:
            データ辞書
            {
                "wav": 音声波形 [time],
                "mel": Mel-Spectrogram [n_mels, frames],
                "text": テキストインデックス [text_len],
                "text_length": テキスト長,
                "wav_length": 音声長,
                "mel_length": Mel-Spectrogram長,
            }
        """
        item = self.metadata[idx]

        # Load audio
        audio_path = self.data_dir / item["audio_path"]
        wav = self.audio_preprocessor.load_and_process(str(audio_path))

        # Crop if too long
        if self.max_wav_length is not None and len(wav) > self.max_wav_length:
            start = random.randint(0, len(wav) - self.max_wav_length)
            wav = wav[start : start + self.max_wav_length]

        # Convert to tensor
        wav_tensor = torch.from_numpy(wav).float()

        # Extract Mel-Spectrogram
        mel = self.mel_extractor(wav_tensor.unsqueeze(0)).squeeze(0)

        # Process text
        text = item["text"]
        text_indices = self.unicode_processor.text_to_ids(text)
        text_tensor = torch.LongTensor(text_indices)

        return {
            "wav": wav_tensor,
            "mel": mel,
            "text": text_tensor,
            "text_length": len(text_indices),
            "wav_length": len(wav),
            "mel_length": mel.size(1),
        }


class AudioDataset(Dataset):
    """
    音声のみのデータセット (Autoencoder学習用)

    テキストなしで音声のみを読み込む
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        audio_files: List[str],
        mel_extractor: MelSpectrogramExtractor,
        sample_rate: int = 44100,
        segment_length: Optional[int] = None,
        normalize: bool = True,
        trim_silence: bool = False,
    ):
        """
        引数:
            data_dir: データディレクトリ
            audio_files: 音声ファイルのリスト
            mel_extractor: Mel-Spectrogram抽出器
            sample_rate: サンプリングレート
            segment_length: セグメント長 (サンプル数、Noneの場合は全体)
            normalize: 音声を正規化するか
            trim_silence: 無音部分を削除するか
        """
        self.data_dir = Path(data_dir)
        self.audio_files = audio_files
        self.mel_extractor = mel_extractor
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        # Audio preprocessor
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            normalize=normalize,
            trim_silence=trim_silence,
        )

    def __len__(self) -> int:
        """データセットのサイズ"""
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データを取得

        引数:
            idx: インデックス

        戻り値:
            データ辞書
            {
                "wav": 音声波形 [time],
                "mel": Mel-Spectrogram [n_mels, frames],
                "wav_length": 音声長,
                "mel_length": Mel-Spectrogram長,
            }
        """
        audio_file = self.audio_files[idx]
        audio_path = self.data_dir / audio_file

        # Load audio
        wav = self.audio_preprocessor.load_and_process(str(audio_path))

        # Random crop if segment_length is specified
        if self.segment_length is not None:
            if len(wav) > self.segment_length:
                start = random.randint(0, len(wav) - self.segment_length)
                wav = wav[start : start + self.segment_length]
            elif len(wav) < self.segment_length:
                # Pad if too short
                pad_length = self.segment_length - len(wav)
                wav = np.pad(wav, (0, pad_length), mode="constant")

        # Convert to tensor
        wav_tensor = torch.from_numpy(wav).float()

        # Extract Mel-Spectrogram
        mel = self.mel_extractor(wav_tensor.unsqueeze(0)).squeeze(0)

        return {
            "wav": wav_tensor,
            "mel": mel,
            "wav_length": len(wav),
            "mel_length": mel.size(1),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader

    バッチ内のデータを適切にパディングして結合

    引数:
        batch: データのリスト

    戻り値:
        バッチ化されたデータ
    """
    # 最大長を取得
    max_wav_length = max(item["wav_length"] for item in batch)
    max_mel_length = max(item["mel_length"] for item in batch)

    batch_size = len(batch)
    n_mels = batch[0]["mel"].size(0)

    # テキストがある場合
    has_text = "text" in batch[0]
    if has_text:
        max_text_length = max(item["text_length"] for item in batch)
        text_padded = torch.zeros(batch_size, max_text_length, dtype=torch.long)
        text_lengths = torch.LongTensor([item["text_length"] for item in batch])

    # パディング
    wav_padded = torch.zeros(batch_size, max_wav_length)
    mel_padded = torch.zeros(batch_size, n_mels, max_mel_length)
    wav_lengths = torch.LongTensor([item["wav_length"] for item in batch])
    mel_lengths = torch.LongTensor([item["mel_length"] for item in batch])

    for i, item in enumerate(batch):
        wav_length = item["wav_length"]
        mel_length = item["mel_length"]

        wav_padded[i, :wav_length] = item["wav"]
        mel_padded[i, :, :mel_length] = item["mel"]

        if has_text:
            text_length = item["text_length"]
            text_padded[i, :text_length] = item["text"]

    result = {
        "wav": wav_padded,
        "mel": mel_padded,
        "wav_lengths": wav_lengths,
        "mel_lengths": mel_lengths,
    }

    if has_text:
        result["text"] = text_padded
        result["text_lengths"] = text_lengths

    return result
