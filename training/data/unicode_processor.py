"""
Unicode Processor for character-level text processing
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union


class UnicodeProcessor:
    """
    Unicode文字処理クラス

    文字レベルでテキストを処理し、Unicodeインデックスに変換
    G2P (Grapheme-to-Phoneme) を使用せず、直接文字を処理
    """

    def __init__(
        self,
        char_dict_path: Optional[Union[str, Path]] = None,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        """
        引数:
            char_dict_path: 文字辞書のパス (Noneの場合は空の辞書)
            pad_token: パディングトークン
            unk_token: 未知トークン
            bos_token: 開始トークン
            eos_token: 終了トークン
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # 特殊トークン
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]

        # 文字辞書の初期化
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

        # 文字辞書の読み込み
        if char_dict_path is not None:
            self.load_char_dict(char_dict_path)
        else:
            # デフォルトの特殊トークンを追加
            for i, token in enumerate(self.special_tokens):
                self.char2idx[token] = i
                self.idx2char[i] = token

    def load_char_dict(self, char_dict_path: Union[str, Path]):
        """
        文字辞書を読み込み

        引数:
            char_dict_path: 文字辞書のパス
        """
        char_dict_path = Path(char_dict_path)

        if not char_dict_path.exists():
            raise FileNotFoundError(f"Char dict not found: {char_dict_path}")

        with open(char_dict_path, "r", encoding="utf-8") as f:
            char_dict = json.load(f)

        # 特殊トークンを先に追加
        for i, token in enumerate(self.special_tokens):
            self.char2idx[token] = i
            self.idx2char[i] = token

        # 文字辞書を追加
        offset = len(self.special_tokens)
        for i, char in enumerate(char_dict):
            idx = i + offset
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def save_char_dict(self, char_dict_path: Union[str, Path]):
        """
        文字辞書を保存

        引数:
            char_dict_path: 保存先パス
        """
        char_dict_path = Path(char_dict_path)
        char_dict_path.parent.mkdir(parents=True, exist_ok=True)

        # 特殊トークンを除いた文字リスト
        chars = [
            self.idx2char[i]
            for i in sorted(self.idx2char.keys())
            if self.idx2char[i] not in self.special_tokens
        ]

        with open(char_dict_path, "w", encoding="utf-8") as f:
            json.dump(chars, f, ensure_ascii=False, indent=2)

    def build_char_dict_from_texts(self, texts: List[str]):
        """
        テキストリストから文字辞書を構築

        引数:
            texts: テキストのリスト
        """
        # ユニークな文字を抽出
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # ソートして追加
        chars = sorted(unique_chars)

        # 特殊トークンを先に追加
        for i, token in enumerate(self.special_tokens):
            self.char2idx[token] = i
            self.idx2char[i] = token

        # 文字を追加
        offset = len(self.special_tokens)
        for i, char in enumerate(chars):
            idx = i + offset
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def text_to_sequence(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        テキストをインデックスシーケンスに変換

        引数:
            text: 入力テキスト
            add_bos: 開始トークンを追加するか
            add_eos: 終了トークンを追加するか

        戻り値:
            インデックスシーケンス
        """
        sequence = []

        if add_bos:
            sequence.append(self.char2idx[self.bos_token])

        for char in text:
            if char in self.char2idx:
                sequence.append(self.char2idx[char])
            else:
                sequence.append(self.char2idx[self.unk_token])

        if add_eos:
            sequence.append(self.char2idx[self.eos_token])

        return sequence

    def sequence_to_text(self, sequence: Union[List[int], np.ndarray, torch.Tensor]) -> str:
        """
        インデックスシーケンスをテキストに変換

        引数:
            sequence: インデックスシーケンス

        戻り値:
            テキスト
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()
        elif isinstance(sequence, list):
            sequence = np.array(sequence)

        text = ""
        for idx in sequence:
            idx = int(idx)
            if idx in self.idx2char:
                char = self.idx2char[idx]
                # 特殊トークンは除外
                if char not in self.special_tokens:
                    text += char
            else:
                text += self.unk_token

        return text

    def __len__(self) -> int:
        """辞書サイズを返す"""
        return len(self.char2idx)

    @property
    def vocab_size(self) -> int:
        """語彙サイズを返す"""
        return len(self.char2idx)

    @property
    def pad_id(self) -> int:
        """パディングIDを返す"""
        return self.char2idx[self.pad_token]

    @property
    def unk_id(self) -> int:
        """未知トークンIDを返す"""
        return self.char2idx[self.unk_token]

    @property
    def bos_id(self) -> int:
        """開始トークンIDを返す"""
        return self.char2idx[self.bos_token]

    @property
    def eos_id(self) -> int:
        """終了トークンIDを返す"""
        return self.char2idx[self.eos_token]


def create_unicode_indexer(texts: List[str], save_path: Optional[Union[str, Path]] = None):
    """
    テキストリストからUnicodeインデクサーを作成

    引数:
        texts: テキストのリスト
        save_path: 保存先パス (Noneの場合は保存しない)

    戻り値:
        UnicodeProcessor
    """
    processor = UnicodeProcessor()
    processor.build_char_dict_from_texts(texts)

    if save_path is not None:
        processor.save_char_dict(save_path)

    return processor
