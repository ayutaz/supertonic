"""
Unicode Text Processing for Training

学習時のテキスト処理モジュール
"""

import json
import torch
import torch.nn as nn
from unicodedata import normalize
from typing import Optional, Tuple, List


class UnicodeIndexer:
    """
    Unicode文字をテキストIDに変換するインデクサー

    unicode_indexer.jsonを使用してUnicode値→Text IDsの変換を行う
    """

    def __init__(self, unicode_indexer_path: str):
        """
        引数:
            unicode_indexer_path: unicode_indexer.jsonのパス
        """
        with open(unicode_indexer_path, "r", encoding="utf-8") as f:
            # 文字列キーを整数に変換
            indexer_str = json.load(f)
            self.indexer = {int(k): v for k, v in indexer_str.items()}

        # 語彙サイズを計算
        self.vocab_size = max(self.indexer.values()) + 1

    def __call__(self, unicode_values: List[int]) -> List[int]:
        """
        Unicode値のリストをText IDsのリストに変換

        引数:
            unicode_values: Unicode値のリスト

        戻り値:
            Text IDsのリスト
        """
        return [self.indexer.get(val, 0) for val in unicode_values]


class UnicodeProcessor:
    """
    テキスト前処理とUnicode変換を行うプロセッサー

    ONNX推論コード (py/helper.py) の実装を参考にしている
    """

    def __init__(self, unicode_indexer_path: Optional[str] = None):
        """
        引数:
            unicode_indexer_path: unicode_indexer.jsonのパス（Noneの場合は変換なし）
        """
        self.indexer = None
        if unicode_indexer_path is not None:
            self.indexer = UnicodeIndexer(unicode_indexer_path)

    def preprocess_text(self, text: str) -> str:
        """
        テキスト正規化

        引数:
            text: 入力テキスト

        戻り値:
            正規化されたテキスト
        """
        # NFKD正規化 (Compatibility Decomposition)
        text = normalize("NFKD", text)
        return text

    def text_to_unicode_values(self, text: str) -> List[int]:
        """
        テキストをUnicode値のリストに変換

        引数:
            text: 入力テキスト

        戻り値:
            Unicode値のリスト
        """
        return [ord(char) for char in text]

    def text_to_ids(self, text: str) -> List[int]:
        """
        テキストをText IDsに変換

        引数:
            text: 入力テキスト

        戻り値:
            Text IDsのリスト
        """
        # 正規化
        text = self.preprocess_text(text)

        # Unicode値に変換
        unicode_values = self.text_to_unicode_values(text)

        # Text IDsに変換
        if self.indexer is not None:
            return self.indexer(unicode_values)
        else:
            # Indexerがない場合はUnicode値をそのまま返す
            return unicode_values

    def __call__(
        self,
        text_list: List[str],
        return_tensors: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        テキストリストをバッチ処理してText IDsとマスクを返す

        引数:
            text_list: テキストのリスト
            return_tensors: Tensorで返すか（Falseの場合はリスト）

        戻り値:
            (text_ids, text_mask): Text IDsとマスク
                - text_ids: [batch, max_len]
                - text_mask: [batch, 1, max_len]
        """
        # 各テキストをIDsに変換
        ids_list = [self.text_to_ids(text) for text in text_list]

        # 長さを取得
        text_lengths = [len(ids) for ids in ids_list]
        max_len = max(text_lengths)

        if return_tensors:
            # パディング
            text_ids = torch.zeros(len(text_list), max_len, dtype=torch.long)
            for i, ids in enumerate(ids_list):
                text_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

            # マスクを作成
            text_mask = torch.zeros(len(text_list), 1, max_len, dtype=torch.float32)
            for i, length in enumerate(text_lengths):
                text_mask[i, 0, :length] = 1.0

            return text_ids, text_mask
        else:
            return ids_list, text_lengths


def length_to_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None,
) -> torch.Tensor:
    """
    長さのテンソルからマスクテンソルを作成

    引数:
        lengths: 各シーケンスの長さ [batch]
        max_len: 最大長（Noneの場合はlengthsの最大値）

    戻り値:
        マスク [batch, 1, max_len]
        有効部分=1.0、パディング=0.0
    """
    if max_len is None:
        max_len = lengths.max().item()

    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    mask = mask.float().unsqueeze(1)  # [batch, 1, max_len]

    return mask
