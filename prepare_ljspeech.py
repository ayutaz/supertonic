#!/usr/bin/env python3
"""
LJSpeechメタデータをSupertonicTTS形式に変換

使用方法:
    python prepare_ljspeech.py

出力:
    data/processed/ljspeech_train.txt (95%)
    data/processed/ljspeech_val.txt (2.5%)
    data/processed/ljspeech_test.txt (2.5%)
"""
import csv
from pathlib import Path


def prepare_ljspeech(
    ljspeech_dir: str = "data/raw/ljspeech/LJSpeech-1.1",
    output_dir: str = "data/processed"
):
    """
    LJSpeechメタデータをSupertonicTTS形式に変換

    引数:
        ljspeech_dir: LJSpeechデータディレクトリ
        output_dir: 出力ディレクトリ

    出力フォーマット:
        wavs/LJ001-0001.wav|Normalized text transcription
    """
    ljspeech_path = Path(ljspeech_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # metadata.csv読み込み
    metadata_file = ljspeech_path / "metadata.csv"

    if not metadata_file.exists():
        print(f"[ERROR] Metadata file not found: {metadata_file}")
        print(f"[INFO] Please download LJSpeech-1.1 and place it in {ljspeech_dir}")
        return

    print(f"[INFO] Reading metadata from {metadata_file}")

    all_data = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) != 3:
                print(f"[WARNING] Skipping invalid row: {row}")
                continue

            file_id, text, normalized_text = row
            wav_path = f"wavs/{file_id}.wav"

            # 正規化テキストを使用（TTS向き）
            all_data.append((wav_path, normalized_text))

    print(f"[INFO] Loaded {len(all_data)} samples")

    # 分割: 95% train, 2.5% val, 2.5% test
    n = len(all_data)
    train_size = int(0.95 * n)
    val_size = int(0.025 * n)

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    print(f"\n[INFO] Splitting dataset:")
    print(f"  Train: {len(train_data)} samples (95%)")
    print(f"  Val:   {len(val_data)} samples (2.5%)")
    print(f"  Test:  {len(test_data)} samples (2.5%)")

    # メタデータファイル書き込み
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_file = output_path / f"ljspeech_{split}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for wav_path, text in data:
                f.write(f"{wav_path}|{text}\n")
        print(f"\n[OK] Created {output_file}")
        print(f"     {len(data)} samples")

        # Show first 3 samples
        if len(data) > 0:
            print(f"     First sample: {data[0][0]} | {data[0][1][:50]}...")

    print(f"\n[SUCCESS] Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"  1. Verify dataset:")
    print(f"     python verify_dataset.py {ljspeech_dir} {output_path}/ljspeech_train.txt")
    print(f"  2. Start training:")
    print(f"     cd training")
    print(f"     uv run python scripts/train_pipeline.py \\")
    print(f"       --config ../assets/onnx/tts.json \\")
    print(f"       --data-dir ../{ljspeech_dir} \\")
    print(f"       --metadata ../{output_path}/ljspeech_train.txt \\")
    print(f"       --output-dir ../outputs/ljspeech \\")
    print(f"       --ae-epochs 200 \\")
    print(f"       --ttl-epochs 150 \\")
    print(f"       --dp-epochs 100 \\")
    print(f"       --batch-size 16")


if __name__ == "__main__":
    prepare_ljspeech()
