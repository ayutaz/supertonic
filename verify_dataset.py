#!/usr/bin/env python3
"""
データセット整合性確認スクリプト

使用方法:
    python verify_dataset.py <data_dir> <metadata_file>

例:
    python verify_dataset.py data/raw/ljspeech/LJSpeech-1.1 data/processed/ljspeech_train.txt
"""
from pathlib import Path
import sys


def verify_dataset(
    data_dir: str,
    metadata_file: str,
    check_audio: bool = True,
    num_samples: int = 10
):
    """
    データセットファイルの存在と読み込み可能性を確認

    引数:
        data_dir: データディレクトリ
        metadata_file: メタデータファイル
        check_audio: 音声ファイルを実際に読み込んで確認するか
        num_samples: 確認するサンプル数

    戻り値:
        bool: エラーがなければTrue
    """
    data_path = Path(data_dir)

    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    print(f"Data directory:   {data_dir}")
    print(f"Metadata file:    {metadata_file}")
    print(f"Audio check:      {'Enabled' if check_audio else 'Disabled'}")
    print(f"Samples to check: {num_samples if check_audio else 'N/A'}")
    print("=" * 60)

    # Check data directory exists
    if not data_path.exists():
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        return False

    # Check metadata file exists
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        print(f"\n[ERROR] Metadata file not found: {metadata_file}")
        return False

    missing_files = []
    corrupted_files = []
    total_duration = 0.0

    # Read metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"\n[INFO] Total samples in metadata: {len(lines)}")

    # Check each line
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) < 2:
            print(f"[WARNING] Invalid line {i}: {line}")
            continue

        audio_path = data_path / parts[0]
        text = parts[1]

        # Check file exists
        if not audio_path.exists():
            missing_files.append(str(audio_path))
            if len(missing_files) <= 5:
                print(f"[ERROR] Missing file: {audio_path}")
            continue

        # Check audio can be loaded (sample a few files)
        if check_audio and i < num_samples:
            try:
                # Try to load with librosa
                import librosa
                y, sr = librosa.load(str(audio_path), sr=None)
                duration = len(y) / sr
                total_duration += duration

                print(f"\n[OK] Sample {i}:")
                print(f"     File:     {audio_path.name}")
                print(f"     Duration: {duration:.2f}s")
                print(f"     SR:       {sr}Hz")
                print(f"     Text:     {text[:60]}...")

            except Exception as e:
                corrupted_files.append(str(audio_path))
                print(f"\n[ERROR] Failed to load {audio_path}:")
                print(f"        {e}")

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total samples:    {len(lines)}")
    print(f"Missing files:    {len(missing_files)}")
    print(f"Corrupted files:  {len(corrupted_files)}")

    if check_audio and total_duration > 0:
        avg_duration = total_duration / min(num_samples, len(lines))
        print(f"Avg duration:     {avg_duration:.2f}s")
        print(f"Total checked:    {min(num_samples, len(lines))} samples")

    # Show missing files (first 10)
    if missing_files:
        print("\n" + "=" * 60)
        print("MISSING FILES (first 10):")
        print("=" * 60)
        for f in missing_files[:10]:
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    # Show corrupted files
    if corrupted_files:
        print("\n" + "=" * 60)
        print("CORRUPTED FILES:")
        print("=" * 60)
        for f in corrupted_files:
            print(f"  - {f}")

    # Result
    success = len(missing_files) == 0 and len(corrupted_files) == 0

    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] All checks passed!")
        print("=" * 60)
        print("\nYou can now start training:")
        print("\ncd training")
        print("uv run python scripts/train_pipeline.py \\")
        print("  --config ../assets/onnx/tts.json \\")
        print(f"  --data-dir ../{data_dir} \\")
        print(f"  --metadata ../{metadata_file} \\")
        print("  --output-dir ../outputs \\")
        print("  --ae-epochs 200 \\")
        print("  --ttl-epochs 150 \\")
        print("  --dp-epochs 100 \\")
        print("  --batch-size 16")
    else:
        print("[FAILED] Some checks failed!")
        print("=" * 60)
        print("\nPlease fix the errors above before training.")

    return success


def main():
    """コマンドライン実行"""
    if len(sys.argv) < 3:
        print("Usage: python verify_dataset.py <data_dir> <metadata_file>")
        print("\nExample:")
        print("  python verify_dataset.py \\")
        print("    data/raw/ljspeech/LJSpeech-1.1 \\")
        print("    data/processed/ljspeech_train.txt")
        sys.exit(1)

    data_dir = sys.argv[1]
    metadata_file = sys.argv[2]

    # Optional: check audio files
    check_audio = "--check-audio" in sys.argv
    num_samples = 10

    # Optional: number of samples to check
    for i, arg in enumerate(sys.argv):
        if arg == "--num-samples" and i + 1 < len(sys.argv):
            try:
                num_samples = int(sys.argv[i + 1])
            except ValueError:
                print(f"[WARNING] Invalid --num-samples value: {sys.argv[i + 1]}")

    success = verify_dataset(data_dir, metadata_file, check_audio, num_samples)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
