"""
TTSDataset完全テスト

TTSDatasetの全機能を検証:
- メタデータ読み込み (JSON/TXT)
- Unicode処理
- 音声処理
- バッチ処理
- DataLoader統合
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader

from data.datasets import TTSDataset, AudioDataset, collate_fn
from data.unicode import UnicodeProcessor
from data.preprocessing import MelSpectrogramExtractor


def create_dummy_audio_file(path: Path, duration: float = 1.0, sample_rate: int = 44100):
    """
    ダミー音声ファイルを作成

    引数:
        path: 保存パス
        duration: 長さ (秒)
        sample_rate: サンプリングレート
    """
    num_samples = int(duration * sample_rate)
    # Simple sine wave
    t = np.linspace(0, duration, num_samples)
    wav = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    wav = wav.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sample_rate)


def test_ttsdataset_txt_metadata():
    """
    Test 1: TXT形式メタデータ読み込み
    """
    print("\n=== Test 1: TXT Metadata Loading ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy audio files
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()

        create_dummy_audio_file(audio_dir / "audio1.wav", duration=1.0)
        create_dummy_audio_file(audio_dir / "audio2.wav", duration=1.5)
        create_dummy_audio_file(audio_dir / "audio3.wav", duration=2.0)

        # Create metadata TXT
        metadata_file = tmpdir / "metadata.txt"
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(f"{audio_dir}/audio1.wav|Hello world.\n")
            f.write(f"{audio_dir}/audio2.wav|This is a test.\n")
            f.write(f"{audio_dir}/audio3.wav|SupertonicTTS is fast!\n")

        # Create UnicodeProcessor
        unicode_processor = UnicodeProcessor()

        # Create MelSpectrogramExtractor
        mel_extractor = MelSpectrogramExtractor(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            n_mels=128,
        )

        # Create TTSDataset
        dataset = TTSDataset(
            data_dir=tmpdir,
            metadata_file=metadata_file,
            unicode_processor=unicode_processor,
            mel_extractor=mel_extractor,
            sample_rate=44100,
        )

        assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"

        # Check first sample
        sample = dataset[0]
        assert "wav" in sample, "Missing 'wav' key"
        assert "mel" in sample, "Missing 'mel' key"
        assert "text" in sample, "Missing 'text' key"
        assert "text_length" in sample, "Missing 'text_length' key"
        assert "wav_length" in sample, "Missing 'wav_length' key"
        assert "mel_length" in sample, "Missing 'mel_length' key"

        print(f"[OK] Loaded {len(dataset)} samples from TXT metadata")
        print(f"[OK] Sample 0: wav shape={sample['wav'].shape}, mel shape={sample['mel'].shape}, text length={sample['text_length']}")

        return True


def test_ttsdataset_json_metadata():
    """
    Test 2: JSON形式メタデータ読み込み
    """
    print("\n=== Test 2: JSON Metadata Loading ===")

    import json

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy audio files
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()

        create_dummy_audio_file(audio_dir / "audio1.wav", duration=1.0)
        create_dummy_audio_file(audio_dir / "audio2.wav", duration=1.5)

        # Create metadata JSON
        metadata_file = tmpdir / "metadata.json"
        metadata = [
            {"audio": str(audio_dir / "audio1.wav"), "text": "Hello world."},
            {"audio": str(audio_dir / "audio2.wav"), "text": "This is a test."},
        ]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        # Create UnicodeProcessor
        unicode_processor = UnicodeProcessor()

        # Create MelSpectrogramExtractor
        mel_extractor = MelSpectrogramExtractor(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            n_mels=128,
        )

        # Create TTSDataset
        dataset = TTSDataset(
            data_dir=tmpdir,
            metadata_file=metadata_file,
            unicode_processor=unicode_processor,
            mel_extractor=mel_extractor,
            sample_rate=44100,
        )

        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"

        print(f"[OK] Loaded {len(dataset)} samples from JSON metadata")

        return True


def test_unicode_processing():
    """
    Test 3: Unicode処理
    """
    print("\n=== Test 3: Unicode Processing ===")

    unicode_processor = UnicodeProcessor()

    # Test text processing
    test_texts = [
        "Hello world.",
        "This is a test!",
        "Numbers: 123, 456.",
        "Special chars: @#$%",
    ]

    for text in test_texts:
        text_ids, text_mask = unicode_processor([text])

        assert text_ids.shape[0] == 1, "Batch size should be 1"
        assert text_mask.shape[0] == 1, "Batch size should be 1"
        assert text_ids.shape[1] == text_mask.shape[2], "Text IDs and mask length mismatch"

        print(f"[OK] Text: '{text}' -> IDs shape: {text_ids.shape}, Mask shape: {text_mask.shape}")

    return True


def test_collate_fn():
    """
    Test 4: collate_fn バッチ処理
    """
    print("\n=== Test 4: Collate Function (Batching) ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy audio files with different lengths
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()

        create_dummy_audio_file(audio_dir / "audio1.wav", duration=1.0)
        create_dummy_audio_file(audio_dir / "audio2.wav", duration=1.5)
        create_dummy_audio_file(audio_dir / "audio3.wav", duration=2.0)

        # Create metadata TXT
        metadata_file = tmpdir / "metadata.txt"
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(f"{audio_dir}/audio1.wav|Short text.\n")
            f.write(f"{audio_dir}/audio2.wav|Medium length text here.\n")
            f.write(f"{audio_dir}/audio3.wav|This is a much longer text for testing padding.\n")

        # Create dataset
        unicode_processor = UnicodeProcessor()
        mel_extractor = MelSpectrogramExtractor(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            n_mels=128,
        )

        dataset = TTSDataset(
            data_dir=tmpdir,
            metadata_file=metadata_file,
            unicode_processor=unicode_processor,
            mel_extractor=mel_extractor,
            sample_rate=44100,
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Get one batch
        batch = next(iter(dataloader))

        assert "wav" in batch, "Missing 'wav' in batch"
        assert "mel" in batch, "Missing 'mel' in batch"
        assert "text" in batch, "Missing 'text' in batch"
        assert "wav_lengths" in batch, "Missing 'wav_lengths' in batch"
        assert "mel_lengths" in batch, "Missing 'mel_lengths' in batch"
        assert "text_lengths" in batch, "Missing 'text_lengths' in batch"

        print(f"[OK] Batch wav shape: {batch['wav'].shape}")
        print(f"[OK] Batch mel shape: {batch['mel'].shape}")
        print(f"[OK] Batch text shape: {batch['text'].shape}")
        print(f"[OK] wav_lengths: {batch['wav_lengths']}")
        print(f"[OK] mel_lengths: {batch['mel_lengths']}")
        print(f"[OK] text_lengths: {batch['text_lengths']}")

        # Check shapes
        batch_size = 3
        assert batch["wav"].shape[0] == batch_size
        assert batch["mel"].shape[0] == batch_size
        assert batch["text"].shape[0] == batch_size

        # Check padding (max length should be used)
        max_wav_len = max(batch["wav_lengths"])
        max_mel_len = max(batch["mel_lengths"])
        max_text_len = max(batch["text_lengths"])

        assert batch["wav"].shape[1] == max_wav_len
        assert batch["mel"].shape[2] == max_mel_len
        assert batch["text"].shape[1] == max_text_len

        print("[OK] Padding is correct (all samples padded to max length)")

        return True


def test_dataloader_iteration():
    """
    Test 5: DataLoader完全統合
    """
    print("\n=== Test 5: DataLoader Integration ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy audio files
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()

        num_samples = 10
        for i in range(num_samples):
            create_dummy_audio_file(audio_dir / f"audio{i}.wav", duration=1.0 + i * 0.1)

        # Create metadata TXT
        metadata_file = tmpdir / "metadata.txt"
        with open(metadata_file, "w", encoding="utf-8") as f:
            for i in range(num_samples):
                f.write(f"{audio_dir}/audio{i}.wav|Sample text {i}.\n")

        # Create dataset
        unicode_processor = UnicodeProcessor()
        mel_extractor = MelSpectrogramExtractor(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            n_mels=128,
        )

        dataset = TTSDataset(
            data_dir=tmpdir,
            metadata_file=metadata_file,
            unicode_processor=unicode_processor,
            mel_extractor=mel_extractor,
            sample_rate=44100,
        )

        # Create DataLoader
        batch_size = 4
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Iterate through batches
        num_batches = 0
        total_samples = 0

        for batch in dataloader:
            num_batches += 1
            batch_size_actual = batch["wav"].shape[0]
            total_samples += batch_size_actual

            print(f"  Batch {num_batches}: {batch_size_actual} samples, "
                  f"wav shape: {batch['wav'].shape}, "
                  f"mel shape: {batch['mel'].shape}, "
                  f"text shape: {batch['text'].shape}")

        assert total_samples == num_samples, f"Expected {num_samples} samples, got {total_samples}"

        print(f"[OK] Iterated through {num_batches} batches")
        print(f"[OK] Total samples: {total_samples}")

        return True


def test_audiodataset():
    """
    Test 6: AudioDataset (音声のみ、Autoencoder用)
    """
    print("\n=== Test 6: AudioDataset (Audio-only) ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy audio files
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()

        create_dummy_audio_file(audio_dir / "audio1.wav", duration=1.0)
        create_dummy_audio_file(audio_dir / "audio2.wav", duration=1.5)
        create_dummy_audio_file(audio_dir / "audio3.wav", duration=2.0)

        # Create audio files list
        audio_files = [
            str(audio_dir / "audio1.wav"),
            str(audio_dir / "audio2.wav"),
            str(audio_dir / "audio3.wav"),
        ]

        # Create MelSpectrogramExtractor
        mel_extractor = MelSpectrogramExtractor(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            n_mels=128,
        )

        # Create AudioDataset
        dataset = AudioDataset(
            data_dir=tmpdir,
            audio_files=audio_files,
            mel_extractor=mel_extractor,
            sample_rate=44100,
            segment_length=44100,  # 1 second segments
        )

        assert len(dataset) > 0, "Dataset should not be empty"

        # Check first sample
        sample = dataset[0]
        assert "wav" in sample, "Missing 'wav' key"
        assert "mel" in sample, "Missing 'mel' key"
        assert "text" not in sample, "AudioDataset should not have 'text'"

        print(f"[OK] Loaded {len(dataset)} samples from AudioDataset")
        print(f"[OK] Sample 0: wav shape={sample['wav'].shape}, mel shape={sample['mel'].shape}")

        return True


def main():
    """
    全テスト実行
    """
    print("=" * 60)
    print("TTSDataset Complete Test Suite")
    print("=" * 60)

    tests = [
        ("TXT Metadata Loading", test_ttsdataset_txt_metadata),
        ("JSON Metadata Loading", test_ttsdataset_json_metadata),
        ("Unicode Processing", test_unicode_processing),
        ("Collate Function (Batching)", test_collate_fn),
        ("DataLoader Integration", test_dataloader_iteration),
        ("AudioDataset (Audio-only)", test_audiodataset),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    failed_tests = total_tests - passed_tests

    for test_name, success in results:
        status = "[PASSED]" if success else "[FAILED]"
        print(f"{status:10} | {test_name}")

    print("=" * 60)
    print(f"Total: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests}")
    print("=" * 60)

    if failed_tests > 0:
        print("\n[FAILED] Some tests failed.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
