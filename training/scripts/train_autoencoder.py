"""
Speech Autoencoder Training Script

学習メインスクリプト
"""

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import SpeechAutoencoder, MultiScaleDiscriminator
from data import AudioDataset, collate_fn
from data.preprocessing import MelSpectrogramExtractor
from trainers import AutoencoderTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Speech Autoencoder")

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/autoencoder.yaml",
        help="Path to config file",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to data directory",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to metadata file(s)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_metadata(metadata_paths: list) -> list:
    """メタデータを読み込み"""
    audio_files = []

    for metadata_path in metadata_paths:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 1:
                    audio_files.append(parts[0])

    return audio_files


def main():
    args = parse_args()

    print("=" * 60)
    print("Speech Autoencoder Training")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    print(f"\nConfig loaded from: {args.config}")

    # Load metadata
    audio_files = load_metadata(args.metadata)
    print(f"Total audio files: {len(audio_files)}")

    # Model config
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    # Create mel extractor
    encoder_config = model_config.get("encoder", {})
    spec_config = encoder_config.get("spec_processor", {})

    mel_extractor = MelSpectrogramExtractor(
        sample_rate=spec_config.get("sample_rate", 44100),
        n_fft=spec_config.get("n_fft", 2048),
        win_length=spec_config.get("win_length", 2048),
        hop_length=spec_config.get("hop_length", 512),
        n_mels=spec_config.get("n_mels", 228),
        eps=spec_config.get("eps", 1e-5),
        norm_mean=spec_config.get("norm_mean", 0.0),
        norm_std=spec_config.get("norm_std", 1.0),
    )

    # Create datasets
    segment_length = data_config.get("segment_length", None)
    if segment_length is not None:
        # Convert seconds to samples
        segment_length = int(segment_length * spec_config.get("sample_rate", 44100))

    full_dataset = AudioDataset(
        data_dir=args.data_dir,
        audio_files=audio_files,
        mel_extractor=mel_extractor,
        sample_rate=spec_config.get("sample_rate", 44100),
        segment_length=segment_length,
        normalize=True,
        trim_silence=False,
    )

    # Train/Val split
    val_ratio = data_config.get("val_ratio", 0.05)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(training_config.get("seed", 42)),
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create data loaders
    batch_size = training_config.get("batch_size", 64)
    num_workers = data_config.get("num_workers", 8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get("val_batch_size", 8),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Create model
    print("\nCreating model...")
    model = SpeechAutoencoder.from_config(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create discriminator
    discriminator = MultiScaleDiscriminator(
        in_channels=1,
        channels=64,
        num_scales=3,
    )

    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {disc_params:,}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = AutoencoderTrainer(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        trainer.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        trainer.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        trainer.scaler_g.load_state_dict(checkpoint["scaler_g"])
        trainer.scaler_d.load_state_dict(checkpoint["scaler_d"])
        trainer.global_step = checkpoint["global_step"]
        trainer.epoch = checkpoint["epoch"]

        if "scheduler_g" in checkpoint and trainer.scheduler_g is not None:
            trainer.scheduler_g.load_state_dict(checkpoint["scheduler_g"])
            trainer.scheduler_d.load_state_dict(checkpoint["scheduler_d"])

        print(f"Resumed from step {trainer.global_step}, epoch {trainer.epoch}")

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()


if __name__ == "__main__":
    main()
