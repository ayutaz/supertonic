"""
TTL (Text-to-Latent) Training Script

Flow Matchingによるテキストから潜在表現への変換学習
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ttl import TTLModel
from models.autoencoder import SpeechAutoencoder
from losses import TTLLoss
from trainers.ttl_trainer import TTLTrainer
from data import TTSDataset


def load_config(config_path: str) -> dict:
    """
    設定ファイル読み込み

    引数:
        config_path: 設定ファイルパス (tts.json)

    戻り値:
        設定辞書
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_ttl_model(config: dict, vocab_size: int) -> TTLModel:
    """
    TTLモデル作成

    引数:
        config: tts.jsonの設定
        vocab_size: 語彙サイズ

    戻り値:
        TTLModel
    """
    ttl_config = config["ttl"]

    # Text Encoder設定
    text_encoder_config = {
        "char_emb_dim": ttl_config["text_encoder"]["text_embedder"]["char_emb_dim"],
        "convnext_idim": ttl_config["text_encoder"]["convnext"]["idim"],
        "convnext_ksz": ttl_config["text_encoder"]["convnext"]["ksz"],
        "convnext_intermediate_dim": ttl_config["text_encoder"]["convnext"]["intermediate_dim"],
        "convnext_num_layers": ttl_config["text_encoder"]["convnext"]["num_layers"],
        "convnext_dilation_lst": ttl_config["text_encoder"]["convnext"]["dilation_lst"],
        "attn_hidden_channels": ttl_config["text_encoder"]["attn_encoder"]["hidden_channels"],
        "attn_filter_channels": ttl_config["text_encoder"]["attn_encoder"]["filter_channels"],
        "attn_n_heads": ttl_config["text_encoder"]["attn_encoder"]["n_heads"],
        "attn_n_layers": ttl_config["text_encoder"]["attn_encoder"]["n_layers"],
        "attn_p_dropout": ttl_config["text_encoder"]["attn_encoder"]["p_dropout"],
        "proj_out_idim": ttl_config["text_encoder"]["proj_out"]["idim"],
        "proj_out_odim": ttl_config["text_encoder"]["proj_out"]["odim"],
    }

    # Style Encoder設定
    style_encoder_config = {
        "ldim": ttl_config["latent_dim"],
        "chunk_compress_factor": ttl_config["chunk_compress_factor"],
        "proj_in_odim": ttl_config["style_encoder"]["proj_in"]["odim"],
        "convnext_idim": ttl_config["style_encoder"]["convnext"]["idim"],
        "convnext_ksz": ttl_config["style_encoder"]["convnext"]["ksz"],
        "convnext_intermediate_dim": ttl_config["style_encoder"]["convnext"]["intermediate_dim"],
        "convnext_num_layers": ttl_config["style_encoder"]["convnext"]["num_layers"],
        "convnext_dilation_lst": ttl_config["style_encoder"]["convnext"]["dilation_lst"],
        "style_input_dim": ttl_config["style_encoder"]["style_token_layer"]["input_dim"],
        "n_style": ttl_config["style_encoder"]["style_token_layer"]["n_style"],
        "style_key_dim": ttl_config["style_encoder"]["style_token_layer"]["style_key_dim"],
        "style_value_dim": ttl_config["style_encoder"]["style_token_layer"]["style_value_dim"],
        "prototype_dim": ttl_config["style_encoder"]["style_token_layer"]["prototype_dim"],
        "n_units": ttl_config["style_encoder"]["style_token_layer"]["n_units"],
        "n_heads": ttl_config["style_encoder"]["style_token_layer"]["n_heads"],
    }

    # Vector Field設定
    vector_field_config = {
        "ldim": ttl_config["latent_dim"],
        "chunk_compress_factor": ttl_config["chunk_compress_factor"],
        "proj_in_odim": ttl_config["vector_field"]["proj_in"]["odim"],
        "time_dim": ttl_config["vector_field"]["time_encoder"]["time_dim"],
        "time_hdim": ttl_config["vector_field"]["time_encoder"]["hdim"],
        "n_blocks": ttl_config["vector_field"]["main_blocks"]["n_blocks"],
        "idim": ttl_config["vector_field"]["main_blocks"]["time_cond_layer"]["idim"],
        "style_dim": ttl_config["style_encoder"]["style_token_layer"]["style_value_dim"],
        "text_dim": ttl_config["text_encoder"]["proj_out"]["odim"],
        "n_heads": ttl_config["vector_field"]["main_blocks"]["text_cond_layer"]["n_heads"],
        "use_residual": ttl_config["vector_field"]["main_blocks"]["text_cond_layer"]["use_residual"],
        "rotary_base": ttl_config["vector_field"]["main_blocks"]["text_cond_layer"]["rotary_base"],
        "rotary_scale": ttl_config["vector_field"]["main_blocks"]["text_cond_layer"]["rotary_scale"],
        "ksz": ttl_config["vector_field"]["main_blocks"]["convnext_0"]["ksz"],
        "intermediate_dim": ttl_config["vector_field"]["main_blocks"]["convnext_0"]["intermediate_dim"],
        "last_convnext_num_layers": ttl_config["vector_field"]["last_convnext"]["num_layers"],
        "last_convnext_dilation_lst": ttl_config["vector_field"]["last_convnext"]["dilation_lst"],
    }

    # TTLModel作成
    model = TTLModel(
        vocab_size=vocab_size,
        text_encoder_config=text_encoder_config,
        style_encoder_config=style_encoder_config,
        vector_field_config=vector_field_config,
    )

    return model


def create_autoencoder(config: dict) -> SpeechAutoencoder:
    """
    Speech Autoencoder作成（Frozen）

    引数:
        config: tts.jsonの設定

    戻り値:
        SpeechAutoencoder
    """
    ae_config = config["ae"]

    autoencoder = SpeechAutoencoder(
        sample_rate=ae_config["sample_rate"],
        n_fft=ae_config["encoder"]["spec_processor"]["n_fft"],
        hop_length=ae_config["encoder"]["spec_processor"]["hop_length"],
        n_mels=ae_config["encoder"]["spec_processor"]["n_mels"],
        encoder_idim=ae_config["encoder"]["idim"],
        encoder_hdim=ae_config["encoder"]["hdim"],
        encoder_odim=ae_config["encoder"]["odim"],
        encoder_num_layers=ae_config["encoder"]["num_layers"],
        encoder_dilation_lst=ae_config["encoder"]["dilation_lst"],
        decoder_hdim=ae_config["decoder"]["hdim"],
        decoder_num_layers=ae_config["decoder"]["num_layers"],
        decoder_dilation_lst=ae_config["decoder"]["dilation_lst"],
        head_hdim=ae_config["decoder"]["head"]["hdim"],
        vocoder_n_iter=32,  # Vocoderは使用しないのでデフォルト値
    )

    return autoencoder


def main(args):
    """
    メイン関数
    """
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 設定読み込み
    config = load_config(args.config)
    print(f"[INFO] Loaded config from: {args.config}")

    # 語彙サイズの設定（unicode_indexer.jsonから取得する場合は別途実装）
    vocab_size = args.vocab_size
    print(f"[INFO] Vocabulary size: {vocab_size}")

    # TTLモデル作成
    ttl_model = create_ttl_model(config, vocab_size)
    print(f"[INFO] Created TTL model")
    print(f"[INFO] TTL model parameters: {sum(p.numel() for p in ttl_model.parameters()):,}")

    # Autoencoder作成
    autoencoder = create_autoencoder(config)
    print(f"[INFO] Created Autoencoder")

    # Autoencoderチェックポイント読み込み
    if args.autoencoder_checkpoint:
        checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
        autoencoder.load_state_dict(checkpoint["model"])
        print(f"[INFO] Loaded Autoencoder checkpoint: {args.autoencoder_checkpoint}")

    # 損失関数
    loss_fn = TTLLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        ttl_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Trainer
    trainer = TTLTrainer(
        ttl_model=ttl_model,
        autoencoder=autoencoder,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # チェックポイント読み込み（再開）
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # データセット（ダミー実装、実際のデータセットに置き換える）
    print("[WARNING] Using dummy dataset. Please implement actual dataset.")
    # train_dataset = TTSDataset(...)
    # val_dataset = TTSDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 学習ループ（ダミー実装）
    print("[INFO] Training loop not implemented. Please add dataset and training loop.")
    print("[INFO] Example:")
    print("  for epoch in range(args.num_epochs):")
    print("      for batch in train_loader:")
    print("          metrics = trainer.train_step(batch)")
    print("      if epoch % args.val_interval == 0:")
    print("          val_metrics = trainer.validate(val_loader)")

    # 最終チェックポイント保存
    # trainer.save_checkpoint("final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TTL (Text-to-Latent) model")

    # Model
    parser.add_argument("--config", type=str, required=True, help="Path to tts.json")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=str,
        default=None,
        help="Path to autoencoder checkpoint",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")

    # Logging
    parser.add_argument("--log-dir", type=str, default="logs/ttl", help="TensorBoard log directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/ttl",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Checkpoint saving interval",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Validation
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval (epochs)")

    args = parser.parse_args()

    main(args)
