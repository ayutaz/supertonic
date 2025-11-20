"""
Duration Predictor Training Script

発話レベルの長さ予測モデルの学習
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

from models.dp import DurationPredictor
from models.autoencoder import SpeechAutoencoder
from trainers.dp_trainer import DPTrainer
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


def create_dp_model(config: dict, vocab_size: int) -> DurationPredictor:
    """
    Duration Predictorモデル作成

    引数:
        config: tts.jsonの設定
        vocab_size: 語彙サイズ

    戻り値:
        DurationPredictor
    """
    dp_config = config["dp"]

    # Sentence Encoder設定
    sentence_encoder_config = {
        "vocab_size": vocab_size,
        "char_emb_dim": dp_config["sentence_encoder"]["text_embedder"]["char_emb_dim"],
        "convnext_idim": dp_config["sentence_encoder"]["convnext"]["idim"],
        "convnext_ksz": dp_config["sentence_encoder"]["convnext"]["ksz"],
        "convnext_intermediate_dim": dp_config["sentence_encoder"]["convnext"]["intermediate_dim"],
        "convnext_num_layers": dp_config["sentence_encoder"]["convnext"]["num_layers"],
        "convnext_dilation_lst": dp_config["sentence_encoder"]["convnext"]["dilation_lst"],
        "attn_hidden_channels": dp_config["sentence_encoder"]["attn_encoder"]["hidden_channels"],
        "attn_filter_channels": dp_config["sentence_encoder"]["attn_encoder"]["filter_channels"],
        "attn_n_heads": dp_config["sentence_encoder"]["attn_encoder"]["n_heads"],
        "attn_n_layers": dp_config["sentence_encoder"]["attn_encoder"]["n_layers"],
        "attn_p_dropout": dp_config["sentence_encoder"]["attn_encoder"]["p_dropout"],
        "proj_out_idim": dp_config["sentence_encoder"]["proj_out"]["idim"],
        "proj_out_odim": dp_config["sentence_encoder"]["proj_out"]["odim"],
    }

    # Style Encoder設定
    style_encoder_config = {
        "ldim": dp_config["latent_dim"],
        "chunk_compress_factor": dp_config["chunk_compress_factor"],
        "proj_in_odim": dp_config["style_encoder"]["proj_in"]["odim"],
        "convnext_idim": dp_config["style_encoder"]["convnext"]["idim"],
        "convnext_ksz": dp_config["style_encoder"]["convnext"]["ksz"],
        "convnext_intermediate_dim": dp_config["style_encoder"]["convnext"]["intermediate_dim"],
        "convnext_num_layers": dp_config["style_encoder"]["convnext"]["num_layers"],
        "convnext_dilation_lst": dp_config["style_encoder"]["convnext"]["dilation_lst"],
        "style_input_dim": dp_config["style_encoder"]["style_token_layer"]["input_dim"],
        "n_style": dp_config["style_encoder"]["style_token_layer"]["n_style"],
        "style_key_dim": dp_config["style_encoder"]["style_token_layer"]["style_key_dim"],
        "style_value_dim": dp_config["style_encoder"]["style_token_layer"]["style_value_dim"],
        "prototype_dim": dp_config["style_encoder"]["style_token_layer"]["prototype_dim"],
        "n_units": dp_config["style_encoder"]["style_token_layer"]["n_units"],
        "n_heads": dp_config["style_encoder"]["style_token_layer"]["n_heads"],
    }

    # Predictor設定
    predictor_config = {
        "sentence_dim": dp_config["predictor"]["sentence_dim"],
        "style_dim": dp_config["predictor"]["style_dim"],
        "hdim": dp_config["predictor"]["hdim"],
        "n_layer": dp_config["predictor"]["n_layer"],
    }

    # DurationPredictorモデル作成
    model = DurationPredictor(
        sentence_encoder_config=sentence_encoder_config,
        style_encoder_config=style_encoder_config,
        predictor_config=predictor_config,
    )

    return model


def create_autoencoder(config: dict) -> SpeechAutoencoder:
    """
    Autoencoderモデル作成（Frozen）

    引数:
        config: tts.jsonの設定

    戻り値:
        SpeechAutoencoder
    """
    ae_config = config["ae"]

    # Encoder設定
    encoder_config = {
        "n_fft": ae_config["encoder"]["spec_processor"]["n_fft"],
        "win_length": ae_config["encoder"]["spec_processor"]["win_length"],
        "hop_length": ae_config["encoder"]["spec_processor"]["hop_length"],
        "n_mels": ae_config["encoder"]["spec_processor"]["n_mels"],
        "sample_rate": ae_config["encoder"]["spec_processor"]["sample_rate"],
        "eps": ae_config["encoder"]["spec_processor"]["eps"],
        "norm_mean": ae_config["encoder"]["spec_processor"]["norm_mean"],
        "norm_std": ae_config["encoder"]["spec_processor"]["norm_std"],
        "ksz_init": ae_config["encoder"]["ksz_init"],
        "ksz": ae_config["encoder"]["ksz"],
        "num_layers": ae_config["encoder"]["num_layers"],
        "dilation_lst": ae_config["encoder"]["dilation_lst"],
        "intermediate_dim": ae_config["encoder"]["intermediate_dim"],
        "idim": ae_config["encoder"]["idim"],
        "hdim": ae_config["encoder"]["hdim"],
        "odim": ae_config["encoder"]["odim"],
    }

    # Decoder設定
    decoder_config = {
        "ksz_init": ae_config["decoder"]["ksz_init"],
        "ksz": ae_config["decoder"]["ksz"],
        "num_layers": ae_config["decoder"]["num_layers"],
        "dilation_lst": ae_config["decoder"]["dilation_lst"],
        "intermediate_dim": ae_config["decoder"]["intermediate_dim"],
        "idim": ae_config["decoder"]["idim"],
        "hdim": ae_config["decoder"]["hdim"],
        "head_idim": ae_config["decoder"]["head"]["idim"],
        "head_hdim": ae_config["decoder"]["head"]["hdim"],
        "head_odim": ae_config["decoder"]["head"]["odim"],
        "head_ksz": ae_config["decoder"]["head"]["ksz"],
        "sample_rate": ae_config["sample_rate"],
        "hop_length": ae_config["encoder"]["spec_processor"]["hop_length"],
    }

    # Autoencoderモデル作成
    model = SpeechAutoencoder(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )

    return model


def train(args):
    """
    学習メイン関数
    """
    # 1. 設定読み込み
    config = load_config(args.config)
    sample_rate = config["ae"]["sample_rate"]

    # 2. デバイス設定
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    # 3. データセット作成（ダミー）
    # 実際にはデータセットクラスを実装する必要がある
    print("Creating datasets...")
    # train_dataset = TTSDataset(...)
    # val_dataset = TTSDataset(...)

    # ダミーデータローダー（実装例）
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. モデル作成
    print("Creating models...")
    vocab_size = args.vocab_size  # Unicode indexerから取得

    dp_model = create_dp_model(config, vocab_size)
    autoencoder = create_autoencoder(config)

    # Autoencoderのチェックポイント読み込み（必要に応じて）
    if args.autoencoder_checkpoint:
        print(f"Loading autoencoder checkpoint: {args.autoencoder_checkpoint}")
        ae_checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
        autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])

    # 5. オプティマイザー・スケジューラー
    optimizer = torch.optim.AdamW(
        dp_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6,
    )

    # 6. Trainer作成
    trainer = DPTrainer(
        dp_model=dp_model,
        autoencoder=autoencoder,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        log_dir=args.log_dir,
        sample_rate=sample_rate,
    )

    # チェックポイント読み込み（必要に応じて）
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # 7. 学習ループ
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Training
        # for batch_idx, batch in enumerate(train_loader):
        #     metrics = trainer.train_step(batch)
        #
        #     if batch_idx % args.log_interval == 0:
        #         print(f"  Step {batch_idx}: Loss={metrics['mse_loss']:.4f}")

        # Validation
        # if (epoch + 1) % args.val_interval == 0:
        #     val_metrics = []
        #     for batch in val_loader:
        #         metrics = trainer.validation_step(batch)
        #         val_metrics.append(metrics)
        #
        #     avg_val_loss = sum([m["mse_loss"] for m in val_metrics]) / len(val_metrics)
        #     print(f"  Validation Loss: {avg_val_loss:.4f}")

        # Checkpoint保存
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"{args.checkpoint_dir}/dp_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(checkpoint_path)

    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Duration Predictor Training")

    # Paths
    parser.add_argument(
        "--config",
        type=str,
        default="../../assets/onnx/tts.json",
        help="Path to config file (tts.json)",
    )
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained autoencoder checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to resume checkpoint",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/dp",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/dp",
        help="Directory for TensorBoard logs",
    )

    # Training
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval (steps)")
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval (epochs)")

    # Model
    parser.add_argument("--vocab_size", type=int, default=1114112, help="Vocabulary size (Unicode)")

    # Device
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
