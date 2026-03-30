"""
train.py — Main Training Entry Point
=====================================
Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/bridge_best.pt
    python train.py --config config.yaml --overrides training.learning_rate=5e-5
"""

import argparse
import logging
import sys
import yaml

from trainer import Trainer


def override_cfg(cfg: dict, overrides: list):
    """Apply key=value overrides to nested config dict (dot-separated keys)."""
    for kv in overrides:
        key, _, val = kv.partition("=")
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        # Attempt numeric cast
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        node[keys[-1]] = val
        print(f"  Override: {key} = {val}")


def main():
    parser = argparse.ArgumentParser(description="Train Mimi-to-HuBERT Bridge")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--resume",   default=None, help="Checkpoint path to resume from")
    parser.add_argument("--overrides", nargs="*", default=[],
                        help="Config overrides: key=value e.g. training.batch_size=8")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("train.log"),
        ],
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        override_cfg(cfg, args.overrides)

    trainer = Trainer(cfg)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
