import argparse
from pathlib import Path

from trainer_dali import AdversarialTrainer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Training")
    parser.add_argument(
        "--config", type=Path, help="Path to training/testing configuration"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    AdversarialTrainer.run(args.config)
