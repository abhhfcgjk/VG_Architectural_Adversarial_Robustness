import argparse
from pathlib import Path

from trainer_dali import AdversarialTrainer
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Training")
    parser.add_argument(
        "--config", type=Path, help="Path to training/testing configuration"
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    AdversarialTrainer.run(args.config)
