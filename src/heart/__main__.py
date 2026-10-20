"""
CLI entry point for the heart package.
Usage:
    python -m heart data
    python -m heart train
    python -m heart evaluate
    python -m heart all
"""
import argparse

from heart.data import clean_data, download_raw_data
from heart.train import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="MLOps Heart Disease Pipeline CLI")
    parser.add_argument("command", choices=["data", "train", "all"], help="Command to run")
    parser.add_argument("--online", action="store_true", help="Force online download from UCI repository")

    args = parser.parse_args()

    if args.command == "data":
        print("--- Ingesting and Cleaning Data ---")
        download_raw_data(force_online=args.online)
        clean_data()
    elif args.command == "train":
        print("--- Training Models with Cross-Validation and MLflow ---")
        train_and_evaluate(online_download=args.online)
    elif args.command == "all":
        print("--- Running Complete Pipeline ---")
        download_raw_data(force_online=args.online)
        clean_data()
        train_and_evaluate(online_download=args.online)

if __name__ == "__main__":
    main()
