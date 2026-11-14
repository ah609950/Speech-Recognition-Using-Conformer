"""Student 4 entry point for single-recording inference."""
import argparse
from conformer_asr.config import load_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", help="Path to a recording")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--check", action="store_true", help="Load configs and exit")
    args = parser.parse_args()
    configs = load_all(args.config_dir)
    if args.check:
        print("Configuration files loaded:", ", ".join(configs))
        return
    parser.exit(2, "Starter only: implement single-recording inference before running this command.\n")


if __name__ == "__main__":
    main()
