"""Pull data from the abc-scout-scanners HuggingFace dataset."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

DATASET_REPO_ID = "arcadia-mars-4-0/abc-scout-scanners"
DEFAULT_OUTPUT_DIR = Path("evals")


def pull_dataset(
    output_dir: Path,
    token: str | None = None,
    allow_patterns: list[str] | None = None,
) -> None:
    """Download the abc-scout-scanners dataset to a local directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {DATASET_REPO_ID} → {output_dir}")

    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=str(output_dir),
        token=token,
        allow_patterns=allow_patterns,
    )

    logger.info(f"Download complete: {output_dir}")


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=f"Pull data from the {DATASET_REPO_ID} HuggingFace dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local directory to download into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for private datasets (default: $HF_TOKEN)",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Optional file pattern to limit what is downloaded; can be repeated",
    )
    args = parser.parse_args()

    pull_dataset(args.output_dir, args.token, args.allow_patterns)


if __name__ == "__main__":
    main()
