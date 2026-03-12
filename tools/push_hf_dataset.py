"""Push data to the abc-scout-scanners HuggingFace dataset."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

DATASET_REPO_ID = "arcadia-mars-4-0/abc-scout-scanners"


def push_dataset(
    eval_logs_dir: Path,
    scan_results_dir: Path,
    scanner_name: str,
    dataset_subdir: str = "xstest",
    token: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Push local directories to HuggingFace under one dataset subdirectory."""
    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / dataset_subdir
        staging_eval_logs = staging / "eval-logs"
        staging_scan_results = staging / "scan-results" / scanner_name

        staging_eval_logs.mkdir(parents=True)
        staging_scan_results.mkdir(parents=True)

        if eval_logs_dir.exists():
            shutil.copytree(eval_logs_dir, staging_eval_logs, dirs_exist_ok=True)
            logger.info(f"Staged eval-logs from {eval_logs_dir}")
        else:
            logger.warning(f"eval-logs directory not found: {eval_logs_dir}")

        if scan_results_dir.exists():
            shutil.copytree(scan_results_dir, staging_scan_results, dirs_exist_ok=True)
            logger.info(f"Staged scan-results from {scan_results_dir} → scan-results/{scanner_name}/")
        else:
            logger.warning(f"scan-results directory not found: {scan_results_dir}")

        destination = f"{dataset_subdir}/"
        logger.info(f"Uploading to {DATASET_REPO_ID} under {destination}")
        api.upload_folder(
            folder_path=tmp,
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            commit_message=commit_message
            or f"Update {dataset_subdir}/ eval-logs and scan-results",
        )

    logger.info("Upload complete.")


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=f"Push data to {DATASET_REPO_ID} under a dataset subdirectory."
    )
    parser.add_argument(
        "--eval-logs-dir",
        type=Path,
        required=True,
        help="Local directory containing eval log files",
    )
    parser.add_argument(
        "--scan-results-dir",
        type=Path,
        required=True,
        help="Local directory containing scan result files",
    )
    parser.add_argument(
        "--scanner-name",
        type=str,
        required=True,
        help="Scanner name used as subfolder inside scan-results/ (e.g. dummy-scanner)",
    )
    parser.add_argument(
        "--dataset-subdir",
        type=str,
        default="xstest",
        help="Dataset subdirectory to write under (default: xstest)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: $HF_TOKEN)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        help="Commit message for the HuggingFace push",
    )
    args = parser.parse_args()

    push_dataset(
        eval_logs_dir=args.eval_logs_dir,
        scan_results_dir=args.scan_results_dir,
        scanner_name=args.scanner_name,
        dataset_subdir=args.dataset_subdir,
        token=args.token,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
