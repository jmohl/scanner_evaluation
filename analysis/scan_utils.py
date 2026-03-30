"""Utilities for loading eval logs, scan results, and validation data into DataFrames."""

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def load_eval_logs(
    eval_logs_dir: str | Path,
    label_segment: str | None = "synth",
    score_key: str | None = None,
    success_fn: Callable | None = None,
    exclude_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Load transcript metadata from ``.eval`` log files.

    Walks *eval_logs_dir* recursively for ``.eval`` ZIP archives, reads each
    ``summaries.json`` inside, and extracts one row per transcript.

    Can be pointed at either a top-level eval-logs directory (e.g.
    ``swe_bench/eval-logs/``) or a specific subdirectory (e.g.
    ``swe_bench/eval-logs/synth``).  Files found under a *label_segment*
    subdirectory get their label from the next path component; files
    outside it are labelled ``"default"``.

    Parameters
    ----------
    eval_logs_dir:
        Directory containing ``.eval`` files (searched recursively).
    label_segment:
        If set, the eval-label is derived from the first path component
        after this segment in the file path (e.g. files under
        ``synth/oh1-obv/`` produce label ``oh1-obv``).  Files not under
        this segment get label ``"default"``.  When *None*, no label
        column is added.
    score_key:
        When the scorer returns a dict, extract this key as the scalar
        score value (e.g. ``"above_median"`` for mle-bench).  When *None*,
        the raw value is used directly.
    success_fn:
        Custom function ``(score_value) -> bool`` to determine success.
        When *None*, :func:`_is_success` is used.
    exclude_patterns:
        List of substrings; ``.eval`` files whose path contains any of
        these strings are skipped (e.g. ``["%2B", "broken"]``).

    Returns
    -------
    pd.DataFrame
        One row per transcript with columns ``transcript_id``, ``task_id``,
        ``transcript_score``, ``transcript_success``, ``eval_file``, and
        optionally ``eval_label``.
    """
    eval_logs_dir = Path(eval_logs_dir)
    exclude_patterns = exclude_patterns or ["%2B"]
    eval_files = sorted(eval_logs_dir.rglob("*.eval"))
    eval_files = [
        f for f in eval_files
        if not any(pat in str(f) for pat in exclude_patterns)
    ]
    if not eval_files:
        raise FileNotFoundError(f"No .eval files found under {eval_logs_dir}")

    check_success = success_fn or _is_success

    rows: list[dict] = []
    for eval_file in eval_files:
        label = _label_from_path(eval_file, label_segment) if label_segment else None
        with zipfile.ZipFile(eval_file) as zf:
            summaries = json.loads(zf.read("summaries.json"))
            for s in summaries:
                scores = s.get("scores", {})
                # Take the first scorer's value
                score_val = None
                if scores:
                    first_score = next(iter(scores.values()))
                    score_val = first_score.get("value")
                if score_key is not None and isinstance(score_val, dict):
                    score_val = score_val.get(score_key)
                row = {
                    "transcript_id": s["uuid"],
                    "task_id": s["id"],
                    "transcript_score": score_val,
                    "transcript_success": check_success(score_val),
                    "eval_file": str(eval_file.relative_to(eval_logs_dir)),
                }
                if label is not None:
                    row["eval_label"] = label
                rows.append(row)

    return pd.DataFrame(rows)


def load_scan_results(scan_results_dir: str | Path) -> pd.DataFrame:
    """Load scanner output parquet files from scan-result directories.

    Recursively finds every ``scan_id=*`` subdirectory under
    *scan_results_dir* and concatenates the parquet files found within.
    Can be pointed at either a top-level scan-results directory or a
    specific subdirectory.

    Parameters
    ----------
    scan_results_dir:
        Path to a scan-results directory (e.g.
        ``eval_grading/swe_bench/scan-results`` or
        ``eval_grading/swe_bench/scan-results/synth``).

    Returns
    -------
    pd.DataFrame
        One row per (transcript, scanner) result from the parquet files,
        with an added ``value_num`` column (``value`` cast to float).
    """
    scan_results_dir = Path(scan_results_dir)
    scan_dirs = sorted(scan_results_dir.rglob("scan_id=*"))
    if not scan_dirs:
        raise FileNotFoundError(
            f"No scan_id=* directories found under {scan_results_dir}"
        )

    frames: list[pd.DataFrame] = []
    for scan_dir in scan_dirs:
        for pq_path in sorted(scan_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(pq_path))

    combined = pd.concat(frames, ignore_index=True)
    combined["value_num"] = pd.to_numeric(combined["value"], errors="coerce")
    return combined


def load_validations(
    validation_dir: str | Path,
    prefix: str = "swe_bench_",
) -> pd.DataFrame:
    """Load validation CSVs and return a single DataFrame keyed by transcript ID.

    Each CSV is expected to have columns ``id`` and ``target``.  CSVs that
    lack these columns are silently skipped.  The filename (minus ``.csv``)
    is used to derive the column name by stripping *prefix*.

    Parameters
    ----------
    validation_dir:
        Directory (searched recursively) containing validation CSV files.
    prefix:
        Leading portion of each filename to strip when building column names.

    Returns
    -------
    pd.DataFrame
        Indexed by ``transcript_id`` with one column per validation CSV.
    """
    validation_dir = Path(validation_dir)
    csv_paths = sorted(validation_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {validation_dir}")

    result: pd.DataFrame | None = None
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "id" not in df.columns or "target" not in df.columns:
            continue
        col_name = csv_path.stem
        if col_name.startswith(prefix):
            col_name = col_name[len(prefix) :]
        val = df[["id", "target"]].rename(
            columns={"id": "transcript_id", "target": col_name}
        )
        if result is None:
            result = val
        else:
            result = result.merge(val, on="transcript_id", how="outer")
    if result is None:
        raise FileNotFoundError(
            f"No CSV files with 'id' and 'target' columns found under {validation_dir}"
        )
    return result


def build_summary(
    logs: pd.DataFrame,
    scans: pd.DataFrame | None = None,
    validation_dir: str | Path | None = None,
    validation_prefix: str = "swe_bench_",
) -> pd.DataFrame:
    """Build a one-row-per-transcript summary from eval logs, scans, and validations.

    Parameters
    ----------
    logs:
        DataFrame from :func:`load_eval_logs`.
    scans:
        DataFrame from :func:`load_scan_results`.  When provided, scanner
        values are pivoted into columns and left-joined onto *logs*.
    validation_dir:
        Optional path to a directory of validation CSVs (see
        :func:`load_validations`).
    validation_prefix:
        Prefix to strip from validation CSV filenames.

    Returns
    -------
    pd.DataFrame
        One row per transcript with eval-log metadata, scanner value columns
        (if *scans* given), and validation columns (if *validation_dir* given).
    """
    summary = logs.copy()

    if scans is not None:
        # Deduplicate: first value per (transcript, scanner)
        agg = (
            scans.groupby(["transcript_id", "scanner_key"])
            .agg(value=("value_num", "first"))
            .reset_index()
        )
        pivot = agg.pivot_table(
            index="transcript_id",
            columns="scanner_key",
            values="value",
        ).reset_index()
        pivot.columns.name = None
        summary = summary.merge(pivot, on="transcript_id", how="left")

    if validation_dir is not None:
        validations = load_validations(validation_dir, prefix=validation_prefix)
        summary = summary.merge(validations, on="transcript_id", how="left")

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _label_from_path(eval_file: Path, segment: str) -> str:
    """Extract an eval label from a file path based on a directory segment.

    Given segment ``"synth"`` and path ``.../synth/oh1-obv/foo.eval``,
    returns ``"oh1-obv"``.  Files not under the segment (e.g. at the
    top level) return ``"default"``.
    """
    parts = str(eval_file).split(f"/{segment}/")
    if len(parts) > 1:
        # e.g. .../synth/oh1-obv/foo.eval -> "oh1-obv"
        sub = parts[1].split("/")
        if len(sub) > 1:
            return sub[0]
        # File is directly inside the segment dir (no subdirectory)
        return "default"
    return "default"


def _is_success(score_value) -> bool:
    """Determine whether a score value represents success."""
    if score_value is None:
        return False
    if isinstance(score_value, str):
        return score_value == "C"
    return float(score_value) > 0
