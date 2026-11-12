"""Compare the bounding box size and center of mass of the STL files of several folders.

The result is a pandas DataFrame with columns
    label, folder, fname, <ax>_com x3, <ax>_size x3
where the axis names come from the axis convention (default STL x -> ML,
y -> DV, z -> AP, as for the skull STLs). It can be saved to a CSV and reloaded
instead of recomputed, as long as it still matches the folders and the STL
files found on disk.

The center of mass assumes a closed (watertight) mesh with uniform density;
open meshes fall back to the area-weighted surface centroid.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .mesh import find_stl_files, load_stl, mesh_stats

AXES = ("ML", "DV", "AP")  # STL x, y, z

TEST_NAMES = {
    "mannwhitney": "Mann-Whitney U test",
    "welch": "Welch's t-test",
    "wilcoxon": "Wilcoxon signed-rank test (paired by fname)",
}
# mannwhitney : independent samples, no normality assumption
# welch       : independent samples, assumes ~normal data
# wilcoxon    : PAIRED samples matched by fname (folders contain the same parts)
STAT_TEST = "mannwhitney"


def metrics(axes=AXES) -> list[str]:
    return [f"{a}_com" for a in axes] + [f"{a}_size" for a in axes]


def columns(axes=AXES) -> list[str]:
    return ["label", "folder", "fname", *metrics(axes)]


# ---------------------------------------------------------------------------
# Data collection (with save / reload)
# ---------------------------------------------------------------------------
def collect_stats(folders, recursive=False, axes=AXES, log=print) -> pd.DataFrame:
    """folders: label -> folder path."""
    com_cols, size_cols = metrics(axes)[:3], metrics(axes)[3:]
    rows = []
    for label, folder in folders.items():
        files = find_stl_files(folder, recursive)
        log(f"[{label}] {len(files)} STL file(s) in {folder}")

        for path in files:
            fname = str(path.relative_to(folder))
            try:
                com, size, solid = mesh_stats(load_stl(path))
            except Exception as e:
                # Keep the file as a row of NaNs: it is excluded from plots and
                # stats, but the saved data still matches the folder contents
                log(f"  could not read {fname}: {e}")
                com = size = [np.nan] * 3
            else:
                if not solid:
                    log(f"  warning: {fname} is not closed, using surface centroid")

            rows.append({"label": label, "folder": str(folder), "fname": fname,
                         **dict(zip(com_cols, com)), **dict(zip(size_cols, size))})

    return pd.DataFrame(rows, columns=columns(axes))


def saved_data_problem(df, folders, recursive=False, axes=AXES, log=print):
    """Return why saved data can't be reused, or None if it matches the settings."""
    if list(df.columns) != columns(axes):
        return "columns differ (other axis convention?)"
    extra = set(df["label"]) - set(folders)
    if extra:
        return f"labels not in folders: {sorted(extra)}"

    for label, folder in folders.items():
        saved = set(df.loc[(df["label"] == label) & (df["folder"] == str(folder)), "fname"])
        if Path(folder).is_dir():
            on_disk = {str(p.relative_to(folder)) for p in find_stl_files(folder, recursive)}
            if saved != on_disk:
                return (f"{label}: {len(on_disk - saved)} new / "
                        f"{len(saved - on_disk)} removed file(s)")
        elif saved:
            log(f"  note: {folder} not found, using saved data for {label}")
        else:
            return f"no saved data for {label} and {folder} not found"
    return None


def load_or_compute(folders, recursive=False, data_file=None, force=False,
                    axes=AXES, log=print) -> pd.DataFrame:
    """Reload data_file if it still matches folders, else compute (and save) it."""
    if data_file and Path(data_file).is_file() and not force:
        df = pd.read_csv(data_file, dtype={"label": str, "folder": str, "fname": str})
        problem = saved_data_problem(df, folders, recursive, axes, log)
        if problem is None:
            log(f"Loaded {len(df)} rows from {data_file} (no recomputation)")
            return df
        log(f"{data_file} is out of date ({problem}), recomputing...")

    df = collect_stats(folders, recursive, axes, log)
    if data_file and not df.empty:
        Path(data_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_file, index=False)
        log(f"Data saved to {data_file}")
    return df


def summary(df, axes=AXES) -> pd.DataFrame:
    """count / mean / std of every metric, per label."""
    return df.groupby("label")[metrics(axes)].agg(["count", "mean", "std"]).T


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def pvalue(df, col, label_a, label_b, test=STAT_TEST) -> float:
    """Two-sided p-value comparing `col` between two labels (NaN if not computable)."""
    if test not in TEST_NAMES:
        raise ValueError(f"unknown test {test!r}, choose from {list(TEST_NAMES)}")
    try:
        if test == "wilcoxon":
            a = df[df["label"] == label_a].set_index("fname")[col].dropna()
            b = df[df["label"] == label_b].set_index("fname")[col].dropna()
            common = a.index.intersection(b.index)
            if len(common) < 2:
                return np.nan
            return stats.wilcoxon(a[common], b[common]).pvalue

        a = df.loc[df["label"] == label_a, col].dropna()
        b = df.loc[df["label"] == label_b, col].dropna()
        if len(a) < 2 or len(b) < 2:
            return np.nan
        if test == "mannwhitney":
            return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        return stats.ttest_ind(a, b, equal_var=False).pvalue
    except ValueError:  # e.g. all paired differences are zero
        return np.nan


def compute_pvalues(df, order, test=STAT_TEST, axes=AXES) -> pd.DataFrame:
    """Table of p-values for every metric and every pair of labels."""
    rows = [{"metric": col, "group1": a, "group2": b,
             "p_value": pvalue(df, col, a, b, test)}
            for col in metrics(axes) for a, b in combinations(order, 2)]
    return pd.DataFrame(rows, columns=["metric", "group1", "group2", "p_value"])


def format_p(p) -> str:
    if np.isnan(p):
        return "p = n/a"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    text = "p < 0.001" if p < 0.001 else f"p = {p:.3g}"
    return f"{text} ({stars})"
