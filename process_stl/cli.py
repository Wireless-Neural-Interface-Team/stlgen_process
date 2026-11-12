"""Command-line entry point: ``process-stl {protocol,stats} ...``"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .regions import CACHE_DIR, COM_METHOD, COM_METHODS, N_POINTS, REGIONS, load_regions
from .stats import AXES, STAT_TEST, TEST_NAMES


def _parse_folders(items):
    """["C57=sources/C57pn56", "sources/CD1pn56"] -> {"C57": ..., "CD1pn56": ...}"""
    folders = {}
    for item in items:
        label, sep, folder = item.partition("=")
        if not sep:
            folder = item
            label = folder.rstrip("/").rsplit("/", 1)[-1]
        folders[label] = folder
    return folders


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="process-stl",
        description="Measure STL geometries and derive the dLGN insertion protocol.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    prot = sub.add_parser("protocol", help="insertion protocol from one region directory")
    prot.add_argument("src_dir", help="directory containing the region STL files")
    prot.add_argument("--whole-brain", help=f"default: {REGIONS['whole_brain']['pattern']}")
    prot.add_argument("--visual-cortex", help=f"default: {REGIONS['visual_cortex']['pattern']}")
    prot.add_argument("--dlgn", help=f"default: {REGIONS['dLGN']['pattern']}")
    prot.add_argument("--skull", help="skull STL (optional, needed for the craniotomy plot)")
    prot.add_argument("--n-points", type=int, default=N_POINTS)
    prot.add_argument("--com-method", choices=list(COM_METHODS), default=COM_METHOD)
    prot.add_argument("--cache-dir", default=CACHE_DIR,
                      help="relative to src_dir; 'none' disables the cache")
    prot.add_argument("--force", action="store_true", help="ignore the cache")
    prot.add_argument("--json", help="save the protocol to this JSON file")
    prot.add_argument("--csv", help="add the protocol as a row of this CSV (created if missing)")
    prot.add_argument("--plot", choices=["none", "points", "surface"], default="none")

    st = sub.add_parser("stats", help="compare bbox size / center of mass between folders")
    st.add_argument("folders", nargs="+", metavar="[LABEL=]FOLDER",
                    help="folders of STL files (label defaults to the folder name)")
    st.add_argument("--recursive", action="store_true", help="also search subfolders")
    st.add_argument("--axes", default=",".join(AXES),
                    help=f"names of the STL x,y,z axes (default: {','.join(AXES)})")
    st.add_argument("--data-file", help="CSV reloaded if it still matches the folders")
    st.add_argument("--force", action="store_true", help="recompute even if --data-file matches")
    st.add_argument("--test", choices=list(TEST_NAMES), default=STAT_TEST)
    st.add_argument("--plot", help="save the box plots to this image")
    st.add_argument("--kind", choices=["all", "com", "size"], default="all")
    st.add_argument("--show", action="store_true", help="open the plot window")
    return p


def run_protocol(args):
    from .protocol import (
        append_protocol_csv,
        format_protocol,
        get_insertion_protocol,
        save_protocol,
    )

    regions = load_regions(
        args.src_dir, whole_brain=args.whole_brain, visual_cortex=args.visual_cortex,
        dlgn=args.dlgn, skull=args.skull, n_points=args.n_points,
        com_method=args.com_method,
        cache_dir=None if args.cache_dir.lower() == "none" else args.cache_dir,
        force=args.force)

    for label, reg in regions.items():
        lcom = f", left CoM {np.round(reg.lcom, 3)}" if reg.is_double else ""
        print(f"{label:>14}: {reg.path.name}, CoM {np.round(reg.com, 3)}{lcom}")
    prot = get_insertion_protocol(regions)
    print(format_protocol(prot))

    if args.json:
        save_protocol(args.json, prot, regions, origin="center of mass",
                      settings={"n_points": args.n_points, "com_method": args.com_method})
        print(f"Protocol saved to {args.json}")
    if args.csv:
        created = append_protocol_csv(args.csv, prot)
        print(f"{'Created' if created else 'Added a row to'} {args.csv}")

    if args.plot != "none":
        from . import viz
        viz.plot_regions(regions, prot, style=args.plot)
        if "skull" in regions:
            viz.plot_craniotomy(regions, prot)


def run_stats(args):
    from .stats import compute_pvalues, load_or_compute, summary

    folders = _parse_folders(args.folders)
    axes = tuple(a.strip() for a in args.axes.split(","))
    if len(axes) != 3:
        raise SystemExit("--axes needs 3 comma-separated names")

    df = load_or_compute(folders, args.recursive, args.data_file, args.force, axes)
    if df.empty:
        print("No data collected, check the folders.")
        return
    print("\nSummary per label:")
    print(summary(df, axes).round(4).to_string())

    order = list(folders)
    pvals = compute_pvalues(df, order, args.test, axes)
    print(f"\np-values ({TEST_NAMES[args.test]}):")
    print(pvals.to_string(index=False, float_format=lambda p: f"{p:.4g}"))

    if args.plot or args.show:
        import matplotlib.pyplot as plt

        from .plotting import plot_boxplots
        fig = plot_boxplots(df, order, pvals, args.test, kind=args.kind, axes=axes)
        if args.plot:
            fig.savefig(args.plot, dpi=150)
            print(f"Plot saved to {args.plot}")
        if args.show:
            plt.show()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    {"protocol": run_protocol, "stats": run_stats}[args.command](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
