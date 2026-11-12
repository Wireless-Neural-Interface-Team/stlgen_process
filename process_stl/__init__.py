"""process-stl: measure STL geometries and derive the dLGN insertion protocol.

Typical use::

    from process_stl import load_regions, get_insertion_protocol
    from process_stl import viz

    regions = load_regions("0_stl/2_full/MuMo_25um_paper",
                           skull="Calgary_Adult_Skull_Atlas.stl")
    prot = get_insertion_protocol(regions, verbose=True)
    viz.plot_regions(regions, prot)

    from process_stl import load_or_compute, compute_pvalues
    from process_stl.plotting import plot_boxplots

    folders = {"C57": "sources/C57pn56", "CD1": "sources/CD1pn56"}
    df = load_or_compute(folders, data_file="stl_stats.csv")
    pvals = compute_pvalues(df, list(folders), "mannwhitney")
    plot_boxplots(df, list(folders), pvals, "mannwhitney")
"""

from .mesh import (
    bbox,
    find_stl_files,
    is_closed,
    load_stl,
    mesh_com,
    mesh_stats,
    sample_surface,
)
from .protocol import (
    entry_point,
    format_protocol,
    get_insertion_protocol,
    in_craniotomy_area,
    protocol_report,
    save_protocol,
    target_point,
)
from .regions import (
    CACHE_DIR,
    COM_METHOD,
    COM_METHODS,
    N_POINTS,
    REGIONS,
    Region,
    find_file,
    load_region,
    load_regions,
)
from .stats import (
    AXES,
    STAT_TEST,
    TEST_NAMES,
    collect_stats,
    compute_pvalues,
    load_or_compute,
    metrics,
    pvalue,
    summary,
)

__version__ = "0.1.0"

__all__ = [
    "load_stl", "is_closed", "bbox", "mesh_com", "mesh_stats", "sample_surface",
    "find_stl_files",
    "REGIONS", "N_POINTS", "CACHE_DIR", "COM_METHOD", "COM_METHODS",
    "Region", "find_file", "load_region", "load_regions",
    "get_insertion_protocol", "format_protocol", "entry_point", "target_point",
    "in_craniotomy_area", "protocol_report", "save_protocol",
    "AXES", "STAT_TEST", "TEST_NAMES", "metrics", "collect_stats",
    "load_or_compute", "summary", "pvalue", "compute_pvalues",
]
