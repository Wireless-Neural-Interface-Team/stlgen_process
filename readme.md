# process-stl

Measure mouse brain / skull STL geometries and derive the **probe insertion
protocol** that reaches the dLGN from above the visual cortex.

`process-stl` is a Python library with a command line (`process-stl`) and a Qt
GUI (`process-stl-gui`) on top. It merges the two former scripts of
`4_Process/`: `insertion_protocol.py` (from
`objective_functions/G01_Compute_CoM.ipynb`) and `stl_stats.py`.

## What it does

The library has two independent tools.

### 1. Insertion protocol (one geometry)

Input: one directory holding the region STLs of one geometry (whole brain,
visual cortex, dLGN, and optionally the skull).

1. **Load the regions.** Each STL is read (binary or ASCII) and sampled into a
   point cloud (50 000 area-weighted surface points by default) for display.
   The result is cached as a `.npz` so later runs skip the STL.
2. **Compute the centers of mass (CoM).** Each region gets a CoM. The visual
   cortex and dLGN, which cover both hemispheres, also get a **left
   hemisphere** CoM (`y > 0`), and only that one is used for the protocol. Two
   methods are available:
   - `volume` (default): solid CoM of the closed mesh, assuming uniform
     density. If the mesh is open it falls back to the surface centroid and
     prints a warning.
   - `points`: the mean of the sampled points, as in the original notebook.
3. **Derive the protocol.** The probe enters the brain surface (DV = 0) above
   the visual cortex, at the dLGN's ML coordinate. It is tilted in the AP-DV
   plane and pushed straight to the dLGN CoM:

   | Value | Meaning | Formula |
   |---|---|---|
   | `AP_tr` | AP translation of the entry point (mm) | `C_VS.AP` |
   | `ML_tr` | ML translation of the entry point (mm) | `C_dLGN.ML` |
   | `ML_rot` | tilt around the ML axis (rad, also `ML_rot_deg`) | `arctan((C_dLGN.AP − C_VS.AP) / C_dLGN.DV)` |
   | `Insert_tr` | insertion depth along the tilted axis (mm) | `hypot(C_dLGN.AP − C_VS.AP, C_dLGN.DV)` |

   Here `C_VS` is the left CoM of the visual cortex and `C_dLGN` is the left
   CoM of the dLGN. The protocol also stores `AP_tr_rel = AP_tr / BL`, where
   `BL` is the bregma–lambda distance of the geometry. `BL` is read from the
   `bregma_lambda_distance_skull` key of a JSON file in the region directory
   (`null` if no JSON has it). The probe tip ends at
   `entry + Insert_tr · (sin ML_rot, 0, cos ML_rot)`.
   **Optimized protocol (experimental).** Instead of aiming at the CoMs, the
   probe can be placed to maximize the electrode length inside **both** the
   visual cortex and the dLGN (`process_stl/optimize.py`):
   - the length inside a region is exact: the probe line is intersected with
     the triangles of the closed mesh;
   - the score is the shorter of the two lengths (`min`, default) or their
     product (`product`), so a trajectory that misses one region scores 0;
   - the insertion depth is not searched: for each trajectory the best depth
     is the exit of the last region crossed, within the depth bounds;
   - the other free parameters (AP, ML, rotation) are searched either with
     scipy's differential evolution, started from the current protocol, or by
     brute force (`random`: uniform random trajectories within the bounds,
     keeping the best). The result can then be refined with Nelder–Mead.

   To remove the feature, delete `process_stl/optimize.py` and
   `process_stl/gui/optimize_dialog.py`, `process_stl/gui/optimize_history.py`,
   and the lines marked `# optimize` in
   `gui/tabs/protocol_tab.py`.
4. **Use it.** A protocol can be:
   - saved as JSON, with the source files and CoMs it came from;
   - appended as one row of a CSV table;
   - loaded back and applied to **another geometry**, to measure how far its
     target lands from that geometry's dLGN CoM;
   - viewed in 3D with the craniotomy area it implies on the skull.

### 2. STL statistics (several folders)

Input: several folders of STL files, for example one per mouse strain.

For every STL the tool measures the **axis-aligned bounding box size** and the
**center of mass** on each axis. It then compares the folders two by two with
one of these tests:

- `mannwhitney` (default): Mann-Whitney U, for independent samples;
- `welch`: Welch's t-test, for independent samples that are roughly normal;
- `wilcoxon`: Wilcoxon signed-rank, for paired samples matched by file name.

The output is a table of the data, a per-folder summary (count / mean / std),
the p-values, and box plots annotated with significance stars. The data can be
saved to a CSV. That CSV is reloaded on the next run instead of being
recomputed, as long as the folders still hold the same files.

## Installation

You need Python 3.10 or later. The recommended tool is
[uv](https://docs.astral.sh/uv/):

```bash
cd 4_Process/stlgen_process
uv sync                      # library + CLI + GUI
uv sync --extra notebook     # adds interactive pyvista plots in Jupyter
```

You can use pip instead:

```bash
pip install -e .             # or: pip install -e ".[notebook]"
```

Dependencies: numpy, scipy, pandas, matplotlib, seaborn and pyvista, plus
PySide6 and pyvistaqt for the GUI. They are all installed by default. Only the
`process_stl.gui` subpackage imports Qt, so the library itself works without a
display.

With uv, prefix the commands below with `uv run`, or activate `.venv` first.

## Usage

### GUI

```bash
uv run process-stl-gui
# or:  uv run python -m process_stl

# Optional flags pre-fill the tabs:
uv run process-stl-gui --protocol-dir ../../0_stl/2_full/MuMo_25um_paper \
    --skull Calgary_Adult_Skull_Atlas.stl \
    --stats-folder C57=../sources/C57pn56 --stats-folder CD1=../sources/CD1pn56 \
    --data-file stl_stats.csv
```

**Insertion protocol tab**

1. *Source geometry*: pick the region directory. The tab pre-selects each
   region's file from its name (`whole_brain*.stl`, `*VIS*.stl`, `*LGd*.stl`,
   and a file containing "skull"). You can change any of them. The skull is
   optional and only needed for the craniotomy.
2. *Centers of mass*: choose the number of points, the CoM method and the
   cache, then click **Compute CoM**.
3. *Insertion protocol*: manage one or several protocols, each with its own
   color.
   - **Compute from CoM** fills a protocol from the CoMs.
   - **Load…** reads a protocol JSON, for example one computed on another
     skull.
   - **Optimize…** (after the CoMs are computed) opens a window. There, each
     parameter is either fixed or optimized between bounds. The default bounds
     are the skull's bounding box for the translations, −90° to 90° for the
     rotation, and 0 to the bounding box diagonal for the depth. **Run**
     shows the progress and the lengths inside each region, and **Apply to
     protocol** puts the result in the selected protocol. The search settings
     and lengths are saved in its JSON.
   - With **track cost** ticked, every trajectory tried is recorded. After the
     run, **Save tries…** writes them to a CSV (`try`, `phase`, `generation`,
     the 4 parameters, `length_VS`, `length_dLGN`, `score`). **Plot tries…**
     opens a 3D view of the trajectories, colored by try order or by score,
     with the best one in red. Next to it is a bar plot per try, with the dLGN
     length at the bottom and the VS length on top. Both show an evenly spaced
     subset of the tries (200 by default), always including the best one.
   - **AP mode** sets how the AP translation is applied to the current
     skull: `absolute` (`AP_tr`, as saved) or `relative` (`AP_tr_rel` × the
     bregma–lambda distance of the current skull). The distance is shown,
     read-only, under *Source geometry*. *Skull JSON* picks the file it is
     read from: any JSON of the directory with the key, or another file
     through **Browse…**.
   - A protocol with no relative AP translation (e.g. an older JSON) shows a
     **Compute relative** button, which sets `AP_tr_rel = AP_tr / BL` with
     the distance of the selected skull JSON.
   - You can edit the angle, the translations and the depth. The entry, the
     target and the **section** (electrode length inside the dLGN and VS,
     exact on the meshes) update live.
   - **Save…** writes the protocol as JSON, or appends it as a CSV row. The
     values saved are the ones shown, so in relative mode `AP_tr` is the AP
     translation on the current skull.
4. *Display*: an embedded 3D view.
   - Show the regions as points or as surfaces, and add any other STL of the
     directory.
   - Set the color and opacity of each component.
   - Show the left CoMs, a legend with the values of every protocol, and the
     craniotomy area on the skull.
   - Use the camera presets and take screenshots.

**STL statistics tab**

1. Add the folders to compare. Double-click a label to rename it.
2. Set the axis names and an optional CSV data file, then click **Compute
   statistics**.
3. Pick the test and the metrics to plot (all, CoM or bbox size). Changing
   them does not recompute the data.
4. Save the plot (PNG / PDF / SVG), the data table and the p-values (CSV).

### Command line

```bash
# Insertion protocol of one geometry
uv run process-stl protocol ../../0_stl/2_full/MuMo_25um_paper \
    --skull Calgary_Adult_Skull_Atlas.stl \
    --json protocol.json --csv protocols.csv --plot surface

# Compare folders of STLs ([LABEL=]FOLDER, the label defaults to the folder name)
uv run process-stl stats C57=../sources/C57pn56 CD1=../sources/CD1pn56 \
    --data-file stl_stats.csv --test mannwhitney --plot boxplots.png --kind size
```

Main options:

- `protocol`
  - `--whole-brain`, `--visual-cortex`, `--dlgn`, `--skull`: file name or
    glob of each region;
  - `--n-points`, `--com-method {volume,points}`;
  - `--cache-dir` (`none` disables the cache), `--force`;
  - `--bregma-lambda` (mm, default: read from a JSON of `src_dir`);
  - `--json`, `--csv`, `--plot {none,points,surface}`.
- `stats`
  - `--recursive`, `--axes ML,DV,AP`;
  - `--data-file`, `--force`;
  - `--test {mannwhitney,welch,wilcoxon}`;
  - `--plot FILE`, `--kind {all,com,size}`, `--show`.

Run `process-stl protocol -h` or `process-stl stats -h` for the full list.

### Python library

```python
from process_stl import load_regions, get_insertion_protocol, save_protocol, viz

# Insertion protocol
regions = load_regions("0_stl/2_full/MuMo_25um_paper",
                       skull="Calgary_Adult_Skull_Atlas.stl")
prot = get_insertion_protocol(regions, verbose=True)
save_protocol("protocol.json", prot, regions)
viz.plot_regions(regions, prot, style="surface")
viz.plot_craniotomy(regions, prot)

# Apply a saved protocol to another geometry
from process_stl.protocol import load_protocol, target_offset
other = load_regions("0_stl/2_full/another_geometry")
print(target_offset(load_protocol("protocol.json"), other))  # (AP, ML, DV) mm

# Statistics between folders
from process_stl import load_or_compute, compute_pvalues, summary
from process_stl.plotting import plot_boxplots

folders = {"C57": "sources/C57pn56", "CD1": "sources/CD1pn56"}
df = load_or_compute(folders, data_file="stl_stats.csv")
print(summary(df))
pvals = compute_pvalues(df, list(folders), "mannwhitney")
plot_boxplots(df, list(folders), pvals, "mannwhitney").savefig("boxplots.png")
```

## Conventions

- **Region STLs (protocol)**: x → AP (posterior > 0), y → ML (left > 0),
  z → DV (ventral > 0). Units are mm, with the origin at bregma.
- **STL statistics**: by default the axis names are x → ML, y → DV, z → AP,
  as in the skull STLs. Change them with `--axes` or in the GUI.
- **Protocol JSON**: holds the `protocol` dict (`ML_rot`, `ML_rot_deg`,
  `ML_tr`, `AP_tr`, `AP_tr_rel`, `Insert_tr`), plus its origin, settings,
  `ap_mode`, `bregma_lambda_distance`, source files and CoMs. A bare protocol
  dict also loads. Older files without `AP_tr_rel` load too, but only in
  absolute mode.
- **Protocol CSV**: the columns are `insertion_angle` (°), `MD_translation`,
  `AP_translation` and `insertion_depth` (mm). The file is created with its
  header if missing; otherwise a row is appended.
- **Points cache**: stored at `<src_dir>/_points/<stem>_<n_points>_<method>.npz`
  and reused while the STL's size and modification time are unchanged.

## Package layout

| Module | Content |
|---|---|
| `mesh.py` | STL reading and geometry: `load_stl`, `is_closed`, `bbox`, `mesh_com`, `mesh_stats`, `sample_surface`, `find_stl_files` |
| `regions.py` | region files of one geometry → cached point clouds and CoMs (`load_regions`, `Region`) |
| `protocol.py` | insertion protocol, entry / target points, craniotomy mask, JSON and CSV import / export |
| `sections.py` | exact electrode length inside the visual cortex and dLGN meshes |
| `optimize.py` | experimental: protocol maximizing the electrode length inside the visual cortex and the dLGN |
| `stats.py` | bbox / CoM table of several folders (with CSV reload), summary, p-values |
| `plotting.py` | matplotlib / seaborn box plots of the statistics |
| `viz.py` | pyvista 3D scenes, usable in any plotter including the GUI's |
| `cli.py` | `process-stl` command line |
| `gui/` | `process-stl-gui` (PySide6): window, background workers, widgets, one tab per tool |

## License

MIT, see [LICENSE](LICENSE).
