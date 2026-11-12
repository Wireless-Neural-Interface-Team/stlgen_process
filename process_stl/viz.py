"""PyVista scenes of the regions and the insertion protocol.

The add_* functions draw into any pyvista plotter (a pv.Plotter, or the
pyvistaqt.QtInteractor of the GUI) and return their legend entries, which
decorate() turns into the legend; plot_regions / plot_craniotomy open a
standalone window, as in G01_Compute_CoM.ipynb.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from .protocol import entry_point, in_craniotomy_area, protocol_lines, target_point

OPACITY = {"skull": 0.2, "whole_brain": 0.1}  # other single regions: 0.05
DOUBLE_OPACITY = 0.05
POINT_SIZE = 5
PART_COLOR = "#D4A373"  # extra parts of the directory
LEGEND_FONT_SIZE = 12   # pixels
TRAJECTORY_COLOR = "red"
#: colors given to the successive protocols of the GUI
PROTOCOL_COLORS = ["red", "darkorange", "royalblue", "green", "magenta",
                   "darkcyan", "saddlebrown", "purple"]

#: camera presets: name -> (vector from the focal point to the camera, view up)
#: (x -> AP, posterior > 0; y -> ML, left > 0; z -> DV, ventral > 0)
VIEWS = {
    "lateral (left)": ((0, 1, 0), (0, 0, -1)),
    "dorsal": ((0, 0, -1), (-1, 0, 0)),
    "anterior": ((-1, 0, 0), (0, 0, -1)),
    "posterior": ((1, 0, 0), (0, 0, -1)),
}


def default_opacity(region) -> float:
    if region.label in OPACITY:
        return OPACITY[region.label]
    return DOUBLE_OPACITY if region.is_double else 0.05


def read_mesh(path) -> pv.PolyData:
    return pv.read(str(path))


def _opacity_kwargs(op) -> dict:
    """None leaves the opacity unset: the actor stays opaque with full shading,
    unlike opacity=1 which is rendered with the translucent pipeline."""
    return {} if op is None else {"opacity": op}


def add_regions(plotter, regions, opacity=None, style="points", meshes=None,
                show_com=True, point_size=POINT_SIZE, colors=None) -> list:
    """Draw the regions and the left centers of mass of the double regions.

    opacity: label -> opacity (overrides default_opacity); 0 hides the region,
    None draws it without any opacity setting.
    style: "points" draws the sampled points, "surface" the STL meshes
    (meshes: label -> pv.PolyData already read, else read here).
    colors: label -> color (overrides the region color).
    """
    opacity = opacity or {}
    colors = colors or {}
    legend = []
    for label, reg in regions.items():
        op = opacity[label] if label in opacity else default_opacity(reg)
        color = colors.get(label, reg.color)
        if op is None or op > 0:
            if style == "surface":
                mesh = (meshes or {}).get(label) or read_mesh(reg.path)
                plotter.add_mesh(mesh, color=color, **_opacity_kwargs(op))
            else:
                plotter.add_points(np.asarray(reg.pts), color=color,
                                   point_size=point_size, render_points_as_spheres=True,
                                   **_opacity_kwargs(op))
        if show_com and reg.is_double:
            plotter.add_points(np.asarray(reg.lcom)[None], color=color,
                               point_size=15, render_points_as_spheres=True)
            legend.append([f"C_{label}: {np.round(reg.lcom, 2)}", color, "circle"])
    return legend


def add_parts(plotter, meshes, color=PART_COLOR, opacity=0.3, style="points",
              point_size=POINT_SIZE):
    """Extra STL meshes, shown for context only (name -> pv.PolyData).

    style "points" draws the mesh vertices. opacity as in add_regions.
    """
    if opacity is not None and opacity <= 0:
        return
    for mesh in meshes.values():
        if style == "surface":
            plotter.add_mesh(mesh, color=color, **_opacity_kwargs(opacity))
        else:
            plotter.add_points(np.asarray(mesh.points), color=color,
                               point_size=point_size, render_points_as_spheres=True,
                               **_opacity_kwargs(opacity))


def add_trajectory(plotter, prot, color=TRAJECTORY_COLOR, title="Insertion protocol",
                   extra_lines=()) -> list:
    """Insertion line from the entry point to the target.

    The legend lists the whole protocol under `title`, then `extra_lines`.
    """
    traj = np.array([entry_point(prot), target_point(prot)])
    plotter.add_lines(traj, color=color, width=4)
    plotter.add_points(traj, color=color, point_size=15, render_points_as_spheres=True)
    return ([[title, color, "rectangle"]]
            + [[f"    {line}", color, "none"]
               for line in [*protocol_lines(prot), *extra_lines]])


def add_craniotomy(plotter, skull, prot, thickness=1, radius=1, opacity=0.5,
                   point_size=POINT_SIZE, color=TRAJECTORY_COLOR) -> int:
    """Highlight the skull points inside the craniotomy area. Returns their count."""
    mask = in_craniotomy_area(np.asarray(skull.pts), prot, thickness, radius)
    if mask.any():
        plotter.add_points(np.asarray(skull.pts)[mask], color=color,
                           opacity=opacity, point_size=point_size,
                           render_points_as_spheres=True)
    return int(mask.sum())


def legend_size(plotter, legend, font_size=LEGEND_FONT_SIZE) -> tuple[float, float]:
    """Legend box (fraction of the window) giving text about font_size pixels
    high: VTK scales the legend text to fill its box."""
    width, height = plotter.window_size
    n_chars = max(len(entry[0]) for entry in legend)
    return (min((n_chars * 0.55 * font_size + 3 * font_size) / max(width, 1), .95),
            min(len(legend) * 1.45 * font_size / max(height, 1), .95))


def decorate(plotter, view="lateral (left)", legend=None, font_size=LEGEND_FONT_SIZE):
    """Axes, camera, and the legend built from the entries returned by add_*."""
    if legend:
        plotter.add_legend(labels=legend, size=legend_size(plotter, legend, font_size),
                           border=True, loc="upper left", bcolor="white")
    plotter.add_axes(xlabel="AP", ylabel="ML", zlabel="DV")
    set_view(plotter, view)


def set_view(plotter, view):
    vector, viewup = VIEWS[view]
    plotter.view_vector(vector, viewup=viewup)


# ---------------------------------------------------------------------------
# Standalone windows
# ---------------------------------------------------------------------------
def plot_regions(regions, prot=None, opacity=None, style="points",
                 window_size=(1500, 1000), show=True):
    """Regions, left centers of mass and insertion trajectory."""
    plotter = pv.Plotter()
    legend = add_regions(plotter, regions, opacity, style)
    if prot is not None:
        legend += add_trajectory(plotter, prot)
    decorate(plotter, legend=legend)
    if show:
        plotter.show(window_size=window_size)
    return plotter


def plot_craniotomy(regions, prot, thickness=1, radius=1,
                    window_size=(1500, 1000), show=True):
    """Skull points with the craniotomy area in red (top view)."""
    skull = regions.get("skull")
    if skull is None:
        raise ValueError("no skull loaded (pass skull=... to load_regions)")

    plotter = pv.Plotter()
    plotter.add_points(np.asarray(skull.pts), color=skull.color, opacity=.05,
                       render_points_as_spheres=True)
    if not add_craniotomy(plotter, skull, prot, thickness, radius):
        print("warning: no skull point in the craniotomy area")
    decorate(plotter, view="dorsal")
    if show:
        plotter.show(window_size=window_size)
    return plotter
