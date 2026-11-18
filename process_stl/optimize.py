"""Optimized insertion protocol: the probe trajectory that maximizes the
electrode length inside both the visual cortex and the dLGN.

Experimental and self-contained: the rest of the library does not import this
module, so it can be removed with process_stl/gui/optimize_dialog.py and the
lines marked "optimize" in gui/tabs/protocol_tab.py.

The length inside a region is exact (see sections.py).

The problem is not linear (the lengths are piecewise smooth, with plateaus
where the probe misses a region), so the search is global:
- the insertion depth is not searched: the lengths only grow with the depth,
  so the best depth is found exactly for each trajectory (the exit of the last
  region crossed, within the depth bounds);
- the remaining free parameters (at most AP, ML, rotation) are searched with
  scipy's differential evolution, seeded with a start protocol (e.g. the CoM
  one), or by brute force (uniform random trajectories within the bounds,
  keeping the best), then refined locally with Nelder-Mead.

Objectives (score to maximize, lengths in mm):
- "min": the shorter of the two lengths, so both regions must be crossed;
- "product": L_VS * L_dLGN, zero as soon as one region is missed.
With track=True every trajectory tried is recorded (history_frame /
save_history turn it into a table).

A small TIE * (L_VS + L_dLGN) breaks ties between equal scores, and when a
region is missed a small GUIDE * (distance to its CoM) leads the search to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .protocol import make_protocol
from .sections import SHORT, TARGETS, _length, line, load_meshes, protocol_lengths  # noqa: F401

#: parameters of a protocol, in the order of the search vector
PARAMS = ("AP_tr", "ML_tr", "ML_rot_deg", "Insert_tr")
OBJECTIVES = {
    "min": "shorter of the two lengths (balanced)",
    "product": "product of the two lengths",
}
OBJECTIVE = "min"
METHODS = {
    "evolution": "differential evolution (scipy)",
    "random": "brute force: random trajectories, keep the best",
}
METHOD = "evolution"
RANDOM_BATCH = 500  # random tries between two progress reports
TIE = 0.01     # weight of the total length, to break ties
GUIDE = 1e-3   # weight of the distance to a missed region


def _distance_to_segment(point, origin, direction, length):
    t = np.clip((point - origin) @ direction, 0.0, length)
    return float(np.linalg.norm(origin + t * direction - point))


def _distance_to_com(mesh, origin, direction, length):
    """Distance from the segment to the region CoM (or its mirror across ML = 0)."""
    if mesh.com is None:
        return 0.0
    mirror = mesh.com * np.array([1.0, -1.0, 1.0])
    return min(_distance_to_segment(c, origin, direction, length) for c in (mesh.com, mirror))


def evaluate(values, meshes, depth_bounds, objective=OBJECTIVE) -> dict:
    """Lengths and score of one trajectory.

    values: AP_tr, ML_tr, ML_rot_deg and Insert_tr; Insert_tr None takes the
    best depth within depth_bounds.
    meshes: {"visual_cortex": RegionMesh, "dLGN": RegionMesh}
    """
    origin, direction = line(values["AP_tr"], values["ML_tr"], values["ML_rot_deg"])
    inside = {k: meshes[k].intervals(origin, direction) for k in TARGETS}
    dmin, dmax = depth_bounds
    depth = values.get("Insert_tr")
    if depth is None:  # the lengths only grow with the depth: stop after the last region
        ends = [min(i1, dmax) for iv in inside.values() for i0, i1 in iv
                if i1 > 0 and i0 < dmax]
        depth = max([dmin, *ends])
    lengths = {k: _length(iv, 0.0, depth) for k, iv in inside.items()}

    a, b = lengths.values()
    main = min(a, b) if objective == "min" else a * b
    score = main + TIE * (a + b)
    for k in TARGETS:
        if lengths[k] == 0:
            score -= GUIDE * _distance_to_com(meshes[k], origin, direction, dmax)
    return {"values": {**values, "Insert_tr": float(depth)}, "lengths": lengths,
            "score": float(score)}


def default_bounds(regions) -> dict:
    """Search bounds: AP / ML over the bounding box of the skull (or of the
    whole brain without a skull), rotation over [-90, 90] deg, depth from 0 to
    the bounding box diagonal."""
    ref = regions.get("skull") or regions["whole_brain"]
    pts = np.asarray(ref.pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return {"AP_tr": (0, float(hi[0])),
            "ML_tr": (0, float(hi[1])),
            "ML_rot_deg": (-70.0, 0.0),
            "Insert_tr": (0.0, float(np.linalg.norm(hi - lo)))}


@dataclass
class OptimizeSettings:
    fixed: dict             # parameter -> fixed value
    bounds: dict            # parameter -> (lo, hi), for the free parameters
    objective: str = OBJECTIVE
    start: dict | None = None  # protocol to seed the search with
    maxiter: int = 200
    popsize: int = 30
    seed: int | None = 0
    refine: bool = True
    track: bool = False     # record every trajectory tried (result["history"])
    method: str = METHOD    # key of METHODS
    n_samples: int = 20_000  # tries of the random search


#: columns of the history of an optimization
HISTORY_COLUMNS = ["try", "phase", "generation", *PARAMS, "length_VS", "length_dLGN",
                   "score"]


def history_frame(history):
    """pandas DataFrame of the trajectories tried (HISTORY_COLUMNS)."""
    import pandas as pd
    return pd.DataFrame(history, columns=HISTORY_COLUMNS)


def save_history(path, history):
    history_frame(history).to_csv(path, index=False)


def _random_search(cost, bounds, x0, settings, values_of, meshes, depth_bounds, state,
                   progress, should_stop):
    """Brute force: n_samples uniform tries within the bounds (the start
    protocol first, if given), keeping the best. Returns (x, cost)."""
    rng = np.random.default_rng(settings.seed)
    lo, hi = np.array(bounds, dtype=float).T
    best_x, best_f = None, np.inf
    if x0 is not None:
        best_x, best_f = np.asarray(x0, dtype=float), cost(x0)
    done = 0
    while done < settings.n_samples:
        batch = min(RANDOM_BATCH, settings.n_samples - done)
        for x in rng.uniform(lo, hi, size=(batch, len(bounds))):
            f = cost(x)
            if f < best_f:
                best_x, best_f = x, f
        done += batch
        state["gen"] += 1
        if progress is not None:
            progress(state["gen"], evaluate(values_of(best_x), meshes, depth_bounds,
                                            settings.objective))
        if should_stop and should_stop():
            break
    return best_x, best_f


def optimize_protocol(meshes, settings: OptimizeSettings, bregma_lambda=None,
                      progress=None, should_stop=None, log=print) -> dict:
    """Search the protocol maximizing the electrode length inside both regions.

    progress(generation, best_evaluation) is called after each generation
    (each RANDOM_BATCH tries for the random search);
    should_stop() returning True ends the search early. Returns a dict with
    "protocol", "lengths", "score", "n_eval", "time_s" and "settings".
    """
    if settings.method not in METHODS:
        raise ValueError(f"unknown method {settings.method!r}, choose from {list(METHODS)}")
    if settings.objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {settings.objective!r}, choose from "
                         f"{list(OBJECTIVES)}")
    depth_free = "Insert_tr" not in settings.fixed
    depth_bounds = (settings.bounds["Insert_tr"] if depth_free
                    else (settings.fixed["Insert_tr"],) * 2)
    free = [k for k in PARAMS if k not in settings.fixed and k != "Insert_tr"]
    bounds = [tuple(sorted(settings.bounds[k])) for k in free]
    n_eval = 0
    history = []
    state = {"gen": 0, "phase": settings.method}

    def values_of(x):
        values = {k: float(settings.fixed[k]) for k in settings.fixed}
        values.update({k: float(v) for k, v in zip(free, x)})
        if depth_free:
            values["Insert_tr"] = None
        return values

    def cost(x):
        nonlocal n_eval
        n_eval += 1
        ev = evaluate(values_of(x), meshes, depth_bounds, settings.objective)
        if settings.track:
            v, ln = ev["values"], ev["lengths"]
            history.append([n_eval, state["phase"], state["gen"], *(v[k] for k in PARAMS),
                            ln["visual_cortex"], ln["dLGN"], ev["score"]])
        return -ev["score"]

    t0 = time.perf_counter()
    if free:
        x0 = None
        if settings.start is not None:
            x0 = [float(np.clip(settings.start[k], lo, hi)) for k, (lo, hi) in zip(free, bounds)]

        def callback(intermediate_result):
            state["gen"] += 1
            if progress is not None:
                progress(state["gen"], evaluate(values_of(intermediate_result.x), meshes,
                                                depth_bounds, settings.objective))
            return bool(should_stop and should_stop())

        if settings.method == "random":
            log(f"  random search over {', '.join(free)}: {settings.n_samples} tries "
                f"({settings.objective} objective)...")
            x, fun = _random_search(cost, bounds, x0, settings, values_of, meshes,
                                    depth_bounds, state, progress, should_stop)
        else:
            log(f"  differential evolution over {', '.join(free)} "
                f"({settings.objective} objective)...")
            res = differential_evolution(cost, bounds, x0=x0, maxiter=settings.maxiter,
                                         popsize=settings.popsize, seed=settings.seed,
                                         polish=False, callback=callback, tol=1e-8)
            x, fun = res.x, res.fun
        if settings.refine and not (should_stop and should_stop()):
            log("  local refinement (Nelder-Mead)...")
            state["phase"] = "refine"
            loc = minimize(cost, x, method="Nelder-Mead", bounds=bounds,
                           options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
            if loc.fun <= fun:
                x = loc.x
    else:
        x = []
        state["phase"] = "fixed"
        cost(x)
    best = evaluate(values_of(x), meshes, depth_bounds, settings.objective)

    v = best["values"]
    prot = make_protocol(v["ML_tr"], v["AP_tr"], v["Insert_tr"], ML_rot_deg=v["ML_rot_deg"],
                         AP_tr_rel=v["AP_tr"] / bregma_lambda if bregma_lambda else None)
    return {"protocol": prot, "lengths": best["lengths"], "score": best["score"],
            "n_eval": n_eval, "time_s": time.perf_counter() - t0,
            "history": history if settings.track else None,
            "settings": {"fixed": settings.fixed,
                         "bounds": {k: list(settings.bounds[k]) for k in free}
                         | ({"Insert_tr": list(depth_bounds)} if depth_free else {}),
                         "objective": settings.objective, "method": settings.method,
                         **({"n_samples": settings.n_samples}
                            if settings.method == "random" else
                            {"maxiter": settings.maxiter, "popsize": settings.popsize}),
                         "seed": settings.seed,
                         "refine": settings.refine}}
