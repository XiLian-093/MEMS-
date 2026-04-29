#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEMS SENSOR OPTIMIZATION - Python single-file version

This script merges the MATLAB workflow into one Python file:
  1) Load FG stress-difference data from pingCH.csv, or create demo data.
  2) GA1: optimize R1 layout.
  3) GA2: optimize R2 layout given fixed R1, scanning n2 = 1..n2_max.
  4) Compute Vout and save topology/stress-field figures.

Coordinate convention:
  The optimization variables and paths keep MATLAB-style 1-based (row, col)
  coordinates, so the translated fitness logic stays close to the MATLAB code.
  Only NumPy indexing converts row/col to 0-based internally.

Dependencies:
  numpy, matplotlib

Example:
  python mems_sensor_optimization.py
  python mems_sensor_optimization.py --generations 50 --population-size 80
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# Use a non-interactive backend so the script works on servers/headless systems.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INVALID_FITNESS = 1.0e9

# ===================== DATA INPUT HERE =====================
# Put your data matrix CSV filename here.
# Requirement: total 10000 values, or an existing 100 x 100 matrix.
# This matches the original MATLAB readmatrix('pingCH.csv').
DATA_MATRIX_FILE = "pingCH.csv"
# ===========================================================


@dataclass(frozen=True)
class MarginConfig:
    """Equivalent to the MATLAB margin_cfg structure."""

    margin: int = 3
    apply_top: bool = False
    apply_bottom: bool = False
    apply_left: bool = False
    apply_right: bool = False


@dataclass
class R2Result:
    n2: int
    vout_mv: float
    best_fval: float
    paths: List[np.ndarray]
    chromosome: np.ndarray


@dataclass
class GAResult:
    best_x: np.ndarray
    best_f: float
    history: List[float]


# -----------------------------------------------------------------------------
# MATLAB helper translations
# -----------------------------------------------------------------------------

def violates_margin(
    row: int,
    col_start: int,
    col_end: int,
    rows: int,
    cols: int,
    margin_cfg: MarginConfig,
) -> bool:
    """Python version of MATLAB violatesMargin(). Coordinates are 1-based."""

    m = margin_cfg.margin

    if margin_cfg.apply_top and row <= m:
        return True
    if margin_cfg.apply_bottom and row > rows - m:
        return True
    if margin_cfg.apply_left and col_start <= m:
        return True
    if margin_cfg.apply_right and col_end > cols - m:
        return True

    return False


def update_forbidden_mask(mask: np.ndarray, path: np.ndarray, distance: int) -> np.ndarray:
    """
    Python version of updateForbiddenMask().

    Parameters
    ----------
    mask:
        Boolean forbidden mask, shape = (map_rows, map_cols), NumPy 0-based.
    path:
        Mx2 array of MATLAB-style 1-based [row, col] path points.
    distance:
        Square dilation radius in grid cells.
    """

    map_rows, map_cols = mask.shape

    for r, c in path.astype(int):
        # MATLAB-style inclusive bounds in 1-based coordinates.
        r_min = max(1, int(r) - distance)
        r_max = min(map_rows, int(r) + distance)
        c_min = max(1, int(c) - distance)
        c_max = min(map_cols, int(c) + distance)

        # Convert 1-based inclusive bounds to NumPy slices.
        mask[r_min - 1 : r_max, c_min - 1 : c_max] = True

    return mask


def fg_sum_on_path(FG: np.ndarray, path: np.ndarray) -> float:
    """Sum FG values on a MATLAB-style 1-based path array."""

    rows = path[:, 0].astype(int) - 1
    cols = path[:, 1].astype(int) - 1
    return float(np.sum(FG[rows, cols]))


def path_hits_forbidden(forbidden_mask: np.ndarray, path: np.ndarray) -> bool:
    """Check whether any MATLAB-style 1-based path point lies in forbidden_mask."""

    rows = path[:, 0].astype(int) - 1
    cols = path[:, 1].astype(int) - 1
    return bool(np.any(forbidden_mask[rows, cols]))


# -----------------------------------------------------------------------------
# Fitness functions translated from MATLAB
# -----------------------------------------------------------------------------

def calculate_fitness_r1_only(
    chromosome: Sequence[float],
    FG: np.ndarray,
    n1_max: int,
    distance_in_cells: int,
    margin_cfg: MarginConfig,
) -> Tuple[float, List[np.ndarray], float, int]:
    """
    GA1 fitness: optimize only R1.

    MATLAB chromosome:
      [L1, n1, R1_1(y,x), R1_2(y,x), ..., R1_n1max(y,x)]

    Goal:
      Maximize average FG on R1, implemented as minimizing -avg_R1.
    """

    map_rows, map_cols = FG.shape
    fitness = INVALID_FITNESS
    r1_paths: List[np.ndarray] = []
    sum_fg_r1 = 0.0
    len_r1_cells = 0

    chrom = np.rint(np.asarray(chromosome, dtype=float)).astype(int)

    L1 = int(chrom[0])
    n1 = int(chrom[1])

    if n1 < 1 or n1 > n1_max:
        return fitness, r1_paths, sum_fg_r1, len_r1_cells

    L1_seg_len = int(math.floor(L1 / n1))
    if L1_seg_len <= 0:
        return fitness, r1_paths, sum_fg_r1, len_r1_cells

    starts_r1 = np.zeros((n1, 2), dtype=int)
    for i in range(n1):
        idx = 2 + 2 * i
        starts_r1[i, :] = chrom[idx : idx + 2]

    forbidden_mask = np.zeros((map_rows, map_cols), dtype=bool)

    for i in range(n1):
        row = int(starts_r1[i, 0])
        col = int(starts_r1[i, 1])
        col_end = col + L1_seg_len - 1

        if violates_margin(row, col, col_end, map_rows, map_cols, margin_cfg):
            return fitness, r1_paths, sum_fg_r1, len_r1_cells

        if row < 1 or row > map_rows or col < 1 or col_end > map_cols:
            return fitness, r1_paths, sum_fg_r1, len_r1_cells

        path_rows = np.full(L1_seg_len, row, dtype=int)
        path_cols = np.arange(col, col_end + 1, dtype=int)
        path_seg = np.column_stack((path_rows, path_cols))

        # Existing R1 segments are enforced through the forbidden mask, mirroring
        # the actual MATLAB implementation.
        if path_hits_forbidden(forbidden_mask, path_seg):
            return fitness, r1_paths, sum_fg_r1, len_r1_cells

        r1_paths.append(path_seg)
        forbidden_mask = update_forbidden_mask(forbidden_mask, path_seg, distance_in_cells)

        sum_fg_r1 += fg_sum_on_path(FG, path_seg)
        len_r1_cells += int(path_seg.shape[0])

    if len_r1_cells == 0:
        return fitness, r1_paths, sum_fg_r1, len_r1_cells

    avg_r1 = sum_fg_r1 / len_r1_cells
    fitness = -avg_r1
    return fitness, r1_paths, sum_fg_r1, len_r1_cells


def calculate_fitness_r2_given_r1(
    chromosome: Sequence[float],
    FG: np.ndarray,
    r1_paths: Sequence[np.ndarray],
    L1: int,
    n2_max: int,
    distance_in_cells: int,
    margin_cfg: MarginConfig,
) -> Tuple[float, List[np.ndarray], float, int]:
    """
    GA2 fitness: optimize R2 with R1 fixed.

    MATLAB chromosome:
      [n2, R2_1(y,x), R2_2(y,x), ..., R2_n2max(y,x)]

    Goal:
      Minimize average FG on R2.
    """

    map_rows, map_cols = FG.shape
    fitness = INVALID_FITNESS
    r2_paths: List[np.ndarray] = []
    sum_fg_r2 = 0.0
    len_r2_cells = 0

    chrom = np.rint(np.asarray(chromosome, dtype=float)).astype(int)

    n2 = int(chrom[0])
    if n2 < 1 or n2 > n2_max:
        return fitness, r2_paths, sum_fg_r2, len_r2_cells

    L1_seg_len = int(math.floor(L1 / n2))
    if L1_seg_len <= 0:
        return fitness, r2_paths, sum_fg_r2, len_r2_cells

    starts_r2 = np.zeros((n2, 2), dtype=int)
    for i in range(n2):
        idx = 1 + 2 * i
        starts_r2[i, :] = chrom[idx : idx + 2]

    forbidden_mask = np.zeros((map_rows, map_cols), dtype=bool)

    # Burn R1 into the forbidden mask first.
    for path_r1 in r1_paths:
        if path_r1 is None or len(path_r1) == 0:
            continue
        forbidden_mask = update_forbidden_mask(forbidden_mask, path_r1, distance_in_cells)

    for i in range(n2):
        row = int(starts_r2[i, 0])
        col = int(starts_r2[i, 1])
        col_end = col + L1_seg_len - 1

        if violates_margin(row, col, col_end, map_rows, map_cols, margin_cfg):
            return fitness, r2_paths, sum_fg_r2, len_r2_cells

        if row < 1 or row > map_rows or col < 1 or col_end > map_cols:
            return fitness, r2_paths, sum_fg_r2, len_r2_cells

        path_rows = np.full(L1_seg_len, row, dtype=int)
        path_cols = np.arange(col, col_end + 1, dtype=int)
        path_seg = np.column_stack((path_rows, path_cols))

        if path_hits_forbidden(forbidden_mask, path_seg):
            return fitness, r2_paths, sum_fg_r2, len_r2_cells

        r2_paths.append(path_seg)
        forbidden_mask = update_forbidden_mask(forbidden_mask, path_seg, distance_in_cells)

        sum_fg_r2 += fg_sum_on_path(FG, path_seg)
        len_r2_cells += int(path_seg.shape[0])

    if len_r2_cells == 0:
        return fitness, r2_paths, sum_fg_r2, len_r2_cells

    avg_r2 = sum_fg_r2 / len_r2_cells
    fitness = avg_r2
    return fitness, r2_paths, sum_fg_r2, len_r2_cells


# -----------------------------------------------------------------------------
# Lightweight integer genetic algorithm
# -----------------------------------------------------------------------------

def _random_population(
    rng: np.random.Generator,
    lb: np.ndarray,
    ub: np.ndarray,
    population_size: int,
) -> np.ndarray:
    """Create integer population with inclusive integer bounds."""

    pop = np.empty((population_size, lb.size), dtype=int)
    for j, (lo, hi) in enumerate(zip(lb, ub)):
        if lo == hi:
            pop[:, j] = lo
        else:
            pop[:, j] = rng.integers(lo, hi + 1, size=population_size)
    return pop


def _tournament_select(
    rng: np.random.Generator,
    fitness_values: np.ndarray,
    tournament_size: int = 3,
) -> int:
    """Return index of a tournament winner."""

    candidates = rng.integers(0, fitness_values.size, size=tournament_size)
    return int(candidates[np.argmin(fitness_values[candidates])])


def integer_ga(
    fitness_fn: Callable[[np.ndarray], float],
    lb: Sequence[int],
    ub: Sequence[int],
    *,
    population_size: int = 150,
    generations: int = 200,
    rng: np.random.Generator,
    elite_fraction: float = 0.04,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.08,
    tournament_size: int = 3,
    display: bool = True,
    label: str = "GA",
) -> GAResult:
    """
    Simple integer GA replacement for MATLAB's ga(..., IntCon=all variables).

    It is intentionally dependency-light, so the whole MATLAB project can be run
    from one Python file without PyGAD/DEAP/scipy. Results will not be identical
    to MATLAB's GA, but the optimization logic and constraints are preserved.
    """

    lb_arr = np.asarray(lb, dtype=int)
    ub_arr = np.asarray(ub, dtype=int)
    if lb_arr.shape != ub_arr.shape:
        raise ValueError("lb and ub must have the same shape")
    if np.any(lb_arr > ub_arr):
        raise ValueError("Every lower bound must be <= upper bound")

    n_vars = lb_arr.size
    elite_count = max(1, int(round(population_size * elite_fraction)))
    elite_count = min(elite_count, population_size)

    pop = _random_population(rng, lb_arr, ub_arr, population_size)
    best_x = pop[0].copy()
    best_f = float("inf")
    history: List[float] = []

    for gen in range(1, generations + 1):
        fitness_values = np.array([fitness_fn(ind) for ind in pop], dtype=float)

        gen_best_idx = int(np.argmin(fitness_values))
        gen_best_f = float(fitness_values[gen_best_idx])
        if gen_best_f < best_f:
            best_f = gen_best_f
            best_x = pop[gen_best_idx].copy()

        history.append(best_f)

        if display and (gen == 1 or gen % 10 == 0 or gen == generations):
            feasible_count = int(np.sum(fitness_values < INVALID_FITNESS))
            print(
                f"[{label}] generation {gen:4d}/{generations}: "
                f"best = {best_f:.6g}, feasible = {feasible_count}/{population_size}"
            )

        elite_indices = np.argsort(fitness_values)[:elite_count]
        new_pop = [pop[idx].copy() for idx in elite_indices]

        while len(new_pop) < population_size:
            p1 = pop[_tournament_select(rng, fitness_values, tournament_size)].copy()
            p2 = pop[_tournament_select(rng, fitness_values, tournament_size)].copy()

            if rng.random() < crossover_rate and n_vars > 1:
                # Uniform crossover keeps every gene integer.
                mask = rng.random(n_vars) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()

            # Random-reset mutation within each variable's integer bounds.
            for j in range(n_vars):
                if lb_arr[j] == ub_arr[j]:
                    child[j] = lb_arr[j]
                elif rng.random() < mutation_rate:
                    child[j] = rng.integers(lb_arr[j], ub_arr[j] + 1)

            child = np.clip(child, lb_arr, ub_arr)
            new_pop.append(child.astype(int))

        pop = np.vstack(new_pop)

    # Re-evaluate once so best_x/best_f are synchronized with final population too.
    final_fitness_values = np.array([fitness_fn(ind) for ind in pop], dtype=float)
    final_best_idx = int(np.argmin(final_fitness_values))
    final_best_f = float(final_fitness_values[final_best_idx])
    if final_best_f < best_f:
        best_f = final_best_f
        best_x = pop[final_best_idx].copy()

    return GAResult(best_x=best_x.astype(int), best_f=best_f, history=history)


# -----------------------------------------------------------------------------
# I/O and plotting
# -----------------------------------------------------------------------------

def load_fg(csv_path: Path, rng: np.random.Generator) -> np.ndarray:
    """
    Load FG data like MATLAB readmatrix('pingCH.csv') + reshape(..., [100,100]).
    If the file is missing or invalid, create a random 100x100 demo matrix.
    """

    if csv_path.exists():
        data = np.loadtxt(csv_path, delimiter=",")
        if data.size != 10000:
            raise ValueError(f"错误: 文件中的数据点总数不是 10000 个! 当前为 {data.size}")

        if data.shape == (100, 100):
            FG = np.array(data, dtype=float)
        else:
            # MATLAB reshape uses column-major order.
            FG = np.reshape(np.asarray(data, dtype=float), (100, 100), order="F")
        print(f"数据已成功加载: {csv_path}")
    else:
        print(f"警告: 无法找到数据文件 {csv_path}，创建随机矩阵用于演示。")
        FG = 1.0e6 * (rng.random((100, 100)) - 0.5)

    return FG


def save_stress_figure(
    FG: np.ndarray,
    r1_paths: Sequence[np.ndarray],
    r2_paths: Sequence[np.ndarray],
    best_n1: int,
    n2: int,
    vout_mv: float,
    output_path: Path,
    dpi: int = 600,
) -> None:
    """Save an image similar to MATLAB exportgraphics(..., Resolution=600)."""

    map_rows, map_cols = FG.shape
    fig, ax = plt.subplots(figsize=(4.72, 3.94), dpi=dpi)  # about 12 cm x 10 cm

    im = ax.imshow(
        FG,
        cmap="jet",
        origin="lower",
        extent=(0.5, map_cols + 0.5, 0.5, map_rows + 0.5),
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MPa")

    for p in r1_paths:
        if p is not None and len(p) > 0:
            ax.plot(p[:, 1], p[:, 0], "k-", linewidth=3)

    for p in r2_paths:
        if p is not None and len(p) > 0:
            ax.plot(p[:, 1], p[:, 0], "k-", linewidth=3)

    ax.set_xlim(0.5, map_cols + 0.5)
    ax.set_ylim(0.5, map_rows + 0.5)
    ax.set_aspect("equal", adjustable="box")

    # Grid styling close to the MATLAB version, but less visually dense for readability.
    ax.grid(True, which="major", linestyle="-", alpha=0.8)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")
    ax.set_xlabel("Grid column index (j)")
    ax.set_ylabel("Grid row index (i)")
    ax.set_title(f"n_1 = {best_n1}, n_2 = {n2}, V_out = {vout_mv:.2f} mV")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_summary_csv(
    output_path: Path,
    best_n1: int,
    best_L1: int,
    L1_um: int,
    best_n2: int,
    best_vout_mv: float,
    r2_results: Sequence[R2Result],
) -> None:
    """Save a compact CSV summary of the optimization result."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["best_n1", best_n1])
        writer.writerow(["best_L1_cells", best_L1])
        writer.writerow(["best_L1_um", L1_um])
        writer.writerow(["best_n2", best_n2])
        writer.writerow(["best_Vout_mV", f"{best_vout_mv:.8g}"])
        writer.writerow([])
        writer.writerow(["n2", "Vout_mV", "best_fval", "chromosome"])
        for item in r2_results:
            writer.writerow([
                item.n2,
                f"{item.vout_mv:.8g}",
                f"{item.best_fval:.8g}",
                " ".join(map(str, item.chromosome.tolist())),
            ])


# -----------------------------------------------------------------------------
# Main workflow translated from sensor.m
# -----------------------------------------------------------------------------

def run_optimization(args: argparse.Namespace) -> Dict[str, object]:
    rng = np.random.default_rng(args.seed)

    print("开始 MEMS 压敏电阻布局优化 (GA1: R1, GA2: R2)...")
    print(f"数据输入文件: {args.csv}")

    # 1. Import FG data.
    FG = load_fg(Path(args.csv), rng)
    map_rows, map_cols = FG.shape

    # 2. Global physical / geometry parameters.
    grid_size_um = args.grid_size_um
    min_distance_um = args.min_distance_um
    distance_in_cells = int(math.ceil(min_distance_um / grid_size_um))

    L1_range = (args.L1_min, args.L1_max)
    n1_max = args.n1_max
    n2_max = args.n2_max

    pi_44 = 138.1e-11
    Vin = 5.0
    MPa_to_Pa = 1.0e6

    # 3. Margin constraint switch, same defaults as MATLAB.
    margin_cfg = MarginConfig(
        margin=args.margin,
        apply_top=args.apply_top,
        apply_bottom=args.apply_bottom,
        apply_left=args.apply_left,
        apply_right=args.apply_right,
    )
    print("已设置 margin_cfg:")
    print(margin_cfg)

    # 4. GA1: optimize R1.
    print("---------------- GA1: 优化 R1 ----------------")
    n_vars1 = 1 + 1 + 2 * n1_max
    lb1 = np.array([L1_range[0], 1] + [1, 1] * n1_max, dtype=int)
    ub1 = np.array([L1_range[1], n1_max] + [map_rows, map_cols] * n1_max, dtype=int)

    def fitness1(chrom: np.ndarray) -> float:
        return calculate_fitness_r1_only(chrom, FG, n1_max, distance_in_cells, margin_cfg)[0]

    ga1_result = integer_ga(
        fitness1,
        lb1,
        ub1,
        population_size=args.population_size,
        generations=args.generations,
        rng=rng,
        display=not args.quiet,
        label="GA1/R1",
    )

    best_chrom1 = np.rint(ga1_result.best_x).astype(int)
    best_L1 = int(best_chrom1[0])
    best_n1 = int(best_chrom1[1])
    L1_um = int(best_L1 * grid_size_um)

    _, r1_paths, sum_fg_r1, len_r1_cells = calculate_fitness_r1_only(
        best_chrom1, FG, n1_max, distance_in_cells, margin_cfg
    )
    if len_r1_cells == 0 or ga1_result.best_f >= INVALID_FITNESS:
        raise RuntimeError("GA1 未找到可行 R1 布局；请增大 population/generations 或放宽约束。")

    avg_r1_pa = (sum_fg_r1 / len_r1_cells) * MPa_to_Pa

    # 5. GA2: optimize R2 while scanning n2.
    print("---------------- GA2: 优化 R2 ----------------")
    r2_results: List[R2Result] = []

    for n2 in range(1, n2_max + 1):
        print(f"\n---- 固定 n2 = {n2} ----")
        n_vars2 = 1 + 2 * n2_max
        lb2 = np.array([n2] + [1, 1] * n2_max, dtype=int)
        ub2 = np.array([n2] + [map_rows, map_cols] * n2_max, dtype=int)

        def fitness2(chrom: np.ndarray) -> float:
            return calculate_fitness_r2_given_r1(
                chrom, FG, r1_paths, best_L1, n2_max, distance_in_cells, margin_cfg
            )[0]

        ga2_result = integer_ga(
            fitness2,
            lb2,
            ub2,
            population_size=args.population_size,
            generations=args.generations,
            rng=rng,
            display=not args.quiet,
            label=f"GA2/R2 n2={n2}",
        )

        chrom2 = np.rint(ga2_result.best_x).astype(int)
        _, r2_paths_curr, sum_fg_r2, len_r2_cells = calculate_fitness_r2_given_r1(
            chrom2, FG, r1_paths, best_L1, n2_max, distance_in_cells, margin_cfg
        )

        if len_r2_cells == 0 or ga2_result.best_f >= INVALID_FITNESS:
            print(f"警告: n2={n2} 未找到可行 R2 布局，Vout 记为 -inf。")
            vout_mv = float("-inf")
        else:
            avg_r2_pa = (sum_fg_r2 / len_r2_cells) * MPa_to_Pa
            vout_v = 0.25 * pi_44 * (avg_r1_pa - avg_r2_pa) * Vin
            vout_mv = vout_v * 1.0e3

        r2_results.append(
            R2Result(
                n2=n2,
                vout_mv=float(vout_mv),
                best_fval=float(ga2_result.best_f),
                paths=r2_paths_curr,
                chromosome=chrom2,
            )
        )

    vouts_all = np.array([item.vout_mv for item in r2_results], dtype=float)
    if not np.any(np.isfinite(vouts_all)):
        raise RuntimeError("GA2 未找到任何可行 R2 布局；请增大 population/generations 或放宽约束。")

    idx_best2 = int(np.nanargmax(vouts_all))
    best_vout_mv = float(vouts_all[idx_best2])
    best_n2 = int(r2_results[idx_best2].n2)

    print("\n================ 最终优化报告 =================")
    print(f"最优配置方案: n1={best_n1}, n2={best_n2}")
    print(f"理论输出电压 Vout: {best_vout_mv:.3f} mV")
    print("================================================")
    print("\n======== 详细几何参数清单 ========")
    print("电阻 R1 (水平/主电阻):")
    print(f"   - 分段数 (n1): {best_n1} 段")
    print(f"   - 总长度 (L):  {best_L1} 个网格 ({L1_um} um)")
    print("电阻 R2 (水平/补偿电阻):")
    print(f"   - 分段数 (n2): {best_n2} 段")
    print(f"   - 总长度 (L):  {best_L1} 个网格 ({L1_um} um) [与R1保持一致]")
    print("==================================")

    output_dir = Path(args.output_dir)
    img_folder = output_dir / "Topology_Stress_Fields"

    if args.save_figures:
        for item in r2_results:
            if not np.isfinite(item.vout_mv):
                continue
            file_name = img_folder / f"Stress_Field_n2_{item.n2}_Enhanced.tif"
            save_stress_figure(
                FG,
                r1_paths,
                item.paths,
                best_n1,
                item.n2,
                item.vout_mv,
                file_name,
                dpi=args.dpi,
            )
        print(f"图像已保存到: {img_folder}")

    summary_path = output_dir / "optimization_summary.csv"
    save_summary_csv(summary_path, best_n1, best_L1, L1_um, best_n2, best_vout_mv, r2_results)
    print(f"结果摘要已保存到: {summary_path}")

    return {
        "FG": FG,
        "r1_paths": r1_paths,
        "r2_results": r2_results,
        "best_n1": best_n1,
        "best_n2": best_n2,
        "best_L1": best_L1,
        "best_vout_mv": best_vout_mv,
        "summary_path": summary_path,
        "image_folder": img_folder,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MEMS pressure sensor R1/R2 layout optimization translated from MATLAB."
    )
    parser.add_argument("--csv", default=DATA_MATRIX_FILE, help=f"Data matrix CSV path. Default: {DATA_MATRIX_FILE}")
    parser.add_argument("--output-dir", default=".", help="Directory for figures and summary CSV.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed. Default: 123")

    parser.add_argument("--grid-size-um", type=float, default=5.0, help="Grid size in micrometers.")
    parser.add_argument("--min-distance-um", type=float, default=12.0, help="Minimum distance in micrometers.")
    parser.add_argument("--L1-min", type=int, default=40, help="Minimum L1 length in grid cells.")
    parser.add_argument("--L1-max", type=int, default=100, help="Maximum L1 length in grid cells.")
    parser.add_argument("--n1-max", type=int, default=4, help="Maximum number of R1 segments.")
    parser.add_argument("--n2-max", type=int, default=4, help="Maximum number of R2 segments.")

    parser.add_argument("--margin", type=int, default=3, help="Margin width in cells.")
    parser.add_argument("--apply-top", action="store_true", help="Apply top margin constraint.")
    parser.add_argument("--apply-bottom", action="store_true", help="Apply bottom margin constraint.")
    parser.add_argument("--apply-left", action="store_true", help="Apply left margin constraint.")
    parser.add_argument("--apply-right", action="store_true", help="Apply right margin constraint.")

    parser.add_argument("--population-size", type=int, default=150, help="GA population size. Default: 150")
    parser.add_argument("--generations", type=int, default=200, help="GA generations. Default: 200")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-generation progress output.")

    parser.add_argument("--save-figures", action="store_true", default=True, help="Save topology figures. Default: true")
    parser.add_argument("--no-save-figures", dest="save_figures", action="store_false", help="Do not save figures.")
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI. Default: 600")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_optimization(args)


if __name__ == "__main__":
    main()
