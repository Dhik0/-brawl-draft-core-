#!/usr/bin/env python3
"""Offline hierarchical-L2 3v3 logistic matchup model.

Rust sources and the SQLite database are inputs only. SQLite is always opened
with a read-only URI and all rows are read inside one consistent transaction.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit

MIN_TIER = 16
BAYES_PRIOR_WEIGHT = 50.0
MATCHUP_PRIOR_K = 20.0
DECAY_HALF_LIFE_DAYS = 30.0
SERIES_WINDOW_SECONDS = 600
SAME_MODE_WEIGHT = 0.6
SAME_MODE_CAP = 40.0
OTHER_MODE_WEIGHT = 0.3
OTHER_MODE_CAP = 25.0
EPS = 1e-12


@dataclass(frozen=True)
class BattleRow:
    battle_time: str
    mode: str
    map_name: str
    avg_tier: int
    winners: tuple[int, int, int]
    losers: tuple[int, int, int]


@dataclass(frozen=True)
class Layout:
    hero_ids: np.ndarray
    pair_lo: np.ndarray
    pair_hi: np.ndarray

    @classmethod
    def create(cls, hero_ids: Iterable[int]) -> "Layout":
        ids = np.asarray(sorted(set(int(v) for v in hero_ids)), dtype=np.int64)
        if len(ids) < 2:
            raise ValueError("At least two heroes are required")
        lo, hi = np.triu_indices(len(ids), 1)
        return cls(ids, lo.astype(np.int32), hi.astype(np.int32))

    @property
    def n_heroes(self) -> int:
        return len(self.hero_ids)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_lo)

    @property
    def n_features(self) -> int:
        return self.n_heroes + 2 * self.n_pairs

    def columns(self, hero_ids: np.ndarray) -> np.ndarray:
        cols = np.searchsorted(self.hero_ids, hero_ids)
        if np.any(cols >= self.n_heroes) or np.any(self.hero_ids[cols] != hero_ids):
            raise ValueError("Unknown hero ID")
        return cols.astype(np.int32)

    def pair_indices(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return canonical_pair_indices(a, b, self.n_heroes)


def canonical_pair_indices(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Row-major upper-triangle index; self pairs are intentionally not used."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return (lo * (2 * n - lo - 1) // 2 + hi - lo - 1).astype(np.int32)


def parse_battle_time(raw: str) -> int | None:
    """Fast integer UTC parser for the two formats accepted by src/db.rs."""
    value = raw.strip()
    try:
        if len(value) == 20 and value[15] == "." and value[16:18].isdigit() and value.endswith("Z"):
            value = value[:15] + "Z"
        if len(value) != 16 or value[8] != "T" or value[-1] != "Z":
            return None
        year, month, day = int(value[0:4]), int(value[4:6]), int(value[6:8])
        hour, minute, second = int(value[9:11]), int(value[11:13]), int(value[13:15])
        # Howard Hinnant's days_from_civil, relative to Unix epoch.
        y = year - (1 if month <= 2 else 0)
        era = (y if y >= 0 else y - 399) // 400
        yoe = y - era * 400
        mp = month + (-3 if month > 2 else 9)
        doy = (153 * mp + 2) // 5 + day - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        days = era * 146097 + doe - 719468
        if not (1 <= month <= 12 and 1 <= day <= 31 and hour < 24 and minute < 60 and second < 60):
            return None
        return days * 86400 + hour * 3600 + minute * 60 + second
    except (ValueError, IndexError):
        return None


def _cluster_ids(
    winners: np.ndarray,
    losers: np.ndarray,
    mode_codes: np.ndarray,
    map_codes: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of Rust's canonical key + chained <=600s rule."""
    n = len(winners)
    sw, sl = np.sort(winners, axis=1), np.sort(losers, axis=1)
    swap = np.zeros(n, dtype=bool)
    decided = np.zeros(n, dtype=bool)
    for col in range(3):
        less, greater = sl[:, col] < sw[:, col], sl[:, col] > sw[:, col]
        swap |= ~decided & less
        decided |= less | greater
    canonical = np.where(swap[:, None], np.concatenate((sl, sw), axis=1),
                         np.concatenate((sw, sl), axis=1))
    original = np.arange(n, dtype=np.int64)
    keys = (original, timestamps) + tuple(canonical[:, i] for i in range(5, -1, -1)) + (
        mode_codes, map_codes
    )
    order = np.lexsort(keys)
    valid = timestamps >= 0
    sorted_valid = valid[order]
    new_cluster = np.ones(n, dtype=bool)
    key_change = np.zeros(n - 1, dtype=bool)
    key_change |= map_codes[order][1:] != map_codes[order][:-1]
    key_change |= mode_codes[order][1:] != mode_codes[order][:-1]
    for col in range(6):
        key_change |= canonical[order, col][1:] != canonical[order, col][:-1]
    new_cluster[1:] = (
        key_change
        | ~sorted_valid[1:]
        | ~sorted_valid[:-1]
        | (timestamps[order][1:] - timestamps[order][:-1] > SERIES_WINDOW_SECONDS)
    )
    sorted_clusters = np.cumsum(new_cluster, dtype=np.int64) - 1
    clusters = np.empty(n, dtype=np.int64)
    clusters[order] = sorted_clusters
    return clusters


def _build_matrix(
    winner_cols: np.ndarray,
    loser_cols: np.ndarray,
    signs: np.ndarray,
    layout: Layout,
) -> csr_matrix:
    """Six main, nine cross and six teammate entries per battle."""
    n, p = len(signs), layout.n_pairs
    wc, lc = winner_cols, loser_cols
    main_cols = np.concatenate((wc, lc), axis=1)
    main_data = np.broadcast_to(np.asarray([1, 1, 1, -1, -1, -1], dtype=np.float64), (n, 6))

    cross_a = np.repeat(wc, 3, axis=1)
    cross_b = np.tile(lc, (1, 3))
    same = cross_a == cross_b
    cross_cols = layout.n_heroes + layout.pair_indices(cross_a, cross_b)
    cross_cols[same] = 0
    cross_data = np.where(cross_a < cross_b, 1.0, -1.0)
    cross_data[same] = 0.0

    left, right = np.asarray([0, 0, 1]), np.asarray([1, 2, 2])
    synergy_cols = np.concatenate(
        (layout.n_heroes + p + layout.pair_indices(wc[:, left], wc[:, right]),
         layout.n_heroes + p + layout.pair_indices(lc[:, left], lc[:, right])), axis=1
    )
    synergy_data = np.broadcast_to(np.asarray([1, 1, 1, -1, -1, -1], dtype=np.float64), (n, 6))

    cols = np.concatenate((main_cols, cross_cols, synergy_cols), axis=1).astype(np.int32)
    data = np.concatenate((main_data, cross_data, synergy_data), axis=1)
    data *= signs[:, None]
    entries = cols.shape[1]
    indptr = np.arange(0, n * entries + 1, entries, dtype=np.int32)
    matrix = csr_matrix((data.ravel(), cols.ravel(), indptr), shape=(n, layout.n_features))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix
def _map_main_matrix(data: "BattleData", rows: np.ndarray) -> csr_matrix:
    """Oriented per-map hero main effects, used only by the Stage 1 experiment."""
    rows = np.asarray(rows, dtype=np.int64)
    n_maps, n_heroes = len(data.map_names), data.layout.n_heroes
    heroes = np.concatenate((data.winner_cols[rows], data.loser_cols[rows]), axis=1)
    cols = data.map_codes[rows, None] * n_heroes + heroes
    values = np.broadcast_to(np.asarray([1, 1, 1, -1, -1, -1], dtype=np.float64), (len(rows), 6)).copy()
    values *= data.signs[rows, None]
    indptr = np.arange(0, len(rows) * 6 + 1, 6, dtype=np.int32)
    matrix = csr_matrix((values.ravel(), cols.astype(np.int32).ravel(), indptr),
                        shape=(len(rows), n_maps * n_heroes))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _map_main_eta(data: "BattleData", rows: np.ndarray,
                  map_main: dict[int, np.ndarray]) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    eta = np.zeros(len(rows), dtype=np.float64)
    for map_code, delta in map_main.items():
        mask = data.map_codes[rows] == map_code
        if not np.any(mask):
            continue
        selected = rows[mask]
        winner_sum = delta[data.winner_cols[selected]].sum(axis=1)
        loser_sum = delta[data.loser_cols[selected]].sum(axis=1)
        eta[mask] = data.signs[selected] * (winner_sum - loser_sum)
    return eta


@dataclass
class BattleData:
    winners: np.ndarray
    losers: np.ndarray
    winner_cols: np.ndarray
    loser_cols: np.ndarray
    timestamps: np.ndarray
    mode_codes: np.ndarray
    map_codes: np.ndarray
    mode_names: tuple[str, ...]
    map_names: tuple[str, ...]
    cluster_ids: np.ndarray
    signs: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    layout: Layout
    X: csr_matrix

    @classmethod
    def from_arrays(
        cls,
        winners: np.ndarray,
        losers: np.ndarray,
        timestamps: np.ndarray,
        mode_codes: np.ndarray,
        map_codes: np.ndarray,
        mode_names: Sequence[str],
        map_names: Sequence[str],
        hero_ids: Iterable[int],
        decay_half_life_days: float = DECAY_HALF_LIFE_DAYS,
        seed: int = 42,
    ) -> "BattleData":
        winners = np.asarray(winners, dtype=np.int64)
        losers = np.asarray(losers, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        mode_codes = np.asarray(mode_codes, dtype=np.int32)
        map_codes = np.asarray(map_codes, dtype=np.int32)
        n = len(winners)
        if n == 0 or winners.shape != (n, 3) or losers.shape != (n, 3):
            raise ValueError("Expected nonempty (n,3) winner/loser arrays")
        layout = Layout.create(hero_ids)
        winner_cols, loser_cols = layout.columns(winners), layout.columns(losers)
        clusters = _cluster_ids(winners, losers, mode_codes, map_codes, timestamps)
        n_clusters = int(clusters.max()) + 1
        counts = np.bincount(clusters, minlength=n_clusters)
        weights = 1.0 / counts[clusters]
        valid = timestamps >= 0
        if np.any(valid) and math.isfinite(decay_half_life_days) and decay_half_life_days > 0:
            latest = int(timestamps[valid].max())
            age_days = np.maximum(latest - timestamps, 0) / 86400.0
            decay = np.power(0.5, age_days / decay_half_life_days)
            decay[~valid] = 1.0
            weights = weights * decay

        # A random permutation with an exactly half/half cluster orientation avoids
        # both storage-order bias and accidental global label imbalance.
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(n_clusters)
        cluster_signs = np.ones(n_clusters, dtype=np.int8)
        cluster_signs[permutation[: n_clusters // 2]] = -1
        signs = cluster_signs[clusters]
        y = (signs > 0).astype(np.float64)
        X = _build_matrix(winner_cols, loser_cols, signs, layout)
        return cls(winners, losers, winner_cols, loser_cols, timestamps, mode_codes,
                   map_codes, tuple(mode_names), tuple(map_names), clusters, signs, y,
                   np.asarray(weights, dtype=np.float64), layout, X)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[BattleRow],
        hero_names: dict[int, str] | None = None,
        decay_half_life_days: float = math.inf,
        seed: int = 42,
    ) -> "BattleData":
        kept = [r for r in rows if r.avg_tier <= 0 or r.avg_tier >= MIN_TIER]
        if not kept:
            raise ValueError("No valid battles")
        mode_names = tuple(sorted({r.mode for r in kept}))
        map_names = tuple(sorted({r.map_name for r in kept}))
        mode_lookup = {v: i for i, v in enumerate(mode_names)}
        map_lookup = {v: i for i, v in enumerate(map_names)}
        ids = set(hero_names or {}) | {int(v) for r in kept for v in r.winners + r.losers}
        return cls.from_arrays(
            np.asarray([r.winners for r in kept]), np.asarray([r.losers for r in kept]),
            np.asarray([parse_battle_time(r.battle_time) if parse_battle_time(r.battle_time) is not None else -1
                        for r in kept]),
            np.asarray([mode_lookup[r.mode] for r in kept]),
            np.asarray([map_lookup[r.map_name] for r in kept]), mode_names, map_names, ids,
            decay_half_life_days, seed,
        )

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_ids.max()) + 1


def load_database(
    path: str | Path,
    since: str | None = None,
    decay_half_life_days: float = DECAY_HALF_LIFE_DAYS,
    seed: int = 42,
) -> tuple[BattleData, dict[int, str]]:
    """Read one SQLite snapshot through a mode=ro URI; never create/write a DB."""
    db_path = Path(path).resolve()
    if since:
        datetime.strptime(since, "%Y-%m-%d")
        since_compact = since.replace("-", "") + "T000000.000Z"
    else:
        since_compact = None
    uri = db_path.as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.execute("PRAGMA query_only=ON")
    try:
        conn.execute("BEGIN")
        where = "(avg_tier <= 0 OR avg_tier >= ?)"
        params: list[object] = [MIN_TIER]
        if since_compact:
            where += " AND battle_time >= ?"
            params.append(since_compact)
        count = conn.execute(f"SELECT COUNT(*) FROM battles WHERE {where}", params).fetchone()[0]
        if count == 0:
            raise ValueError("No battles remain after filtering")
        hero_names = {int(i): str(n) for i, n in conn.execute("SELECT id, name FROM brawlers")}
        mode_names = tuple(row[0] for row in conn.execute(
            f"SELECT DISTINCT mode FROM battles WHERE {where} ORDER BY mode", params))
        map_names = tuple(row[0] for row in conn.execute(
            f"SELECT DISTINCT map_name FROM battles WHERE {where} ORDER BY map_name", params))
        mode_lookup = {v: i for i, v in enumerate(mode_names)}
        map_lookup = {v: i for i, v in enumerate(map_names)}
        winners = np.empty((count, 3), dtype=np.int64)
        losers = np.empty((count, 3), dtype=np.int64)
        timestamps = np.empty(count, dtype=np.int64)
        modes = np.empty(count, dtype=np.int32)
        maps = np.empty(count, dtype=np.int32)
        query = conn.execute(
            f"SELECT battle_time, mode, map_name, winners_json, losers_json FROM battles WHERE {where}",
            params,
        )
        pos = 0
        while True:
            rows = query.fetchmany(10000)
            if not rows:
                break
            for raw_time, mode, map_name, raw_winners, raw_losers in rows:
                w, l = json.loads(raw_winners), json.loads(raw_losers)
                if len(w) != 3 or len(l) != 3:
                    raise ValueError("Database contains a non-3v3 row")
                winners[pos], losers[pos] = w, l
                parsed = parse_battle_time(raw_time)
                timestamps[pos] = -1 if parsed is None else parsed
                modes[pos], maps[pos] = mode_lookup[mode], map_lookup[map_name]
                pos += 1
        if pos != count:
            raise RuntimeError(f"Snapshot row count changed unexpectedly: {pos} != {count}")
        seen = set(int(v) for v in winners.ravel()) | set(int(v) for v in losers.ravel())
        data = BattleData.from_arrays(winners, losers, timestamps, modes, maps, mode_names,
                                      map_names, seen, decay_half_life_days, seed)
        return data, hero_names
    finally:
        conn.close()


def split_cluster_indices(
    cluster_ids: np.ndarray, seed: int, train_fraction: float = 0.8,
    allowed_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(cluster_ids), dtype=np.int64) if allowed_indices is None else np.asarray(allowed_indices)
    clusters = np.unique(cluster_ids[rows])
    if len(clusters) < 2:
        raise ValueError("At least two BO3 clusters are needed for a split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(clusters)
    n_train = min(max(int(round(len(clusters) * train_fraction)), 1), len(clusters) - 1)
    train_clusters = set(int(v) for v in shuffled[:n_train])
    train_mask = np.asarray([int(v) in train_clusters for v in cluster_ids[rows]])
    train_rows, valid_rows = rows[train_mask], rows[~train_mask]
    assert set(cluster_ids[train_rows]).isdisjoint(set(cluster_ids[valid_rows]))
    return train_rows, valid_rows


@dataclass(frozen=True)
class RegConfig:
    main: float
    matchup: float
    synergy: float
    mode_multiplier: float = 2.0
    map_multiplier: float = 5.0

    def vector(self, layout: Layout, multiplier: float = 1.0) -> np.ndarray:
        return multiplier * np.concatenate((
            np.full(layout.n_heroes, self.main, dtype=np.float64),
            np.full(layout.n_pairs, self.matchup, dtype=np.float64),
            np.full(layout.n_pairs, self.synergy, dtype=np.float64),
        ))


@dataclass
class FitInfo:
    iterations: int
    success: bool
    objective: float
    message: str


@dataclass
class HierarchyModel:
    layout: Layout
    reg: RegConfig
    global_theta: np.ndarray
    mode_delta: dict[int, np.ndarray]
    map_delta: dict[int, np.ndarray]
    map_to_mode: dict[int, int]
    fit_info: dict[str, FitInfo] = field(default_factory=dict)
    stage1_map_main: bool = False
    map_main: dict[int, np.ndarray] = field(default_factory=dict)

    def base_predict(self, data: BattleData, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        eta = np.asarray(data.X[rows] @ self.global_theta).ravel()
        if self.stage1_map_main:
            eta += _map_main_eta(data, rows, self.map_main)
        return eta

    def predict(self, data: BattleData, rows: np.ndarray | None = None, level: str = "map") -> np.ndarray:
        rows = np.arange(len(data.y), dtype=np.int64) if rows is None else np.asarray(rows)
        X = data.X[rows]
        eta = self.base_predict(data, rows)
        if level == "global":
            return eta
        for mode, delta in self.mode_delta.items():
            mask = data.mode_codes[rows] == mode
            if np.any(mask):
                eta[mask] += np.asarray(X[mask] @ delta).ravel()
        if level == "mode":
            return eta
        if level != "map":
            raise ValueError(f"Unknown level: {level}")
        for map_code, delta in self.map_delta.items():
            mask = data.map_codes[rows] == map_code
            if np.any(mask):
                eta[mask] += np.asarray(X[mask] @ delta).ravel()
        return eta

    def theta_for_mode(self, mode: int) -> np.ndarray:
        return self.global_theta + self.mode_delta.get(mode, 0.0)

    def theta_for_map(self, map_code: int) -> np.ndarray:
        mode = self.map_to_mode.get(map_code, -1)
        theta = self.theta_for_mode(mode) + self.map_delta.get(map_code, 0.0)
        if self.stage1_map_main:
            theta = theta.copy()
            theta[:self.layout.n_heroes] += self.map_main.get(
                map_code, np.zeros(self.layout.n_heroes, dtype=np.float64))
        return theta


def logistic_objective(
    theta: np.ndarray, X: csr_matrix, y: np.ndarray, weights: np.ndarray,
    offset: np.ndarray, penalty: np.ndarray,
) -> tuple[float, np.ndarray]:
    logits = offset + np.asarray(X @ theta).ravel()
    losses = np.logaddexp(0.0, logits) - y * logits
    value = float(np.dot(weights, losses) + 0.5 * np.dot(penalty * theta, theta))
    residual = weights * (expit(logits) - y)
    gradient = np.asarray(X.T @ residual).ravel() + penalty * theta
    if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("Non-finite logistic objective or gradient")
    return value, gradient


def fit_logistic(
    X: csr_matrix, y: np.ndarray, weights: np.ndarray, offset: np.ndarray,
    penalty: np.ndarray, initial: np.ndarray | None = None, maxiter: int = 100,
) -> tuple[np.ndarray, FitInfo]:
    start = np.zeros(X.shape[1], dtype=np.float64) if initial is None else np.asarray(initial).copy()
    result = minimize(
        logistic_objective, start, args=(X, y, weights, offset, penalty), jac=True,
        method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxcor": 10},
    )
    if not np.all(np.isfinite(result.x)):
        raise FloatingPointError("Optimizer returned non-finite coefficients")
    return result.x, FitInfo(int(result.nit), bool(result.success), float(result.fun), str(result.message))


def _fit_pair_synergy_residual(
    data: BattleData, rows: np.ndarray, offset: np.ndarray, penalty: np.ndarray,
    initial: np.ndarray | None, maxiter: int,
) -> tuple[np.ndarray, FitInfo]:
    """Freeze all main effects; only matchup and synergy residuals can move."""
    active = np.arange(data.layout.n_heroes, data.layout.n_features)
    start = None if initial is None else initial[active]
    fitted, info = fit_logistic(data.X[rows][:, active], data.y[rows], data.weights[rows],
                                offset, penalty[active], start, maxiter)
    delta = np.zeros(data.layout.n_features, dtype=np.float64)
    delta[active] = fitted
    return delta, info


def fit_hierarchy(
    data: BattleData, rows: np.ndarray, reg: RegConfig,
    initial: HierarchyModel | None = None, maxiter: int = 100,
    stage1_map_main: bool = False,
) -> HierarchyModel:
    rows = np.asarray(rows, dtype=np.int64)
    X, y, w = data.X[rows], data.y[rows], data.weights[rows]
    zero = np.zeros(len(rows), dtype=np.float64)
    n_maps, n_heroes = len(data.map_names), data.layout.n_heroes
    if stage1_map_main:
        map_X = _map_main_matrix(data, rows)
        stage1_X = hstack((X, map_X), format="csr")
        penalty = np.concatenate((reg.vector(data.layout),
                                  np.full(n_maps * n_heroes, reg.main, dtype=np.float64)))
        if initial is None:
            start = None
        else:
            start = np.concatenate((initial.global_theta, *(initial.map_main.get(
                map_code, np.zeros(n_heroes)) for map_code in range(n_maps))))
        fitted, info = fit_logistic(stage1_X, y, w, zero, penalty, start, maxiter)
        theta = fitted[:data.layout.n_features]
        map_main = {map_code: fitted[data.layout.n_features + map_code * n_heroes:
                                     data.layout.n_features + (map_code + 1) * n_heroes].copy()
                    for map_code in np.unique(data.map_codes[rows])}
    else:
        theta, info = fit_logistic(X, y, w, zero, reg.vector(data.layout),
                                   None if initial is None else initial.global_theta, maxiter)
        map_main = {}
    model = HierarchyModel(data.layout, reg, theta, {}, {}, {}, {"global": info},
                           stage1_map_main, map_main)
    for mode in np.unique(data.mode_codes[rows]):
        mask = data.mode_codes[rows] == mode
        subrows = rows[mask]
        offset = model.base_predict(data, subrows)
        start = None if initial is None else initial.mode_delta.get(int(mode))
        penalty = reg.vector(data.layout, reg.mode_multiplier)
        if stage1_map_main:
            delta, detail = _fit_pair_synergy_residual(data, subrows, offset, penalty, start, maxiter)
        else:
            delta, detail = fit_logistic(data.X[subrows], data.y[subrows], data.weights[subrows],
                                         offset, penalty, start, maxiter)
        model.mode_delta[int(mode)] = delta
        model.fit_info[f"mode:{data.mode_names[mode]}"] = detail
    for map_code in np.unique(data.map_codes[rows]):
        map_rows = rows[data.map_codes[rows] == map_code]
        mode = int(np.argmax(np.bincount(data.mode_codes[map_rows], minlength=len(data.mode_names))))
        model.map_to_mode[int(map_code)] = mode
        offset = model.base_predict(data, map_rows)
        offset += np.asarray(data.X[map_rows] @ model.mode_delta[mode]).ravel()
        start = None if initial is None else initial.map_delta.get(int(map_code))
        penalty = reg.vector(data.layout, reg.map_multiplier)
        if stage1_map_main:
            delta, detail = _fit_pair_synergy_residual(data, map_rows, offset, penalty, start, maxiter)
        else:
            delta, detail = fit_logistic(data.X[map_rows], data.y[map_rows], data.weights[map_rows],
                                         offset, penalty, start, maxiter)
        model.map_delta[int(map_code)] = delta
        model.fit_info[f"map:{data.map_names[map_code]}"] = detail
    return model


def _discount(n: np.ndarray, e: np.ndarray, weight: float, cap: float) -> tuple[np.ndarray, np.ndarray]:
    n_eff = np.minimum(weight * n, cap)
    scale = np.divide(n_eff, n, out=np.zeros_like(n_eff), where=n > 0)
    return n_eff, e * scale


@dataclass
class MapCore:
    mode: int
    n: np.ndarray
    e: np.ndarray
    base: np.ndarray
    delta: np.ndarray | None = None


class RustBaseline:
    """Rust analyze's map Log5 residual statistics mapped to a 3v3 logit."""
    def __init__(self, data: BattleData, rows: np.ndarray):
        self.layout = data.layout
        rows = np.asarray(rows, dtype=np.int64)
        n_heroes, n_pairs = data.layout.n_heroes, data.layout.n_pairs
        global_n = np.zeros(n_pairs)
        global_e = np.zeros(n_pairs)
        global_games = np.zeros(n_heroes)
        global_wins = np.zeros(n_heroes)
        mode_n = {m: np.zeros(n_pairs) for m in range(len(data.mode_names))}
        mode_e = {m: np.zeros(n_pairs) for m in range(len(data.mode_names))}
        self.cores: dict[int, MapCore] = {}
        for map_code in np.unique(data.map_codes[rows]):
            r = rows[data.map_codes[rows] == map_code]
            W, L, weights = data.winner_cols[r], data.loser_cols[r], data.weights[r]
            repeated = np.repeat(weights, 3)
            games = (np.bincount(W.ravel(), weights=repeated, minlength=n_heroes)
                     + np.bincount(L.ravel(), weights=repeated, minlength=n_heroes))
            wins = np.bincount(W.ravel(), weights=repeated, minlength=n_heroes)
            a, b = W[:, :, None], L[:, None, :]
            same = a == b
            pair = data.layout.pair_indices(a, b)
            pair_weights = np.broadcast_to(np.repeat(weights, 9).reshape(-1, 3, 3), pair.shape)
            mask = ~same
            n = np.bincount(pair[mask], weights=pair_weights[mask], minlength=n_pairs)
            w = np.bincount(pair[mask], weights=(pair_weights * (a < b))[mask], minlength=n_pairs)
            base = np.clip((wins + BAYES_PRIOR_WEIGHT * 0.5) / (games + BAYES_PRIOR_WEIGHT), 0.05, 0.95)
            pa, pb = base[data.layout.pair_lo], base[data.layout.pair_hi]
            expected = pa * (1 - pb) / (pa * (1 - pb) + (1 - pa) * pb)
            e = w - n * expected
            mode = int(np.argmax(np.bincount(data.mode_codes[r], minlength=len(data.mode_names))))
            self.cores[int(map_code)] = MapCore(mode, n, e, base)
            global_n += n
            global_e += e
            global_games += games
            global_wins += wins
            mode_n[mode] += n
            mode_e[mode] += e
        self.global_delta = global_e / (global_n + MATCHUP_PRIOR_K)
        self.global_base = np.clip(
            (global_wins + BAYES_PRIOR_WEIGHT * 0.5) / (global_games + BAYES_PRIOR_WEIGHT), 0.05, 0.95)
        for core in self.cores.values():
            n2 = np.maximum(mode_n[core.mode] - core.n, 0.0)
            e2 = mode_e[core.mode] - core.e
            n3 = np.maximum(global_n - mode_n[core.mode], 0.0)
            e3 = global_e - mode_e[core.mode]
            n2_eff, e2_eff = _discount(n2, e2, SAME_MODE_WEIGHT, SAME_MODE_CAP)
            n3_eff, e3_eff = _discount(n3, e3, OTHER_MODE_WEIGHT, OTHER_MODE_CAP)
            core.delta = ((core.e + e2_eff + e3_eff)
                          / (core.n + n2_eff + n3_eff + MATCHUP_PRIOR_K))

    def predict(self, data: BattleData, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        eta = np.zeros(len(rows), dtype=np.float64)
        signs = data.signs[rows, None]
        A = np.where(signs > 0, data.winner_cols[rows], data.loser_cols[rows])
        B = np.where(signs > 0, data.loser_cols[rows], data.winner_cols[rows])
        for map_code in np.unique(data.map_codes[rows]):
            mask = data.map_codes[rows] == map_code
            a, b = A[mask], B[mask]
            core = self.cores.get(int(map_code))
            base = self.global_base if core is None else core.base
            delta = self.global_delta if core is None else core.delta
            pa, pb = np.clip(base[a], 0.05, 0.95), np.clip(base[b], 0.05, 0.95)
            score = np.sum(np.log(pa / (1 - pa)) - np.log(pb / (1 - pb)), axis=1)
            aa, bb = a[:, :, None], b[:, None, :]
            same = aa == bb
            pair = data.layout.pair_indices(aa, bb)
            directed = np.where(aa < bb, delta[pair], -delta[pair])
            directed[same] = 0.0
            pair_probability = np.clip(0.5 + directed, 0.05, 0.95)
            score += np.sum(np.log(pair_probability / (1 - pair_probability)), axis=(1, 2))
            eta[mask] = score
        return eta


def metrics(y: np.ndarray, weights: np.ndarray, eta: np.ndarray) -> dict[str, float]:
    total = float(weights.sum())
    log_loss = float(np.dot(weights, np.logaddexp(0.0, eta) - y * eta) / total)
    brier = float(np.dot(weights, (expit(eta) - y) ** 2) / total)
    if not math.isfinite(log_loss) or not math.isfinite(brier):
        raise FloatingPointError("Non-finite evaluation metric")
    return {"log_loss": log_loss, "brier": brier}


def directed_gamma(theta: np.ndarray, layout: Layout, hero_a: int, hero_b: int) -> float:
    if hero_a == hero_b:
        return 0.0
    a, b = layout.columns(np.asarray([hero_a, hero_b]))
    pair = int(layout.pair_indices(a, b))
    value = float(theta[layout.n_heroes + pair])
    return value if a < b else -value


def matchup_entries(theta: np.ndarray, layout: Layout) -> list[list[float | int]]:
    gamma = theta[layout.n_heroes:layout.n_heroes + layout.n_pairs]
    values = np.round(100.0 * (expit(gamma) - 0.5), 2)
    values[values == 0] = 0.0
    return [[int(layout.hero_ids[lo]), int(layout.hero_ids[hi]), float(value)]
            for lo, hi, value in zip(layout.pair_lo, layout.pair_hi, values)]


def main_entries(theta: np.ndarray, layout: Layout) -> list[list[float | int]]:
    values = np.round(theta[:layout.n_heroes], 6)
    return [[int(hero), float(value)] for hero, value in zip(layout.hero_ids, values)]


def synergy_entries(theta: np.ndarray, layout: Layout) -> list[list[float | int]]:
    start = layout.n_heroes + layout.n_pairs
    values = np.round(theta[start:start + layout.n_pairs], 6)
    return [[int(layout.hero_ids[lo]), int(layout.hero_ids[hi]), float(value)]
            for lo, hi, value in zip(layout.pair_lo, layout.pair_hi, values)]


def export_json(
    path: Path, data: BattleData, model: HierarchyModel, best_reg: RegConfig,
    holdout: dict[str, dict[str, float]], selection: list[dict[str, object]],
    elapsed_seconds: float, extra_meta: dict[str, object] | None = None,
) -> None:
    payload = {
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "battles": len(data.y),
            "bo3_clusters": data.n_clusters,
            "effective_weight": float(data.weights.sum()),
            "best_regularization": asdict(best_reg),
            "holdout_metrics": holdout,
            "inner_selection": selection,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "time_decay_half_life_days": DECAY_HALF_LIFE_DAYS,
            "bayes_prior_weight": BAYES_PRIOR_WEIGHT,
            "stage1_main_effect": "map_aware" if model.stage1_map_main else "global_only",
        },
        "global_matchups": matchup_entries(model.global_theta, data.layout),
        "main_effects": main_entries(model.global_theta, data.layout),
        "synergies": synergy_entries(model.global_theta, data.layout),
        "modes": [], "maps": [],
    }
    for mode, name in enumerate(data.mode_names):
        theta = model.theta_for_mode(mode)
        payload["modes"].append({"mode": name, "matchup_table": matchup_entries(theta, data.layout),
                                 "main_effects": main_entries(theta, data.layout),
                                 "synergies": synergy_entries(theta, data.layout)})
    for map_code, name in enumerate(data.map_names):
        theta = model.theta_for_map(map_code)
        payload["maps"].append({"map_name": name,
                                "mode": data.mode_names[model.map_to_mode[map_code]],
                                "matchup_table": matchup_entries(theta, data.layout),
                                "main_effects": main_entries(theta, data.layout),
                                "synergies": synergy_entries(theta, data.layout)})
    if extra_meta:
        payload["meta"].update(extra_meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def top_changes(data: BattleData, model: HierarchyModel, baseline: RustBaseline,
                hero_names: dict[int, str], min_eff: float = 50.0, limit: int = 20) -> list[dict[str, object]]:
    out = []
    for map_code, core in baseline.cores.items():
        theta = model.theta_for_map(map_code)
        new = 100.0 * (expit(theta[data.layout.n_heroes:data.layout.n_heroes + data.layout.n_pairs]) - 0.5)
        for pair in np.flatnonzero(core.n >= min_eff):
            lo, hi = data.layout.pair_lo[pair], data.layout.pair_hi[pair]
            id_a, id_b = int(data.layout.hero_ids[lo]), int(data.layout.hero_ids[hi])
            old_pp, new_pp = float(core.delta[pair] * 100), float(new[pair])
            out.append({"map": data.map_names[map_code], "id_a": id_a, "id_b": id_b,
                        "name_a": hero_names.get(id_a, f"Unknown#{id_a}"),
                        "name_b": hero_names.get(id_b, f"Unknown#{id_b}"),
                        "old_delta_pp": old_pp, "new_delta_pp": new_pp,
                        "n_eff": float(core.n[pair]), "change": abs(new_pp - old_pp)})
    out.sort(key=lambda item: (-float(item["change"]), str(item["map"]), int(item["id_a"]), int(item["id_b"])))
    return out[:limit]


def print_report(holdout: dict[str, dict[str, float]], changes: list[dict[str, object]]) -> None:
    print("\n=== 20% BO3-cluster holdout (weighted) ===")
    print(f"{'Model':<24} {'Log-Loss':>12} {'Brier':>12}")
    for name, values in holdout.items():
        print(f"{name:<24} {values['log_loss']:>12.6f} {values['brier']:>12.6f}")
    print("\n=== Top 20 map matchup changes (map n_eff >= 50) ===")
    print(f"{'Map':<22} {'Hero A':<20} {'Hero B':<20} {'Old pp':>9} {'New pp':>9} {'n_eff':>9}")
    for item in changes:
        print(f"{str(item['map'])[:22]:<22} {str(item['name_a'])[:20]:<20} {str(item['name_b'])[:20]:<20} "
              f"{item['old_delta_pp']:>9.2f} {item['new_delta_pp']:>9.2f} {item['n_eff']:>9.1f}")
def map_delta_distribution(data: BattleData, model: HierarchyModel,
                           baseline: RustBaseline, min_eff: float = 5.0) -> dict[str, float | int]:
    values = []
    for map_code, core in baseline.cores.items():
        theta = model.theta_for_map(map_code)
        delta = 100.0 * (expit(theta[data.layout.n_heroes:
                                     data.layout.n_heroes + data.layout.n_pairs]) - 0.5)
        values.append(np.abs(delta[core.n >= min_eff]))
    joined = np.concatenate(values)
    quantiles = np.quantile(joined, [0.50, 0.75, 0.90, 0.95])
    return {"count": int(len(joined)), "p50": float(quantiles[0]), "p75": float(quantiles[1]),
            "p90": float(quantiles[2]), "p95": float(quantiles[3]),
            "share_abs_ge_1_5": float(np.mean(joined >= 1.5))}


def focused_matchup_changes(data: BattleData, old_model: HierarchyModel, new_model: HierarchyModel,
                            baseline: RustBaseline, hero_names: dict[int, str]) -> list[dict[str, object]]:
    name_to_id = {name.upper(): hero_id for hero_id, name in hero_names.items()}
    map_lookup = {name: code for code, name in enumerate(data.map_names)}
    targets = [("MICO", "MELODIE", "Hot Potato"), ("MICO", "MELODIE", "Belle's Rock"),
               ("MICO", "MELODIE", "Parallel Plays"), ("AMBER", "GIGI", "Gem Fort"),
               ("AMBER", "GIGI", "Hard Rock Mine")]
    rows = []
    for name_a, name_b, map_name in targets:
        id_a, id_b, map_code = name_to_id[name_a], name_to_id[name_b], map_lookup[map_name]
        old_gamma = directed_gamma(old_model.theta_for_map(map_code), data.layout, id_a, id_b)
        new_gamma = directed_gamma(new_model.theta_for_map(map_code), data.layout, id_a, id_b)
        a_col, b_col = data.layout.columns(np.asarray([id_a, id_b]))
        pair = int(data.layout.pair_indices(a_col, b_col))
        old_pp = float(100 * (expit(old_gamma) - 0.5))
        new_pp = float(100 * (expit(new_gamma) - 0.5))
        rows.append({"map": map_name, "hero_a": name_a, "hero_b": name_b,
                     "old_delta_pp": old_pp, "new_delta_pp": new_pp,
                     "change_pp": new_pp - old_pp, "n_eff": float(baseline.cores[map_code].n[pair])})
    return rows


def print_comparison_report(holdout: dict[str, dict[str, float]], focus: list[dict[str, object]],
                            distributions: dict[str, dict[str, float | int]]) -> None:
    print("\n=== Same-snapshot 20% BO3-cluster holdout (weighted) ===")
    print(f"{'Stage 1 main effects':<34} {'Log-Loss':>12} {'Brier':>12}")
    for name, values in holdout.items():
        print(f"{name:<34} {values['log_loss']:>12.6f} {values['brier']:>12.6f}")
    print("\n=== Focused map matchup deltas ===")
    print(f"{'Map':<20} {'Pair':<22} {'Old pp':>9} {'New pp':>9} {'Change':>9} {'n_eff':>9}")
    for item in focus:
        pair = f"{item['hero_a']} vs {item['hero_b']}"
        print(f"{item['map']:<20} {pair:<22} {item['old_delta_pp']:>9.2f} "
              f"{item['new_delta_pp']:>9.2f} {item['change_pp']:>9.2f} {item['n_eff']:>9.1f}")
    print("\n=== |delta_pp| distribution across map entries with n_eff >= 5 ===")
    print(f"{'Stage 1 main effects':<34} {'Count':>8} {'P50':>7} {'P75':>7} {'P90':>7} {'P95':>7} {'>=1.5pp':>10}")
    for name, values in distributions.items():
        print(f"{name:<34} {values['count']:>8d} {values['p50']:>7.3f} {values['p75']:>7.3f} "
              f"{values['p90']:>7.3f} {values['p95']:>7.3f} {100 * values['share_abs_ge_1_5']:>9.2f}%")


def fit_variant(data: BattleData, inner_train: np.ndarray, inner_valid: np.ndarray,
                train_rows: np.ndarray, valid_rows: np.ndarray, args: argparse.Namespace,
                stage1_map_main: bool, label: str) -> dict[str, object]:
    started = time.perf_counter()
    grid = [RegConfig(0.1, 1.0, 1.0), RegConfig(1.0, 10.0, 10.0), RegConfig(10.0, 100.0, 100.0)]
    selection: list[dict[str, object]] = []
    inner_models: list[HierarchyModel] = []
    for index, reg in enumerate(grid, 1):
        tick = time.perf_counter()
        print(f"[{label} grid {index}/{len(grid)}] {asdict(reg)}", flush=True)
        candidate = fit_hierarchy(data, inner_train, reg, maxiter=args.grid_maxiter,
                                  stage1_map_main=stage1_map_main)
        score = metrics(data.y[inner_valid], data.weights[inner_valid],
                        candidate.predict(data, inner_valid, "map"))
        selection.append({"regularization": asdict(reg), **score,
                          "seconds": round(time.perf_counter() - tick, 3)})
        inner_models.append(candidate)
        print(f"  inner log_loss={score['log_loss']:.6f}; brier={score['brier']:.6f}", flush=True)
    best_index = min(range(len(grid)), key=lambda i: float(selection[i]["log_loss"]))
    best_reg = grid[best_index]
    print(f"[{label}] selected {asdict(best_reg)}", flush=True)
    outer = fit_hierarchy(data, train_rows, best_reg, initial=inner_models[best_index],
                          maxiter=args.maxiter, stage1_map_main=stage1_map_main)
    holdout = metrics(data.y[valid_rows], data.weights[valid_rows], outer.predict(data, valid_rows, "map"))
    all_rows = np.arange(len(data.y), dtype=np.int64)
    final = fit_hierarchy(data, all_rows, best_reg, initial=outer, maxiter=args.maxiter,
                          stage1_map_main=stage1_map_main)
    return {"selection": selection, "best_reg": best_reg, "holdout": holdout,
            "final_model": final, "elapsed": time.perf_counter() - started}


def run_comparison(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    print(f"Loading one shared read-only snapshot: {Path(args.db).resolve()}", flush=True)
    data, hero_names = load_database(args.db, args.since, DECAY_HALF_LIFE_DAYS, args.seed)
    train_rows, valid_rows = split_cluster_indices(data.cluster_ids, args.seed + 101)
    inner_train, inner_valid = split_cluster_indices(data.cluster_ids, args.seed + 202,
                                                     allowed_indices=train_rows)
    print(f"Battles={len(data.y):,}; clusters={data.n_clusters:,}; outer={len(train_rows):,}/{len(valid_rows):,}; "
          f"inner={len(inner_train):,}/{len(inner_valid):,}", flush=True)
    old = fit_variant(data, inner_train, inner_valid, train_rows, valid_rows, args, False, "old/global-main")
    new = fit_variant(data, inner_train, inner_valid, train_rows, valid_rows, args, True, "new/map-main")
    full_baseline = RustBaseline(data, np.arange(len(data.y), dtype=np.int64))
    holdout = {"Old +Map (global main)": old["holdout"], "New +Map (map main)": new["holdout"]}
    old_model, new_model = old["final_model"], new["final_model"]
    focus = focused_matchup_changes(data, old_model, new_model, full_baseline, hero_names)
    distributions = {"Old (global main)": map_delta_distribution(data, old_model, full_baseline),
                     "New (map main)": map_delta_distribution(data, new_model, full_baseline)}
    output = Path(args.output)
    old_output = output.with_name(output.stem + "_global_main" + output.suffix)
    extra = {"focused_matchups": focus, "map_delta_distribution": distributions,
             "same_snapshot_holdout": holdout, "experiment_total_seconds": time.perf_counter() - started}
    export_json(old_output, data, old_model, old["best_reg"], holdout, old["selection"], old["elapsed"],
                {"comparison_counterpart": str(output)})
    export_json(output, data, new_model, new["best_reg"], holdout, new["selection"], new["elapsed"],
                {**extra, "old_regularization": asdict(old["best_reg"]),
                 "comparison_counterpart": str(old_output)})
    print_comparison_report(holdout, focus, distributions)
    print(f"\nOld export: {old_output.resolve()}\nNew export: {output.resolve()}")
    print(f"Total comparison elapsed: {time.perf_counter() - started:.1f}s", flush=True)


def run(args: argparse.Namespace) -> None:
    if args.compare_stage1_main:
        return run_comparison(args)
    started = time.perf_counter()
    print(f"Loading read-only snapshot: {Path(args.db).resolve()}", flush=True)
    data, hero_names = load_database(args.db, args.since, DECAY_HALF_LIFE_DAYS, args.seed)
    print(f"Battles={len(data.y):,}; BO3 clusters={data.n_clusters:,}; heroes={data.layout.n_heroes}; "
          f"features={data.layout.n_features:,}; nnz={data.X.nnz:,}", flush=True)
    train_rows, valid_rows = split_cluster_indices(data.cluster_ids, args.seed + 101)
    inner_train, inner_valid = split_cluster_indices(data.cluster_ids, args.seed + 202,
                                                     allowed_indices=train_rows)
    print(f"Outer train/valid battles: {len(train_rows):,}/{len(valid_rows):,}; "
          f"inner train/valid: {len(inner_train):,}/{len(inner_valid):,}", flush=True)
    grid = [RegConfig(0.1, 1.0, 1.0), RegConfig(1.0, 10.0, 10.0), RegConfig(10.0, 100.0, 100.0)]
    stage1_map_main = args.stage1_main == "map"
    selection: list[dict[str, object]] = []
    inner_models: list[HierarchyModel] = []
    for index, reg in enumerate(grid, 1):
        tick = time.perf_counter()
        print(f"[grid {index}/{len(grid)}] {asdict(reg)}", flush=True)
        candidate = fit_hierarchy(data, inner_train, reg, maxiter=args.grid_maxiter,
                                  stage1_map_main=stage1_map_main)
        score = metrics(data.y[inner_valid], data.weights[inner_valid],
                        candidate.predict(data, inner_valid, "map"))
        elapsed = time.perf_counter() - tick
        selection.append({"regularization": asdict(reg), **score, "seconds": round(elapsed, 3)})
        inner_models.append(candidate)
        print(f"  inner +Map log_loss={score['log_loss']:.6f}, brier={score['brier']:.6f}, "
              f"elapsed={elapsed:.1f}s", flush=True)
    best_index = min(range(len(grid)), key=lambda i: float(selection[i]["log_loss"]))
    best_reg = grid[best_index]
    print(f"Selected: {asdict(best_reg)}", flush=True)
    tick = time.perf_counter()
    outer_model = fit_hierarchy(data, train_rows, best_reg, initial=inner_models[best_index],
                                maxiter=args.maxiter, stage1_map_main=stage1_map_main)
    baseline = RustBaseline(data, train_rows)
    print(f"Outer fit + Rust baseline elapsed={time.perf_counter() - tick:.1f}s", flush=True)
    holdout = {
        "Baseline-0": metrics(data.y[valid_rows], data.weights[valid_rows], np.zeros(len(valid_rows))),
        "Baseline-Rust": metrics(data.y[valid_rows], data.weights[valid_rows], baseline.predict(data, valid_rows)),
        "Logit-3v3 (Global)": metrics(data.y[valid_rows], data.weights[valid_rows],
                                       outer_model.predict(data, valid_rows, "global")),
        "Logit-3v3 (+Mode)": metrics(data.y[valid_rows], data.weights[valid_rows],
                                      outer_model.predict(data, valid_rows, "mode")),
        "Logit-3v3 (+Map)": metrics(data.y[valid_rows], data.weights[valid_rows],
                                     outer_model.predict(data, valid_rows, "map")),
    }
    all_rows = np.arange(len(data.y), dtype=np.int64)
    tick = time.perf_counter()
    final_model = fit_hierarchy(data, all_rows, best_reg, initial=outer_model,
                                maxiter=args.maxiter, stage1_map_main=stage1_map_main)
    final_baseline = RustBaseline(data, all_rows)
    print(f"Full-data refit + baseline elapsed={time.perf_counter() - tick:.1f}s", flush=True)
    changes = top_changes(data, final_model, final_baseline, hero_names)
    elapsed = time.perf_counter() - started
    export_json(Path(args.output), data, final_model, best_reg, holdout, selection, elapsed)
    print_report(holdout, changes)
    print(f"\nExported: {Path(args.output).resolve()}")
    print(f"Total elapsed: {elapsed:.1f}s", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="brawl.db", help="SQLite path (opened with mode=ro)")
    parser.add_argument("--since", help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--output", default="dist/regression_matchups.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--grid-maxiter", type=int, default=50)
    parser.add_argument("--stage1-main", choices=("global", "map"), default="map",
                        help="Global-only or map-aware Stage 1 hero main effects")
    parser.add_argument("--compare-stage1-main", action="store_true",
                        help="Fit both variants on one shared snapshot and split")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
