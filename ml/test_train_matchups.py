from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.special import expit

from train_matchups import (
    BattleData,
    BattleRow,
    RegConfig,
    directed_gamma,
    fit_hierarchy,
    fit_logistic,
    logistic_objective,
    matchup_entries,
    split_cluster_indices,
)


def row_at(index: int, winners: tuple[int, int, int], losers: tuple[int, int, int],
           map_name: str = "Test Map", mode: str = "knockout") -> BattleRow:
    moment = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index * 601)
    return BattleRow(moment.strftime("%Y%m%dT%H%M%S") + ".000Z", mode, map_name, 20, winners, losers)


class TrainMatchupsTests(unittest.TestCase):
    def test_red_blue_swap_invariance(self) -> None:
        rows = [row_at(0, (1, 2, 3), (4, 5, 6)), row_at(1, (2, 3, 6), (1, 4, 5))]
        data = BattleData.from_rows(rows)
        rng = np.random.default_rng(7)
        theta = rng.normal(0, 0.2, data.layout.n_features)
        eta = np.asarray(data.X @ theta).ravel()
        swapped = np.asarray((-data.X) @ theta).ravel()
        self.assertLess(float(np.max(np.abs(expit(eta) + expit(swapped) - 1.0))), 1e-9)

    def test_matchup_strict_antisymmetry(self) -> None:
        data = BattleData.from_rows([row_at(0, (1, 2, 3), (4, 5, 6))])
        theta = np.linspace(-0.3, 0.3, data.layout.n_features)
        for a, b in [(1, 4), (2, 6), (5, 3)]:
            self.assertEqual(directed_gamma(theta, data.layout, a, b),
                             -directed_gamma(theta, data.layout, b, a))
        entries = matchup_entries(theta, data.layout)
        lookup = {(a, b): delta for a, b, delta in entries}
        for (a, b), delta in lookup.items():
            reverse = directed_gamma(theta, data.layout, b, a)
            self.assertAlmostEqual(float(delta), 100 * (expit(-reverse) - 0.5), places=2)

    def test_extreme_data_is_numerically_stable(self) -> None:
        data = BattleData.from_rows([row_at(0, (1, 2, 3), (4, 5, 6))])
        X = data.X
        penalty = np.full(data.layout.n_features, 0.1)
        for label in (0.0, 1.0):
            y = np.asarray([label])
            w = np.ones(1)
            offset = np.zeros(1)
            for magnitude in (0.0, 1e3, 1e6):
                theta = np.full(data.layout.n_features, magnitude)
                value, gradient = logistic_objective(theta, X, y, w, offset, penalty)
                self.assertTrue(np.isfinite(value))
                self.assertTrue(np.all(np.isfinite(gradient)))
            fitted, _ = fit_logistic(X, y, w, offset, penalty, maxiter=30)
            self.assertTrue(np.all(np.isfinite(fitted)))
            self.assertTrue(np.isfinite(fit_logistic(X, y, w, offset, penalty, maxiter=2)[1].objective))

    def test_cluster_split_has_zero_leakage(self) -> None:
        rows = [
            BattleRow("20260101T000000.000Z", "knockout", "M", 20, (1, 2, 3), (4, 5, 6)),
            BattleRow("20260101T000100.000Z", "knockout", "M", 20, (4, 5, 6), (1, 2, 3)),
            BattleRow("20260101T010000.000Z", "knockout", "M", 20, (1, 2, 3), (4, 5, 6)),
            row_at(100, (1, 2, 4), (3, 5, 6)),
        ]
        data = BattleData.from_rows(rows)
        self.assertEqual(data.cluster_ids[0], data.cluster_ids[1])
        self.assertNotEqual(data.cluster_ids[0], data.cluster_ids[2])
        for seed in range(10):
            train, valid = split_cluster_indices(data.cluster_ids, seed, train_fraction=0.5)
            self.assertTrue(set(data.cluster_ids[train]).isdisjoint(set(data.cluster_ids[valid])))

    def test_stage1_map_main_controls_each_map_and_freezes_lower_mains(self) -> None:
        rows = []
        for index in range(40):
            map_name = "Alpha" if index < 20 else "Beta"
            winners, losers = ((1, 2, 3), (4, 5, 6)) if index < 20 else ((4, 5, 6), (1, 2, 3))
            rows.append(row_at(index, winners, losers, map_name=map_name))
        data = BattleData.from_rows(rows)
        model = fit_hierarchy(data, np.arange(len(rows)), RegConfig(0.01, 0.1, 0.1),
                              maxiter=80, stage1_map_main=True)
        self.assertTrue(model.stage1_map_main)
        self.assertEqual(len(model.map_main), 2)
        self.assertGreater(abs(model.map_main[0][0] - model.map_main[1][0]), 0.05)
        for delta in list(model.mode_delta.values()) + list(model.map_delta.values()):
            self.assertTrue(np.all(delta[:data.layout.n_heroes] == 0.0))

        # Swap A/B while preserving the physical winner: orientation, labels and
        # design matrix flip together. Swapping winners/losers would instead flip
        # the actual outcome and is not a red/blue invariance test.
        swapped_data = copy.copy(data)
        swapped_data.signs = -data.signs
        swapped_data.y = 1.0 - data.y
        swapped_data.X = -data.X
        swapped_model = fit_hierarchy(swapped_data, np.arange(len(rows)), RegConfig(0.01, 0.1, 0.1),
                                      maxiter=80, stage1_map_main=True)
        p_original = expit(model.predict(data, level="map"))
        p_swapped = expit(swapped_model.predict(swapped_data, level="map"))
        differences = np.abs(p_original + p_swapped - 1.0)
        worst = int(np.argmax(differences))
        self.assertLess(float(differences[worst]), 1e-6,
                        f"row={worst}, p={p_original[worst]}/{p_swapped[worst]}, "
                        f"global fits={model.fit_info['global']}/{swapped_model.fit_info['global']}")

    def test_deconfounds_bound_strong_teammate(self) -> None:
        """A never appears without C; C is strong, while the true A-B effect is zero."""
        rng = np.random.default_rng(20261124)
        a, b, c, d, e, f, g = 1, 2, 3, 4, 5, 6, 7
        strong_lineups = [(a, c, d), (c, d, g)]
        weak_lineups = [(b, e, f), (e, f, g)]
        rows = []
        index = 0
        for strong in strong_lineups:
            for weak in weak_lineups:
                for _ in range(600):
                    # Only C carries a real effect. Randomizing the outcome avoids a
                    # deterministic fit and gives the naive A-B pair its false edge.
                    strong_wins = rng.random() < expit(1.4)
                    winners, losers = (strong, weak) if strong_wins else (weak, strong)
                    rows.append(row_at(index, winners, losers))
                    index += 1
        data = BattleData.from_rows(rows)
        model = fit_hierarchy(data, np.arange(len(rows)), RegConfig(0.05, 1.0, 0.2), maxiter=150)

        wins = games = 0
        for battle in rows:
            if a in battle.winners and b in battle.losers:
                games += 1
                wins += 1
            elif a in battle.losers and b in battle.winners:
                games += 1
        naive_delta = wins / games - 0.5
        gamma_ab = directed_gamma(model.global_theta, data.layout, a, b)
        alpha_c = model.global_theta[int(data.layout.columns(np.asarray([c]))[0])]
        alpha_b = model.global_theta[int(data.layout.columns(np.asarray([b]))[0])]

        def synergy(x: int, y: int) -> float:
            xc, yc = data.layout.columns(np.asarray([x, y]))
            pair = int(data.layout.pair_indices(xc, yc))
            value = model.global_theta[data.layout.n_heroes + data.layout.n_pairs + pair]
            return float(value if xc < yc else -value)

        # Main effects and teammate synergies are deliberately both valid sinks for
        # C's strength; requiring alpha_C alone to carry everything would contradict
        # the model specification. Average C attribution over its two actual lineups.
        c_attribution = 0.5 * (
            alpha_c + synergy(c, a) + synergy(c, d)
            + alpha_c + synergy(c, d) + synergy(c, g)
        )
        self.assertGreater(naive_delta, 0.15, "Synthetic naive pair must show the confounded edge")
        self.assertLess(abs(gamma_ab), 0.20, f"gamma(A,B) should be near zero, got {gamma_ab}")
        self.assertGreater(c_attribution, 0.20,
                           f"C main+synergy attribution should be positive, got {c_attribution}")
        self.assertGreater(alpha_c - alpha_b, 0.20,
                           f"C should be strong relative to B, got alpha gap {alpha_c - alpha_b}")


if __name__ == "__main__":
    unittest.main()
