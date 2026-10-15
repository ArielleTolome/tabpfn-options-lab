"""Small stdlib-only causal checks; no model downloads, fits, or GPU use."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from demo import evaluate, validate_input
from tabpfn_filter import (
    RollingTradeClassifier, _chronological_split, _class1_probabilities,
    _timestamp, probability_quantity,
)


X = [50, 0.03, 0.9, 55, 0.15, -0.16]
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def trade(entry, label, y):
    return {"x": [50 + entry, *X[1:]], "y": y,
            "entry_time": (START + timedelta(days=entry)).isoformat(),
            "label_time": (START + timedelta(days=label)).isoformat()}


def small_input():
    return {"schema_version": 1, "data_kind": "authorized",
            "history": [trade(0, 1, 0), trade(1, 2, 1)],
            "candidates": [dict(id="first", **trade(4, 6, 1)),
                           dict(id="second", **trade(5, 8, 0))]}


class CausalChecks(unittest.TestCase):
    def test_resolve_before_window_and_strict_cutoff(self):
        original = [trade(0, 1, 1), trade(1, 2, 0), trade(2, 3, 1),
                    trade(3, 4, 0), trade(3.5, 9, 1), trade(7, 8, 0)]
        model = RollingTradeClassifier("empirical", original, min_samples=2, max_samples=2)
        cutoff = START + timedelta(days=4)
        rows = model._training_rows(cutoff)
        self.assertEqual([row["x"][0] for row in rows], [51, 52])
        self.assertTrue(all(row["label_time"] < cutoff for row in rows))
        self.assertEqual(model.predict(X, cutoff), 0.5)
        self.assertEqual(model.metadata["n_train"], 2)
        self.assertEqual(model.metadata["latest_label_time"], "2024-01-04T00:00:00Z")
        # A label at the decision instant is not yet admissible.
        self.assertNotIn(_timestamp(original[3]["label_time"]), [row["label_time"] for row in rows])
        original[1]["x"][0] = 99
        original[1]["y"] = 1
        self.assertEqual(model.predict(X, cutoff), 0.5)
        self.assertEqual(model._training_rows(cutoff)[0]["x"][0], 51)
        self.assertIsNone(model.predict(X, START + timedelta(days=1)))

    def test_purged_tuning_excludes_boundary_labels(self):
        model = RollingTradeClassifier("empirical", [trade(0, 1, 0), trade(1, 6, 1),
                                                       trade(2, 4, 1), trade(3, 3.5, 1),
                                                       trade(4, 5, 0)], min_samples=2)
        training, validation = _chronological_split(model._training_rows(START + timedelta(days=10)))
        self.assertEqual([row["x"][0] for row in training], [50, 53])
        self.assertEqual(len(validation), 1)
        self.assertTrue(all(row["label_time"] < validation[0]["entry_time"] for row in training))

    def test_probability_tiers_and_class_order(self):
        for p, quantity in ((0, 0), (0.819999, 0), (0.82, 1), (0.899999, 1), (0.90, 2), (1, 2)):
            with self.subTest(p=p):
                self.assertEqual(probability_quantity(p), quantity)
        self.assertEqual(_class1_probabilities([1, 0], [[0.9, 0.1]]), [0.9])
        for value in (True, "0.9", float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probability_quantity(value)
        with self.assertRaises(ValueError):
            _class1_probabilities([0, 1], [[0.8, 0.8]])

    def test_input_errors_and_insufficient_history(self):
        payload = small_input()
        self.assertEqual(validate_input(payload)["data_kind"], "authorized")
        mutations = [
            lambda p: p.update(schema_version=True),
            lambda p: p.update(data_kind="market"),
            lambda p: p.update(unexpected="ignored?"),
            lambda p: p["history"][0].update(y=True),
            lambda p: p["history"][0].update(x=[101, *X[1:]]),
            lambda p: p["history"][0].update(x=[50, float("nan"), *X[2:]]),
            lambda p: p["history"][0].update(x=None),
            lambda p: p["history"][0].update(entry_time="2024-01-01T00:00:00"),
            lambda p: p["history"][0].update(label_time=p["history"][0]["entry_time"]),
            lambda p: p["history"].append(deepcopy(p["history"][0])),
            lambda p: p["candidates"].reverse(),
            lambda p: p["candidates"][1].update(id="first"),
        ]
        for index, mutate in enumerate(mutations):
            bad = deepcopy(payload)
            mutate(bad)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_input(bad)
        with self.assertRaisesRegex(ValueError, "500\\+"):
            evaluate(payload)
        with self.assertRaises(ValueError):
            RollingTradeClassifier("empirical", device="invalid")


if __name__ == "__main__":
    unittest.main()
