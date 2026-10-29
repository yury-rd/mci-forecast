import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import features
import forecast


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)
        cls.splits = json.loads((ROOT / "data/splits.json").read_text())
        cls.data = forecast.synthetic_data()
        cls.expected = json.loads((ROOT / "tests/expected_demo.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def assert_results_match(self, result):
        self.assertEqual(result["participants"], 48)
        self.assertEqual(result["outer_repeats"], 3)
        self.assertEqual(result["results"].keys(), self.expected["results"].keys())
        for name, values in result["results"].items():
            expected = self.expected["results"][name]
            np.testing.assert_allclose(values["MAE_by_repeat"], expected["MAE_by_repeat"], rtol=0, atol=1e-10)
            self.assertAlmostEqual(values["mean_MAE"], expected["mean_MAE"], places=10)
            self.assertEqual(values["choices"], expected["choices"])

    def test_synthetic_regression(self):
        self.assert_results_match(forecast.evaluate(self.data, self.splits))

    def test_outer_test_rows_do_not_change_training(self):
        split = self.splits[0]
        before = forecast.fit_model(self.data, split, "combined")
        changed = copy.deepcopy(self.data)
        changed["combined"][split["test"]] += 300
        changed["clinical_numeric"][split["test"]] -= 400
        changed["categories"][split["test"]] = "unseen"
        changed["target"][split["test"]] += 1000
        after = forecast.fit_model(changed, split, "combined")
        self.assertEqual(before["selected"], after["selected"])
        self.assertEqual(before["outer_transform"]["categories"], after["outer_transform"]["categories"])
        np.testing.assert_array_equal(before["outer_transform"]["MRI"]["train_scores"],
                                      after["outer_transform"]["MRI"]["train_scores"])
        for a, b in zip(before["outer_ridge_parameters"], after["outer_ridge_parameters"]):
            np.testing.assert_array_equal(a, b)

    def test_inner_validation_rows_do_not_change_their_training_projection(self):
        split = self.splits[0]
        before = forecast.fit_model(self.data, split, "combined")
        changed = copy.deepcopy(self.data)
        validation = split["inner"][0]["test"]
        changed["combined"][validation] += 300
        changed["target"][validation] += 1000
        after = forecast.fit_model(changed, split, "combined")
        np.testing.assert_array_equal(before["inner_transforms"][0]["MRI"]["train_scores"],
                                      after["inner_transforms"][0]["MRI"]["train_scores"])

    def test_splits_reject_overlap_missing_folds_and_invalid_indices(self):
        forecast.validate_splits(self.splits, 48)
        cases = [[], self.splits[:-1]]
        overlap = copy.deepcopy(self.splits)
        overlap[0]["train"].append(overlap[0]["test"][0])
        cases.append(overlap)
        inner_overlap = copy.deepcopy(self.splits)
        inner_overlap[0]["inner"][0]["train"].append(inner_overlap[0]["inner"][0]["test"][0])
        cases.append(inner_overlap)
        for index in [0.5, "0", True, -1, 48]:
            changed = copy.deepcopy(self.splits)
            changed[0]["train"][0] = index
            cases.append(changed)
        for invalid in cases:
            with self.subTest(case=len(invalid)), self.assertRaises(ValueError):
                forecast.validate_splits(invalid, 48)

    def test_data_validation_and_empty_categories(self):
        data = copy.deepcopy(self.data)
        data["categories"] = np.empty((48, 0), dtype="U1")
        self.assertEqual(forecast.validate_data(data)["categories"].shape, (48, 0))
        fitted = forecast.fit_model(data, self.splits[0], "clinical")
        self.assertTrue(np.isfinite(fitted["outer_prediction"]).all())
        invalid_cases = [
            {**data, "target": np.zeros((48, 1))},
            {**data, "static": np.zeros((48, 20))},
            {**data, "target": np.full(48, np.nan)},
            {**data, "categories": np.zeros((48, 1))},
            {**data, "participant_ids": np.zeros(48)},
            {key: value for key, value in data.items() if key != "target"},
        ]
        for invalid in invalid_cases:
            with self.assertRaises(ValueError):
                forecast.validate_data(invalid)

    def test_insufficient_pls_rank_is_reported(self):
        data = {**self.data, "static": np.zeros((48, 21))}
        with self.assertRaisesRegex(ValueError, "insufficient rank"):
            forecast.fit_model(data, self.splits[0], "static")

    def test_features_match_direct_window_correlations(self):
        x = np.random.default_rng(732401).normal(size=(197, 7))
        measured = features.extract(x, 3.0)
        self.assertEqual((measured["window_samples"], measured["stride_samples"]), (20, 10))
        edges = np.triu_indices(7, 1)
        np.testing.assert_allclose(measured["static"], np.corrcoef(x, rowvar=False)[edges], rtol=0, atol=1e-12)
        windows = np.stack([np.corrcoef(x[start:start + 20], rowvar=False)[edges]
                            for start in range(0, 178, 10)])
        np.testing.assert_allclose(measured["dispersion"], windows.std(axis=0, ddof=1), rtol=0, atol=1e-12)
        for invalid in [x[:20], np.ones((197, 7))]:
            with self.assertRaises(ValueError):
                features.extract(invalid, 3.0)

    def test_common_phase_preserves_cross_spectra_and_static_fc(self):
        for samples in [196, 197]:
            x = np.random.default_rng(samples).normal(size=(samples, 7))
            y = features.common_phase(x, 41)
            before, after = np.fft.rfft(x, axis=0), np.fft.rfft(y, axis=0)
            np.testing.assert_allclose(before[:, :, None] * before[:, None, :].conj(),
                                       after[:, :, None] * after[:, None, :].conj(), rtol=0, atol=1e-10)
            np.testing.assert_allclose(features.correlation(x), features.correlation(y), rtol=0, atol=1e-12)

    def test_four_window_controls_use_equal_unique_frame_counts(self):
        for starts in features.nonoverlapping_windows(197, 20).values():
            self.assertEqual(len(starts), 4)
            frames = [frame for start in starts for frame in range(start, start + 20)]
            self.assertEqual(len(set(frames)), 80)
            self.assertTrue(all(0 <= frame < 197 for frame in frames))
        for mode in ["first8", "middle8", "last8", "spread8"]:
            indices = features.window_indices(18, mode)
            self.assertEqual(len(set(indices)), 8)
            self.assertTrue(np.all((0 <= indices) & (indices < 18)))

    def test_cli_summary_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            process = subprocess.run([sys.executable, str(ROOT / "run.py"), "--mode", "summary"],
                                     cwd=folder, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(process.stdout), json.loads((ROOT / "data/results.json").read_text()))

    def test_cli_local_input_and_output_path_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            np.savez(directory / "features.npz", **self.data)
            output = directory / "output folder/predictions.json"
            subprocess.run([sys.executable, str(ROOT / "run.py"), "--input", str(directory / "features.npz"),
                            "--splits", str(ROOT / "data/splits.json"), "--output", str(output)],
                           cwd=folder, capture_output=True, text=True, check=True)
            result = json.loads(output.read_text())
        self.assertEqual(result["scope"], "local cohort")
        self.assert_results_match(result)

    def test_cli_rejects_ambiguous_local_splits_and_summary_arguments(self):
        for arguments in [["--input", "features.npz"], ["--mode", "summary", "--splits", "splits.json"]]:
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), *arguments], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
