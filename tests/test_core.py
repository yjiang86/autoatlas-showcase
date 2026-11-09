"""Independent tiny CPU oracles for the extracted numerical components."""
from __future__ import annotations
import itertools
import unittest
import numpy as np
import torch
from autoatlas_showcase.planes import AxisTiltCandidate, sample_axis_tilt_planes_torch
from autoatlas_showcase.sequence import circular_180_error, normal_error_deg, plane_normal, sequence_decode

class PlaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self, volume, candidate, side="both", width=None, height=None, mode="nearest"):
        return sample_axis_tilt_planes_torch(
            volume, [candidate], side=side, width=width or volume.shape[2],
            height=height or volume.shape[1], mode=mode, device="cpu", batch_size=1)[0]

    def test_coronal_plane_matches_direct_array_slice(self):
        v = np.arange(5 * 7 * 8, dtype=np.float32).reshape(5, 7, 8)
        for mode in ("nearest", "bilinear"):
            np.testing.assert_allclose(self.sample(v, AxisTiltCandidate(2, 0, 0, 0), mode=mode),
                                       v[2], atol=1e-4, rtol=0)

    def test_left_and_right_partition(self):
        v = np.arange(5 * 7 * 8, dtype=np.float32).reshape(5, 7, 8)
        c = AxisTiltCandidate(2, 0, 0, 0)
        np.testing.assert_array_equal(self.sample(v, c, "left", 4), v[2, :, :4])
        np.testing.assert_array_equal(self.sample(v, c, "right", 4), v[2, :, 4:])

    def test_ninety_degree_rotation_has_independent_numpy_oracle(self):
        v = np.arange(5 * 5 * 5, dtype=np.float32).reshape(5, 5, 5)
        np.testing.assert_array_equal(self.sample(v, AxisTiltCandidate(2, 90, 0, 0)),
                                      np.rot90(v[2], k=-1))

    def test_tilt_samples_expected_ap_per_row(self):
        a, s, r = np.indices((9, 5, 6))
        v = (100 * a + 10 * s + r).astype(np.float32)
        expected = np.stack([v[4 + row - 2, row, :] for row in range(5)])
        np.testing.assert_array_equal(self.sample(v, AxisTiltCandidate(4, 0, 0, 45)), expected)

    def test_chunking_matches_direct_slices(self):
        v = np.arange(5 * 7 * 8, dtype=np.float32).reshape(5, 7, 8)
        cs = [AxisTiltCandidate(i, 0, 0, 0) for i in (0, 2, 4)]
        for chunk in (1, 2, 3):
            actual = sample_axis_tilt_planes_torch(v, cs, side="both", width=8, height=7,
                                                  mode="nearest", device="cpu", batch_size=chunk)
            np.testing.assert_array_equal(np.stack(actual), v[[0, 2, 4]])

    def test_outside_volume_is_zero(self):
        v = np.ones((5, 7, 8), dtype=np.float32)
        np.testing.assert_array_equal(self.sample(v, AxisTiltCandidate(50, 0, 0, 0)), np.zeros((7, 8)))

    def test_invalid_side_and_interpolation_reject(self):
        v = np.ones((5, 7, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.sample(v, AxisTiltCandidate(2, 0, 0, 0), side="unknown")
        with self.assertRaises(ValueError):
            self.sample(v, AxisTiltCandidate(2, 0, 0, 0), mode="unknown")

class SequenceTests(unittest.TestCase):
    def fixture(self):
        candidates = [{"row_id": f"plane-{i}", "ap_index": str(i), "rot_2d_deg": "0",
                       "tilt_axis_deg": "0", "tilt_deg": "0"} for i in (1, 2, 3)]
        queries = [{"dataset_id": "generated", "sample_id": f"synthetic-{i}",
                    "source_number": str(i), "standard_section": "true"} for i in (1, 2, 3)]
        costs = np.asarray([[2.0, 0.2, 0.0], [0.0, 0.1, 2.0], [2.0, 0.0, 0.1]])
        config = {"sequence_decoder": {"candidate_states_per_section": 3,
                  "reversal_tolerance_um": 0, "soft_max_forward_step_um": 100,
                  "reversal_penalty": 100, "large_step_penalty": 0,
                  "adjacent_pose_normal_weight": 0, "adjacent_rot_weight": 0}}
        return candidates, queries, costs, config

    def test_decoder_matches_exhaustive_scalar_path_search(self):
        candidates, queries, costs, config = self.fixture()
        def path_cost(path):
            emission = sum(float(costs[i, p]) for i, p in enumerate(path))
            reversal = sum(100 * max(0, (right - left) * 10) / 500
                           for left, right in zip(path, path[1:]))
            return emission + reversal
        oracle = min(itertools.product(range(3), repeat=3), key=path_cost)
        trace, selected = sequence_decode(queries, costs, candidates, config)
        actual = tuple(selected[("generated", q["sample_id"])] for q in queries)
        self.assertEqual(actual, oracle)
        self.assertEqual(actual, (2, 1, 1))
        self.assertAlmostEqual(path_cost(actual), 0.1)
        self.assertEqual(len(trace), 3)

    def test_query_order_is_from_section_metadata(self):
        candidates, queries, costs, config = self.fixture()
        order = [2, 0, 1]
        _, selected = sequence_decode([queries[i] for i in order], costs[order], candidates, config)
        self.assertEqual([selected[("generated", q["sample_id"])] for q in queries], [2, 1, 1])

    def test_nonstandard_section_is_independent(self):
        candidates, queries, costs, config = self.fixture()
        queries[1]["standard_section"] = "false"
        trace, selected = sequence_decode(queries, costs, candidates, config)
        self.assertEqual(selected[("generated", "synthetic-2")], 0)
        row = next(r for r in trace if r["sample_id"] == "synthetic-2")
        self.assertEqual(row["selection_mode"], "per_image_alternate_not_in_sequence")

    def test_plane_normal_and_angle_conventions(self):
        np.testing.assert_allclose(plane_normal(0, 0, 0), [1, 0, 0])
        self.assertAlmostEqual(normal_error_deg(plane_normal(0, 0, 0), plane_normal(0, 0, 30)), 30)
        self.assertAlmostEqual(circular_180_error(179, 1), 2)

if __name__ == "__main__":
    unittest.main()
