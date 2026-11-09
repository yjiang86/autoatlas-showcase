"""Generated CPU software example, not a real registration benchmark."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from autoatlas_showcase.planes import AxisTiltCandidate, sample_axis_tilt_planes_torch
from autoatlas_showcase.sequence import sequence_decode

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo-output"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    a, s, r = np.indices((5, 7, 8))
    volume = (100 * a + 10 * s + r).astype(np.float32)
    planes = sample_axis_tilt_planes_torch(
        volume, [AxisTiltCandidate(float(i), 0.0, 0.0, 0.0) for i in (1, 2, 3)],
        side="both", width=8, height=7, mode="nearest", device="cpu", batch_size=2)
    with (args.out / "generated_plane_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["synthetic_ap_index", "rows", "columns", "mean"])
        w.writeheader()
        for i, plane in zip((1, 2, 3), planes):
            w.writerow({"synthetic_ap_index": i, "rows": plane.shape[0],
                        "columns": plane.shape[1], "mean": float(plane.mean())})
    candidates = [{"row_id": f"plane-{i}", "ap_index": str(i), "rot_2d_deg": "0",
                   "tilt_axis_deg": "0", "tilt_deg": "0"} for i in (1, 2, 3)]
    queries = [{"dataset_id": "generated-example", "sample_id": f"synthetic-{i}",
                "source_number": str(i), "standard_section": "true"} for i in (1, 2, 3)]
    costs = np.asarray([[2.0, 0.2, 0.0], [0.0, 0.1, 2.0], [2.0, 0.0, 0.1]])
    config = {"sequence_decoder": {"candidate_states_per_section": 3,
              "reversal_tolerance_um": 0, "soft_max_forward_step_um": 100,
              "reversal_penalty": 100, "large_step_penalty": 0,
              "adjacent_pose_normal_weight": 0, "adjacent_rot_weight": 0}}
    trace, _ = sequence_decode(queries, costs, candidates, config)
    with (args.out / "generated_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0]))
        w.writeheader()
        w.writerows(trace)
    print("Generated software example complete: 3 planes and 3 sequence rows.")
    print("No real subject, trained model, truth scoring, or scientific benchmark was used.")

if __name__ == "__main__":
    main()
