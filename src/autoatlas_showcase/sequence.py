"""Portable numerical excerpts from the AutoAtlas sequence diagnostic."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def plane_normal(rot_deg: float, axis_deg: float, tilt_deg: float) -> np.ndarray:
    effective_axis = math.radians(float(axis_deg) + float(rot_deg))
    slope = math.tan(math.radians(float(tilt_deg)))
    vector = np.asarray([1.0, -slope * math.cos(effective_axis), -slope * math.sin(effective_axis)])
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def normal_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    dot = float(np.clip(abs(float(np.dot(left, right))), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def circular_180_error(left: float, right: float) -> float:
    return abs(((float(left) - float(right) + 90.0) % 180.0) - 90.0)


def sequence_decode(
    query_rows: list[dict[str, str]],
    joint_scores: np.ndarray,
    candidates: list[dict[str, str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    contract = config["sequence_decoder"]
    top_n = int(contract["candidate_states_per_section"])
    selected_by_key: dict[tuple[str, str], int] = {}
    trace_rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[int]] = {}
    for query_index, row in enumerate(query_rows):
        if str(row["standard_section"]).lower() == "true":
            by_dataset.setdefault(row["dataset_id"], []).append(query_index)
    normals = np.stack(
        [
            plane_normal(float(row["rot_2d_deg"]), float(row["tilt_axis_deg"]), float(row["tilt_deg"]))
            for row in candidates
        ]
    )
    rotations = np.asarray([float(row["rot_2d_deg"]) for row in candidates])
    aps = np.asarray([int(row["ap_index"]) for row in candidates])
    candidate_keys = np.asarray([row["row_id"] for row in candidates])
    for dataset_id, query_indices in by_dataset.items():
        query_indices.sort(key=lambda index: (int(query_rows[index]["source_number"]), query_rows[index]["sample_id"]))
        state_indices = [
            np.lexsort((candidate_keys, joint_scores[index]))[:top_n].astype(np.int32)
            for index in query_indices
        ]
        costs = joint_scores[query_indices[0], state_indices[0]].astype(np.float64)
        backpointers: list[np.ndarray] = []
        for step in range(1, len(query_indices)):
            previous = state_indices[step - 1]
            current = state_indices[step]
            prev_ap = aps[previous][:, None]
            curr_ap = aps[current][None, :]
            reversal_um = (curr_ap - prev_ap) * 10.0
            reversal = np.maximum(0.0, reversal_um - float(contract["reversal_tolerance_um"]))
            forward_drop_um = (prev_ap - curr_ap) * 10.0
            large = np.maximum(0.0, forward_drop_um - float(contract["soft_max_forward_step_um"]))
            transition = float(contract["reversal_penalty"]) * reversal / 500.0
            transition += float(contract["large_step_penalty"]) * (large / 600.0) ** 2
            normal_dot = np.clip(np.abs(normals[previous] @ normals[current].T), 0.0, 1.0)
            normal_delta = np.degrees(np.arccos(normal_dot))
            transition += float(contract["adjacent_pose_normal_weight"]) * normal_delta / 30.0
            rot_delta = np.abs(((rotations[previous][:, None] - rotations[current][None, :] + 90.0) % 180.0) - 90.0)
            transition += float(contract["adjacent_rot_weight"]) * rot_delta / 30.0
            total = costs[:, None] + transition
            pointer = np.argmin(total, axis=0)
            costs = total[pointer, np.arange(current.size)] + joint_scores[query_indices[step], current]
            backpointers.append(pointer.astype(np.int32))
        chosen_positions = [int(np.argmin(costs))]
        for pointer in reversed(backpointers):
            chosen_positions.append(int(pointer[chosen_positions[-1]]))
        chosen_positions.reverse()
        for step, query_index in enumerate(query_indices):
            candidate_index = int(state_indices[step][chosen_positions[step]])
            key = (query_rows[query_index]["dataset_id"], query_rows[query_index]["sample_id"])
            selected_by_key[key] = candidate_index
            trace_rows.append(
                {
                    "dataset_id": key[0],
                    "sample_id": key[1],
                    "query_index": query_index,
                    "candidate_index": candidate_index,
                    "candidate_row_id": candidates[candidate_index]["row_id"],
                    "emission_score": float(joint_scores[query_index, candidate_index]),
                    "selection_mode": "cohort_viterbi_truthfree",
                }
            )
    for query_index, row in enumerate(query_rows):
        key = (row["dataset_id"], row["sample_id"])
        if key in selected_by_key:
            continue
        candidate_index = int(np.lexsort((candidate_keys, joint_scores[query_index]))[0])
        selected_by_key[key] = candidate_index
        trace_rows.append(
            {
                "dataset_id": key[0],
                "sample_id": key[1],
                "query_index": query_index,
                "candidate_index": candidate_index,
                "candidate_row_id": candidates[candidate_index]["row_id"],
                "emission_score": float(joint_scores[query_index, candidate_index]),
                "selection_mode": "per_image_alternate_not_in_sequence",
            }
        )
    return trace_rows, selected_by_key
