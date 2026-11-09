from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AxisTiltCandidate:
    ap: float
    rot_2d_deg: float
    tilt_axis_deg: float
    tilt_deg: float


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _as_volume_tensor(volume: np.ndarray, *, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    np_dtype = np.float32 if dtype == torch.float32 else None
    arr = np.asarray(volume, dtype=np_dtype)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    resolved = _resolve_device(device)
    return torch.as_tensor(arr, device=resolved, dtype=dtype).unsqueeze(0).unsqueeze(0)


def _normalize_grid(coord: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(coord)
    return (2.0 * coord / float(size - 1)) - 1.0


def build_axis_tilt_grid(
    candidates: list[AxisTiltCandidate],
    *,
    volume_shape: tuple[int, int, int],
    side: str,
    width: int,
    height: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Build a batched grid_sample grid matching v408a2 scipy coordinates.

    Volume axes are Allen `(AP, S/I, R/L)`. PyTorch 5D grid_sample uses grid
    order `(x, y, z)` mapping to `(R/L, S/I, AP)`.
    """
    shape_a, shape_s, shape_r = volume_shape
    mid_r = shape_r // 2
    if side == "left":
        r_min, r_max = 0.0, float(mid_r - 1)
    elif side == "right":
        r_min, r_max = float(mid_r), float(shape_r - 1)
    elif side == "both":
        r_min, r_max = 0.0, float(shape_r - 1)
    else:
        raise ValueError(f"Unsupported side={side!r}")

    s_axis = torch.linspace(0.0, float(shape_s - 1), height, device=device, dtype=torch.float32)
    r_axis = torch.linspace(r_min, r_max, width, device=device, dtype=torch.float32)
    ss, rr = torch.meshgrid(s_axis, r_axis, indexing="ij")
    s_center = (shape_s - 1) / 2.0
    r_center = (r_min + r_max) / 2.0
    s_rel_base = ss - s_center
    r_rel_base = rr - r_center

    grids: list[torch.Tensor] = []
    for cand in candidates:
        rot = math.radians(float(cand.rot_2d_deg))
        c, s = math.cos(rot), math.sin(rot)
        s_rot = c * s_rel_base - s * r_rel_base
        r_rot = s * s_rel_base + c * r_rel_base

        axis = math.radians(float(cand.tilt_axis_deg))
        tilt_coord = math.cos(axis) * s_rot + math.sin(axis) * r_rot
        aa = float(cand.ap) + math.tan(math.radians(float(cand.tilt_deg))) * tilt_coord
        ss2 = s_center + s_rot
        rr2 = r_center + r_rot

        grid = torch.stack(
            [
                _normalize_grid(rr2, shape_r),
                _normalize_grid(ss2, shape_s),
                _normalize_grid(aa, shape_a),
            ],
            dim=-1,
        )
        grids.append(grid)

    return torch.stack(grids, dim=0)


def sample_axis_tilt_planes_torch(
    volume: np.ndarray | torch.Tensor,
    candidates: list[AxisTiltCandidate],
    *,
    side: str,
    width: int,
    height: int,
    mode: str,
    device: str | torch.device = "cuda",
    batch_size: int = 256,
) -> list[np.ndarray]:
    """Sample many axis/tilt planes on GPU.

    `mode` is `bilinear` for reference volume and `nearest` for annotation.
    """
    if not candidates:
        return []
    if mode not in {"bilinear", "nearest"}:
        raise ValueError(f"Unsupported mode={mode!r}")

    resolved_device = _resolve_device(device)
    if isinstance(volume, torch.Tensor):
        vol_t = volume.to(device=resolved_device, dtype=torch.float32)
        if vol_t.ndim == 3:
            vol_t = vol_t.unsqueeze(0).unsqueeze(0)
        elif vol_t.ndim != 5:
            raise ValueError(f"Expected 3D or 5D tensor, got {tuple(vol_t.shape)}")
        volume_shape = tuple(int(x) for x in vol_t.shape[-3:])
    else:
        vol_np = np.asarray(volume)
        volume_shape = tuple(int(x) for x in vol_np.shape)
        vol_t = _as_volume_tensor(vol_np, device=resolved_device, dtype=torch.float32)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            grid2d = build_axis_tilt_grid(
                chunk,
                volume_shape=volume_shape,  # type: ignore[arg-type]
                side=side,
                width=width,
                height=height,
                device=resolved_device,
            )
            grid = grid2d.unsqueeze(1)
            sampled = F.grid_sample(
                vol_t.expand(len(chunk), -1, -1, -1, -1),
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )
            planes = sampled[:, 0, 0].detach().cpu().numpy()
            outputs.extend([planes[i] for i in range(planes.shape[0])])
    return outputs


def sample_axis_tilt_plane_torch(
    volume: np.ndarray | torch.Tensor,
    *,
    ap: float,
    side: str,
    width: int,
    height: int,
    rot_2d_deg: float,
    tilt_axis_deg: float,
    tilt_deg: float,
    mode: str,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    return sample_axis_tilt_planes_torch(
        volume,
        [AxisTiltCandidate(ap=ap, rot_2d_deg=rot_2d_deg, tilt_axis_deg=tilt_axis_deg, tilt_deg=tilt_deg)],
        side=side,
        width=width,
        height=height,
        mode=mode,
        device=device,
        batch_size=1,
    )[0]


__all__ = [
    "AxisTiltCandidate",
    "build_axis_tilt_grid",
    "sample_axis_tilt_plane_torch",
    "sample_axis_tilt_planes_torch",
]
