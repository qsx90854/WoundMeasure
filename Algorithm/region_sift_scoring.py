"""Descriptor-only group scoring and reliable SIFT frame consistency.

Low-gradient anchors are ordinary descriptor rows, including when their
descriptor is zero.  A cell-balanced all-row term preserves their spatial
constraints even if the robust best-K term trims their mismatches.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def _frame_array(frames: Mapping[str, Any], key: str,
                 shape: tuple[int, ...], *, optional: bool = False) -> np.ndarray:
    if key not in frames:
        if optional:
            return np.zeros(shape, dtype=bool)
        raise ValueError(f"frames must contain {key}")
    result = np.asarray(frames[key], dtype=bool if optional else np.float64)
    # A single candidate may pass its frame dictionary directly.
    if len(shape) == 2 and shape[0] == 1 and result.shape == shape[1:]:
        result = result[np.newaxis, :]
    if result.shape != shape:
        raise ValueError(f"frame {key} must have shape {shape}, got {result.shape}")
    return result


def _circular_median(values: np.ndarray) -> float:
    """Robust angular center, including clusters straddling -180/+180.

    An L1 circular medoid selects an unwrap origin; an ordinary median then
    refines the center without pulling it toward isolated angular outliers.
    """
    differences = _wrap_degrees(values[:, None] - values[None, :])
    origin = values[int(np.argmin(np.sum(np.abs(differences), axis=1)))]
    return float(_wrap_degrees(
        np.asarray(origin + np.median(_wrap_degrees(values - origin)))))


def score_candidate_groups(
    distances: np.ndarray,
    point_metadata: Sequence[Mapping[str, Any]],
    left_frames: Mapping[str, Any],
    right_frames: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Score B candidate groups containing the same N fixed anchors.

    ``distances`` has shape (B, N).  Left frame arrays have shape (N,), right
    arrays (B, N), and use ``size_px``, ``angle_deg``, ``scale_reliable`` and
    ``orientation_reliable``.  Missing reliability flags mean no evidence.
    Metadata assigns each row to ``cell_row``/``cell_col`` (P belongs to the
    central cell).  Every occupied cell contributes equally to balanced_mean.

    group_score = (1-alpha) * best_K_mean + alpha * balanced_all_row_mean
    objective_without_epi = group_score + frame_penalty

    Frame consistency compares inter-image scale ratios and angle changes
    around their robust group centers, not around zero.  Each point therefore
    retains its own intrinsic scale/orientation and a common group change is
    allowed.  Each reliable residual uses min((residual/tolerance)^2, 1), so
    individual outliers cannot dominate.  The frame penalty averages available
    scale/orientation losses and uses descriptor units (512, or 1 when unit
    normalization is enabled).  Too few reliable pairs yield no frame penalty.
    """
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("distances must have shape (candidate_count, point_count>0)")
    if np.any(np.isnan(values)) or np.any(values < 0):
        raise ValueError("descriptor distances must be nonnegative and not NaN")
    batch_count, point_count = values.shape
    if len(point_metadata) != point_count:
        raise ValueError("point_metadata length must match descriptor row count")

    alpha = float(config.group_balance_weight)
    frame_weight = float(config.frame_consistency_weight)
    scale_tolerance = float(config.frame_scale_tolerance_log2)
    angle_tolerance = float(config.frame_angle_tolerance_deg)
    min_pairs = int(config.frame_min_reliable_pairs)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("group_balance_weight must be in [0, 1]")
    if not np.isfinite(frame_weight) or frame_weight < 0:
        raise ValueError("frame_consistency_weight must be finite and nonnegative")
    if not np.isfinite(scale_tolerance) or scale_tolerance <= 0:
        raise ValueError("frame_scale_tolerance_log2 must be finite and positive")
    if not np.isfinite(angle_tolerance) or angle_tolerance <= 0:
        raise ValueError("frame_angle_tolerance_deg must be finite and positive")
    if min_pairs < 3:
        raise ValueError("frame_min_reliable_pairs must be at least 3")

    if config.keep_best_count is None:
        ratio = float(config.keep_best_ratio)
        if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError("keep_best_ratio must be in (0, 1]")
        keep_count = int(round(point_count * ratio))
    else:
        keep_count = int(config.keep_best_count)
    keep_count = min(point_count, max(1, keep_count))
    order = np.argsort(values, axis=1, kind="stable")
    keep_mask = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(keep_mask, order[:, :keep_count], True, axis=1)
    trimmed_mean = np.mean(
        np.take_along_axis(values, order[:, :keep_count], axis=1), axis=1)

    try:
        cell_for_point = [(int(row["cell_row"]), int(row["cell_col"]))
                          for row in point_metadata]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("point metadata needs integer cell_row and cell_col") from exc
    cell_keys = sorted(set(cell_for_point))
    cell_indices = np.asarray([cell_keys.index(cell) for cell in cell_for_point])
    cell_mean_scores = np.stack([
        np.mean(values[:, cell_indices == index], axis=1)
        for index in range(len(cell_keys))], axis=1)
    balanced_mean = np.mean(cell_mean_scores, axis=1)
    # Branch at the endpoints to avoid 0 * inf for invalid candidates.
    if alpha == 0.0:
        group_score = trimmed_mean.copy()
    elif alpha == 1.0:
        group_score = balanced_mean.copy()
    else:
        group_score = (1.0 - alpha) * trimmed_mean + alpha * balanced_mean

    left_size = _frame_array(left_frames, "size_px", (point_count,))
    left_angle = _frame_array(left_frames, "angle_deg", (point_count,))
    right_size = _frame_array(right_frames, "size_px", values.shape)
    right_angle = _frame_array(right_frames, "angle_deg", values.shape)
    scale_pair_mask = (
        _frame_array(left_frames, "scale_reliable", (point_count,), optional=True)[None, :]
        & _frame_array(right_frames, "scale_reliable", values.shape, optional=True)
        & np.isfinite(left_size)[None, :] & (left_size[None, :] > 0)
        & np.isfinite(right_size) & (right_size > 0))
    orientation_pair_mask = (
        _frame_array(left_frames, "orientation_reliable", (point_count,), optional=True)[None, :]
        & _frame_array(right_frames, "orientation_reliable", values.shape, optional=True)
        & np.isfinite(left_angle)[None, :] & np.isfinite(right_angle))
    scale_pair_count = np.sum(scale_pair_mask, axis=1)
    orientation_pair_count = np.sum(orientation_pair_mask, axis=1)
    scale_center = np.full(batch_count, np.nan)
    angle_center = np.full(batch_count, np.nan)
    scale_residual = np.full(values.shape, np.nan)
    angle_residual = np.full(values.shape, np.nan)
    scale_loss = np.zeros(batch_count)
    angle_loss = np.zeros(batch_count)
    scale_evidence = scale_pair_count >= min_pairs
    angle_evidence = orientation_pair_count >= min_pairs
    for row in range(batch_count):
        if scale_evidence[row]:
            valid = scale_pair_mask[row]
            changes = np.log2(right_size[row, valid] / left_size[valid])
            center = float(np.median(changes))
            residual = changes - center
            scale_center[row] = center
            scale_residual[row, valid] = residual
            scale_loss[row] = np.mean(np.minimum(
                (residual / scale_tolerance) ** 2, 1.0))
        if angle_evidence[row]:
            valid = orientation_pair_mask[row]
            changes = _wrap_degrees(right_angle[row, valid] - left_angle[valid])
            center = _circular_median(changes)
            residual = _wrap_degrees(changes - center)
            angle_center[row] = center
            angle_residual[row, valid] = residual
            angle_loss[row] = np.mean(np.minimum(
                (residual / angle_tolerance) ** 2, 1.0))

    evidence_count = scale_evidence.astype(int) + angle_evidence.astype(int)
    combined_frame_loss = (scale_loss + angle_loss) / np.maximum(evidence_count, 1)
    descriptor_unit = 1.0 if config.normalize_descriptors else 512.0
    frame_penalty = frame_weight * descriptor_unit * combined_frame_loss
    return {
        "trimmed_mean": trimmed_mean,
        "balanced_mean": balanced_mean,
        "group_score": group_score,
        "keep_mask": keep_mask,
        "keep_count": keep_count,
        "cell_mean_scores": cell_mean_scores,
        "cell_keys": cell_keys,
        "point_cell_indices": cell_indices,
        "frame_penalty": frame_penalty,
        "objective_without_epi": group_score + frame_penalty,
        "frame_scale_center_log2": scale_center,
        "frame_angle_center_deg": angle_center,
        "frame_scale_residual_log2": scale_residual,
        "frame_angle_residual_deg": angle_residual,
        "frame_scale_pair_mask": scale_pair_mask,
        "frame_orientation_pair_mask": orientation_pair_mask,
        "frame_scale_pair_count": scale_pair_count,
        "frame_orientation_pair_count": orientation_pair_count,
        "frame_scale_loss": scale_loss,
        "frame_orientation_loss": angle_loss,
    }


__all__ = ["score_candidate_groups"]
