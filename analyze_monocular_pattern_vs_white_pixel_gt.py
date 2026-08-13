"""Analyze monocular ArUco-corner videos against a white-pixel ground truth.

The input is either one explicit pattern/white video pair, or a directory whose
files are paired by the names *_pattern and *_white.  Unlike the HBVCAM stereo
version, every decoded frame is one complete monocular image and is never split
into left and right views.

The two videos do not need to be synchronized.  Stable ArUco corners from the
pattern video establish four search centers.  Cross-intersection GT uses a
temporal-mean white frame and a multi-edge-level consensus; its model spread
and compatibility with the pattern geometry are reported explicitly.  Raw
measurements remain available for diagnosis, but a coherent four-corner
systematic offset is rejected from accuracy summaries.  Measurements are made
in the shared raw camera-pixel domain, so calibration, stereo RT, and baseline
data are neither required nor reported.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from aligned_roi_diagnostic import (
    ALIGNED_ROI_OUTPUT_SIZE_PX,
    BILATERAL_DIAMETER_PX,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    build_aligned_roi_spec,
    make_aligned_roi_frame,
    open_aligned_roi_writer,
)
from analyze_hbvcam_4pattern_subset_rt import AdaptiveArucoDetector
from analyze_hbvcam_aruco_corner_rt_stability import (
    ExcelFormula,
    excel_column,
    write_xlsx,
)
from analyze_hbvcam_pattern_vs_white_pixel_gt import (
    CROSS_PREVIEW_ARM_PX,
    CROSS_PREVIEW_JPEG_QUALITY,
    CROSS_PREVIEW_LINE_WIDTH_PX,
    CROSS_MAX_DIRECTION_CHANGE_DEG,
    CROSS_MAX_INTERSECTION_SHIFT_PX,
    CROSS_MAX_LINE_FIT_RMS_PX,
    CROSS_MAX_WIDTH_MAD_PX,
    CROSS_MIN_PROFILE_COUNT,
    CROSS_MIN_SIDE_PROFILE_COUNT,
    CROSS_PROFILE_ALONG_STEP_PX,
    CROSS_PROFILE_MAX_ARM_PX,
    DIAGNOSTIC_ROI_RADIUS_PX,
    DIAGNOSTIC_TILE_SIZE,
    GT_DETECTION_MODE,
    CrossIntersectionDetector,
    WhiteBlobDetector,
    add_distance_charts_to_xlsx,
    collect_video_pairs,
    crop_diagnostic_tile,
    open_diagnostic_writer,
    parse_distance,
)


DEFAULT_SUBPIX_STABILITY_MAX_RAW_PX = 2.0
DEFAULT_CROSS_CORE_EXCLUSION_PX = 11.0
DEFAULT_CROSS_ENSEMBLE_LEVELS = (0.30, 0.40, 0.50, 0.60)
DEFAULT_CROSS_MODEL_SPREAD_MAX_PX = 0.20
DEFAULT_CROSS_FRAME_CONSENSUS_MAX_PX = 0.20
GT_SYSTEMATIC_OFFSET_MIN_PX = 0.25
GT_SYSTEMATIC_OFFSET_SIGMA_MULTIPLIER = 3.0


class CrossIntersectionConsensusDetector:
    """Per-frame cross detector plus temporal-mean, multi-level consensus.

    A single half-height model is retained for frame-level diagnostics.  The
    fixed GT reference is detected on the temporal-mean white frame at several
    edge levels.  Profiles from both ends of each physical rectangle side are
    pooled into one global side fit before adjacent sides are intersected.
    Their coordinate median is the estimate, and their spread is reported as
    model uncertainty instead of silently treating one threshold as exact.
    """

    detector_mode = "cross_intersection_consensus"

    def __init__(
        self,
        search_radius_px: int,
        core_exclusion_px: float,
        edge_levels: tuple[float, ...],
        model_spread_max_px: float,
    ):
        levels = tuple(sorted({float(value) for value in edge_levels}))
        if len(levels) < 2:
            raise ValueError("cross consensus requires at least two edge levels")
        self.edge_levels = levels
        self.core_exclusion_px = float(core_exclusion_px)
        self.model_spread_max_px = float(model_spread_max_px)
        self.models = [
            CrossIntersectionDetector(
                search_radius_px,
                core_exclusion_px=self.core_exclusion_px,
                edge_level_ratio=level,
            )
            for level in levels
        ]
        self.primary_index = min(
            range(len(levels)), key=lambda index: abs(levels[index] - 0.5)
        )
        self.minimum_model_count = max(2, len(levels) // 2 + 1)

    def detect_four(self, frame: np.ndarray, search_centers):
        """Use the half-height-nearest model for per-frame measurements."""
        return self.models[self.primary_index].detect_four(frame, search_centers)

    @staticmethod
    def _unit(vector) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(2)
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            raise ValueError("degenerate side direction")
        return vector / length

    @classmethod
    def _fit_global_side(
        cls,
        detector: CrossIntersectionDetector,
        gray: np.ndarray,
        first_anchor: np.ndarray,
        second_anchor: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict]:
        """Pool profiles at both endpoints into one physical side line."""
        first_anchor = np.asarray(first_anchor, dtype=np.float64).reshape(2)
        second_anchor = np.asarray(second_anchor, dtype=np.float64).reshape(2)
        direction = cls._unit(second_anchor - first_anchor)
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        midpoint = (first_anchor + second_anchor) / 2.0
        points = []
        widths = []
        contrasts = []
        endpoint_ids = []
        arm_signs = []
        endpoint_raw_counts = []
        for endpoint_id, anchor in enumerate((first_anchor, second_anchor)):
            profiles, sampled_direction, sampled_normal = (
                detector._sample_profile_centers(gray, anchor, direction)
            )
            endpoint_raw_counts.append(len(profiles))
            for profile in profiles:
                point = (
                    anchor
                    + float(profile["along_px"]) * sampled_direction
                    + float(profile["center_offset_px"]) * sampled_normal
                )
                points.append(point)
                widths.append(float(profile["width_px"]))
                contrasts.append(float(profile["contrast_gray"]))
                endpoint_ids.append(endpoint_id)
                arm_signs.append(
                    -1 if float(profile["along_px"]) < 0 else 1
                )
        arm_signs = np.asarray(arm_signs, dtype=np.int32)
        endpoint_ids = np.asarray(endpoint_ids, dtype=np.int32)
        if any(
            count < CROSS_MIN_SIDE_PROFILE_COUNT
            for count in endpoint_raw_counts
        ):
            return None, {
                "failure_reason": (
                    "too few valid profiles at one global-side endpoint"
                ),
                "endpoint1_raw_profile_count": endpoint_raw_counts[0],
                "endpoint2_raw_profile_count": endpoint_raw_counts[1],
            }
        raw_arm_support = {
            (endpoint_id, arm_sign): int(
                np.count_nonzero(
                    (endpoint_ids == endpoint_id) & (arm_signs == arm_sign)
                )
            )
            for endpoint_id in (0, 1)
            for arm_sign in (-1, 1)
        }
        if any(
            count < CROSS_MIN_SIDE_PROFILE_COUNT
            for count in raw_arm_support.values()
        ):
            return None, {
                "failure_reason": (
                    "too few raw profiles on one global-side endpoint arm"
                ),
                "endpoint1_raw_profile_count": endpoint_raw_counts[0],
                "endpoint2_raw_profile_count": endpoint_raw_counts[1],
            }

        points = np.asarray(points, dtype=np.float64)
        along = (points - midpoint) @ direction
        offsets = (points - midpoint) @ normal
        design = np.column_stack((np.ones(len(points)), along))
        first_indices = np.where(endpoint_ids == 0)[0]
        second_indices = np.where(endpoint_ids == 1)[0]
        cross_endpoint_numerators = (
            offsets[second_indices, None] - offsets[None, first_indices]
        )
        cross_endpoint_denominators = (
            along[second_indices, None] - along[None, first_indices]
        )
        valid_slopes = np.abs(cross_endpoint_denominators) > 1e-6
        if not np.any(valid_slopes):
            return None, {
                "failure_reason": "global-side endpoint samples overlap",
                "endpoint1_raw_profile_count": endpoint_raw_counts[0],
                "endpoint2_raw_profile_count": endpoint_raw_counts[1],
            }
        slope = float(
            np.median(
                cross_endpoint_numerators[valid_slopes]
                / cross_endpoint_denominators[valid_slopes]
            )
        )
        coefficients = np.asarray(
            [float(np.median(offsets - slope * along)), slope],
            dtype=np.float64,
        )

        def endpoint_balanced_weights(modifiers: np.ndarray) -> np.ndarray:
            modifiers = np.asarray(modifiers, dtype=np.float64)
            weights = np.zeros(len(points), dtype=np.float64)
            target_sum = 0.5 * len(points)
            for endpoint_id in (0, 1):
                selected = endpoint_ids == endpoint_id
                modifier_sum = float(np.sum(modifiers[selected]))
                if modifier_sum > 1e-12:
                    weights[selected] = (
                        modifiers[selected] * target_sum / modifier_sum
                    )
            return weights

        weights = endpoint_balanced_weights(np.ones(len(points)))
        for _iteration in range(8):
            square_root_weights = np.sqrt(weights)
            coefficients = np.linalg.lstsq(
                design * square_root_weights[:, None],
                offsets * square_root_weights,
                rcond=None,
            )[0]
            residuals = offsets - design @ coefficients
            robust_scale = float(
                1.4826
                * np.median(np.abs(residuals - np.median(residuals)))
            ) + 1e-6
            huber_delta = max(0.12, 1.5 * robust_scale)
            weights = endpoint_balanced_weights(
                np.minimum(
                    1.0,
                    huber_delta / np.maximum(np.abs(residuals), 1e-9),
                )
            )

        residuals = offsets - design @ coefficients
        robust_scale = float(
            1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        ) + 1e-6
        residual_gate = max(0.35, 3.0 * robust_scale)
        inliers = np.abs(residuals) <= residual_gate
        endpoint_support = [
            int(np.count_nonzero(inliers & (endpoint_ids == endpoint_id)))
            for endpoint_id in (0, 1)
        ]
        if any(
            count < CROSS_MIN_SIDE_PROFILE_COUNT for count in endpoint_support
        ):
            return None, {
                "failure_reason": (
                    "too few fitted profiles at one global-side endpoint"
                ),
                "endpoint1_raw_profile_count": endpoint_raw_counts[0],
                "endpoint2_raw_profile_count": endpoint_raw_counts[1],
                "endpoint1_support_profiles": endpoint_support[0],
                "endpoint2_support_profiles": endpoint_support[1],
            }
        inlier_arm_support = {
            (endpoint_id, arm_sign): int(
                np.count_nonzero(
                    inliers
                    & (endpoint_ids == endpoint_id)
                    & (arm_signs == arm_sign)
                )
            )
            for endpoint_id in (0, 1)
            for arm_sign in (-1, 1)
        }
        if any(
            count < CROSS_MIN_SIDE_PROFILE_COUNT
            for count in inlier_arm_support.values()
        ):
            return None, {
                "failure_reason": (
                    "too few fitted profiles on one global-side endpoint arm"
                ),
                "endpoint1_raw_profile_count": endpoint_raw_counts[0],
                "endpoint2_raw_profile_count": endpoint_raw_counts[1],
                "endpoint1_support_profiles": endpoint_support[0],
                "endpoint2_support_profiles": endpoint_support[1],
            }

        selected_design = design[inliers]
        selected_offsets = offsets[inliers]
        selected_endpoint_ids = endpoint_ids[inliers]
        final_weights = np.zeros(np.count_nonzero(inliers), dtype=np.float64)
        target_sum = 0.5 * len(final_weights)
        for endpoint_id in (0, 1):
            selected = selected_endpoint_ids == endpoint_id
            final_weights[selected] = target_sum / np.count_nonzero(selected)
        square_root_weights = np.sqrt(final_weights)
        coefficients = np.linalg.lstsq(
            selected_design * square_root_weights[:, None],
            selected_offsets * square_root_weights,
            rcond=None,
        )[0]
        selected_residuals = selected_offsets - selected_design @ coefficients
        fit_rms = float(
            np.sqrt(
                np.average(
                    np.square(selected_residuals), weights=final_weights
                )
            )
        )
        direction_delta = float(np.degrees(np.arctan(coefficients[1])))
        selected_widths = np.asarray(widths, dtype=np.float64)[inliers]
        width_median = float(np.median(selected_widths))
        width_mad = float(
            1.4826 * np.median(np.abs(selected_widths - width_median))
        )
        diagnostics = {
            "endpoint1_raw_profile_count": endpoint_raw_counts[0],
            "endpoint2_raw_profile_count": endpoint_raw_counts[1],
            "endpoint1_support_profiles": endpoint_support[0],
            "endpoint2_support_profiles": endpoint_support[1],
            "support_profiles": int(np.count_nonzero(inliers)),
            "fit_rms_px": fit_rms,
            "width_median_px": width_median,
            "width_mad_px": width_mad,
            "contrast_median_gray": float(np.median(contrasts)),
            "direction_delta_deg": direction_delta,
        }
        if abs(direction_delta) > CROSS_MAX_DIRECTION_CHANGE_DEG:
            return None, {
                **diagnostics,
                "failure_reason": (
                    f"global direction correction {direction_delta:.2f} deg "
                    "exceeds gate"
                ),
            }
        if fit_rms > CROSS_MAX_LINE_FIT_RMS_PX:
            return None, {
                **diagnostics,
                "failure_reason": (
                    f"global side fit RMS {fit_rms:.2f} px exceeds gate"
                ),
            }
        if width_mad > CROSS_MAX_WIDTH_MAD_PX:
            return None, {
                **diagnostics,
                "failure_reason": (
                    f"global side width MAD {width_mad:.2f} px exceeds gate"
                ),
            }
        line_point = midpoint + coefficients[0] * normal
        line_direction = cls._unit(direction + coefficients[1] * normal)
        return (line_point, line_direction), diagnostics

    @classmethod
    def _detect_global_rectangle(
        cls,
        detector: CrossIntersectionDetector,
        frame: np.ndarray,
        search_centers,
    ) -> tuple[np.ndarray | None, list[dict]]:
        """Fit each of the four physical screen lines once, then intersect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        centers = np.asarray(search_centers, dtype=np.float64).reshape(4, 2)
        fitted_sides = []
        side_diagnostics = []
        for side_index in range(4):
            fitted, diagnostics = cls._fit_global_side(
                detector,
                gray,
                centers[side_index],
                centers[(side_index + 1) % 4],
            )
            fitted_sides.append(fitted)
            side_diagnostics.append(diagnostics)
        if any(side is None for side in fitted_sides):
            reason = "; ".join(
                f"side {index}: {diagnostics.get('failure_reason', 'failed')}"
                for index, (side, diagnostics) in enumerate(
                    zip(fitted_sides, side_diagnostics)
                )
                if side is None
            )
            return None, [
                {
                    "accepted": False,
                    "detector_mode": "cross_intersection_global_sides",
                    "failure_reason": reason,
                }
                for _corner_index in range(4)
            ]

        output_points = []
        output_diagnostics = []
        for corner_index in range(4):
            previous_side_index = (corner_index - 1) % 4
            following_side_index = corner_index
            first_point, first_direction = fitted_sides[previous_side_index]
            second_point, second_direction = fitted_sides[following_side_index]
            matrix = np.column_stack((first_direction, -second_direction))
            if abs(float(np.linalg.det(matrix))) < 0.15:
                return None, [
                    {
                        "accepted": False,
                        "detector_mode": "cross_intersection_global_sides",
                        "failure_reason": "adjacent global sides are nearly parallel",
                    }
                    for _corner_index in range(4)
                ]
            parameters = np.linalg.solve(matrix, second_point - first_point)
            intersection = first_point + parameters[0] * first_direction
            distance = float(np.linalg.norm(intersection - centers[corner_index]))
            intersection_shift_gate = min(
                CROSS_MAX_INTERSECTION_SHIFT_PX,
                detector.radius * 0.75,
            )
            if distance > intersection_shift_gate:
                return None, [
                    {
                        "accepted": False,
                        "detector_mode": "cross_intersection_global_sides",
                        "failure_reason": (
                            f"global intersection shift {distance:.2f} px "
                            "exceeds search gate"
                        ),
                    }
                    for _corner_index in range(4)
                ]
            previous_diagnostics = side_diagnostics[previous_side_index]
            following_diagnostics = side_diagnostics[following_side_index]
            diagnostics = {
                "accepted": True,
                "detector_mode": "cross_intersection_global_sides",
                "x": float(intersection[0]),
                "y": float(intersection[1]),
                "distance_from_search_center_px": distance,
                "line1_fit_rms_px": previous_diagnostics["fit_rms_px"],
                "line2_fit_rms_px": following_diagnostics["fit_rms_px"],
                "line_fit_rms_px": float(
                    np.sqrt(
                        0.5
                        * (
                            previous_diagnostics["fit_rms_px"] ** 2
                            + following_diagnostics["fit_rms_px"] ** 2
                        )
                    )
                ),
                "global_side_support_profiles": min(
                    previous_diagnostics["support_profiles"],
                    following_diagnostics["support_profiles"],
                ),
            }
            output_points.append(intersection)
            output_diagnostics.append(diagnostics)
        return np.stack(output_points), output_diagnostics

    def detect_consensus(self, frame: np.ndarray, search_centers):
        """Median global-side coordinates across edge-level models."""
        model_results = [
            self._detect_global_rectangle(detector, frame, search_centers)
            for detector in self.models
        ]
        output_points = []
        output_diagnostics = []
        for corner_index in range(4):
            accepted = []
            for model_index, (points, diagnostics) in enumerate(model_results):
                diagnostic = diagnostics[corner_index]
                if diagnostic.get("accepted"):
                    accepted.append(
                        (
                            model_index,
                            np.asarray(
                                [diagnostic["x"], diagnostic["y"]],
                                dtype=np.float64,
                            ),
                            diagnostic,
                        )
                    )
            accepted_model_indices = {entry[0] for entry in accepted}
            primary_accepted = self.primary_index in accepted_model_indices
            if (
                len(accepted) < self.minimum_model_count
                or not primary_accepted
            ):
                output_diagnostics.append(
                    {
                        "accepted": False,
                        "detector_mode": self.detector_mode,
                        "failure_reason": (
                            f"only {len(accepted)}/{len(self.models)} edge-level "
                            f"models accepted C{corner_index}; primary accepted="
                            f"{primary_accepted}"
                        ),
                        "model_count": len(accepted),
                        "model_requested_count": len(self.models),
                    }
                )
                output_points.append(None)
                continue

            values = np.stack([entry[1] for entry in accepted])
            point = np.median(values, axis=0)
            distances = np.linalg.norm(values - point, axis=1)
            spread_rms = float(np.sqrt(np.mean(np.square(distances))))
            spread_max = float(
                np.max(
                    np.linalg.norm(
                        values[:, None, :] - values[None, :, :],
                        axis=2,
                    )
                )
            )
            reference = min(
                accepted,
                key=lambda entry: abs(self.edge_levels[entry[0]] - 0.5),
            )[2]
            diagnostic = {
                **reference,
                "accepted": spread_max <= self.model_spread_max_px,
                "detector_mode": self.detector_mode,
                "x": float(point[0]),
                "y": float(point[1]),
                "model_count": len(accepted),
                "model_requested_count": len(self.models),
                "model_spread_rms_px": spread_rms,
                "model_spread_max_px": spread_max,
                "edge_levels": ",".join(
                    f"{self.edge_levels[entry[0]]:g}" for entry in accepted
                ),
                "core_exclusion_px": self.core_exclusion_px,
            }
            if not diagnostic["accepted"]:
                diagnostic["failure_reason"] = (
                    f"edge-level model pairwise disagreement {spread_max:.3f} "
                    "px exceeds "
                    f"{self.model_spread_max_px:.3f} px"
                )
                output_points.append(None)
            else:
                output_points.append(point)
            output_diagnostics.append(diagnostic)

        if any(point is None for point in output_points):
            return None, output_diagnostics
        consensus_points = np.stack(output_points)
        consensus_edges = np.roll(consensus_points, -1, axis=0) - consensus_points
        following_consensus_edges = np.roll(consensus_edges, -1, axis=0)
        consensus_crosses = (
            consensus_edges[:, 0] * following_consensus_edges[:, 1]
            - consensus_edges[:, 1] * following_consensus_edges[:, 0]
        )
        centers = np.asarray(search_centers, dtype=np.float64).reshape(4, 2)
        center_edges = np.roll(centers, -1, axis=0) - centers
        following_center_edges = np.roll(center_edges, -1, axis=0)
        center_crosses = (
            center_edges[:, 0] * following_center_edges[:, 1]
            - center_edges[:, 1] * following_center_edges[:, 0]
        )
        consensus_area = 0.5 * abs(
            float(
                np.sum(
                    consensus_points[:, 0]
                    * np.roll(consensus_points[:, 1], -1)
                    - consensus_points[:, 1]
                    * np.roll(consensus_points[:, 0], -1)
                )
            )
        )
        center_area = 0.5 * abs(
            float(
                np.sum(
                    centers[:, 0] * np.roll(centers[:, 1], -1)
                    - centers[:, 1] * np.roll(centers[:, 0], -1)
                )
            )
        )
        same_orientation = (
            np.all(consensus_crosses > 0) and np.all(center_crosses > 0)
        ) or (
            np.all(consensus_crosses < 0) and np.all(center_crosses < 0)
        )
        area_ratio = consensus_area / center_area if center_area > 1e-9 else math.nan
        if not same_orientation or not 0.5 <= area_ratio <= 2.0:
            reason = (
                "consensus corners fail convex/order/area geometry gate; "
                f"area ratio={area_ratio:.3f}"
            )
            return None, [
                {
                    **diagnostic,
                    "accepted": False,
                    "failure_reason": reason,
                    "quadrilateral_area_ratio": area_ratio,
                }
                for diagnostic in output_diagnostics
            ]
        for diagnostic in output_diagnostics:
            diagnostic["quadrilateral_area_ratio"] = area_ratio
        return consensus_points, output_diagnostics


def point_fields(prefix: str = "") -> list[str]:
    stem = f"{prefix}_" if prefix else ""
    return [
        f"{stem}c{corner_index}_{axis}_px"
        for corner_index in range(4)
        for axis in ("x", "y")
    ]


PATTERN_FRAME_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "frame_index",
    "frame_width_px",
    "frame_height_px",
    "marker_id",
    "status",
    "failure_reason",
    "subpixel_half_window",
    "subpixel_stability_max_raw_px",
    "subpixel_shift_rms_px",
    "corner_gradient_mean",
    "marker_laplacian_variance",
    "estimated_corner_uncertainty_px",
    *point_fields(),
    "marker_mean_side_length_px",
    "processing_time_ms",
]


WHITE_FRAME_FIELDS = [
    "distance_cm",
    "white_video_file",
    "gt_detection_mode",
    "frame_index",
    "frame_width_px",
    "frame_height_px",
    "status",
    "failure_reason",
    "detected_corner_count",
    *point_fields(),
    "diagnostic_video",
    "pattern_diagnostic_video",
    "gt_diagnostic_video",
    "diagnostic_roi_x_px",
    "diagnostic_roi_y_px",
    "diagnostic_roi_size_source_px",
    "diagnostic_roi_size_output_px",
    "cross_preview_jpg",
    "processing_time_ms",
]


FRAME_COMPARISON_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "frame_index",
    "pattern_status",
    "white_same_index_status",
    "gt_quality_status",
    "gt_quality_reason",
    "gt_approved_as_independent_reference",
    "pattern_mean_side_length_px",
    "pattern_vs_white_same_index_error_mean_px",
    "pattern_vs_white_same_index_error_rms_px",
    "pattern_vs_white_same_index_error_max_px",
    "pattern_vs_gt_median_error_mean_px",
    "pattern_vs_gt_median_error_rms_px",
    "pattern_vs_gt_median_error_max_px",
    "normalized_gt_error_mean_percent",
    "normalized_gt_error_rms_percent",
    "normalized_gt_error_max_percent",
]


CORNER_COMPARISON_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "gt_detection_mode",
    "frame_index",
    "corner_index",
    "gt_quality_status",
    "gt_approved_as_independent_reference",
    "pattern_x_px",
    "pattern_y_px",
    "white_same_index_x_px",
    "white_same_index_y_px",
    "gt_median_x_px",
    "gt_median_y_px",
    "pattern_vs_white_same_index_error_px",
    "pattern_vs_gt_median_error_px",
    "pattern_mean_side_length_px",
    "normalized_corner_error_percent",
    "white_same_index_vs_gt_median_error_px",
    "white_peak_gray",
    "white_background_gray",
    "white_contrast_gray",
    "white_blob_area_px",
    "white_distance_from_pattern_search_center_px",
    "cross_line1_raw_profile_count",
    "cross_line2_raw_profile_count",
    "cross_line1_support_profiles",
    "cross_line2_support_profiles",
    "cross_line1_negative_arm_profiles",
    "cross_line1_positive_arm_profiles",
    "cross_line2_negative_arm_profiles",
    "cross_line2_positive_arm_profiles",
    "cross_line1_fit_rms_px",
    "cross_line2_fit_rms_px",
    "cross_line1_width_median_px",
    "cross_line2_width_median_px",
    "cross_line1_contrast_median_gray",
    "cross_line2_contrast_median_gray",
    "cross_line1_direction_delta_deg",
    "cross_line2_direction_delta_deg",
    "cross_line_fit_rms_px",
]


GT_MEDIAN_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "marker_id",
    "gt_detection_mode",
    "gt_reference_provenance",
    "gt_reference_estimator",
    "gt_quality_status",
    "gt_quality_reason",
    "gt_approved_as_independent_reference",
    *point_fields("gt_median"),
    "valid_white_gt_frames",
    "frame_median_vs_consensus_rms_px",
    "cross_model_count_min",
    "cross_model_spread_rms_px",
    "cross_model_spread_max_px",
    "cross_ensemble_edge_levels",
    "cross_core_exclusion_px",
]


PAIR_SUMMARY_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "marker_id",
    "gt_detection_mode",
    "gt_reference_provenance",
    "gt_reference_estimator",
    "gt_quality_status",
    "gt_quality_reason",
    "gt_approved_as_independent_reference",
    "gt_inward_corner_count",
    "gt_mean_inward_offset_px",
    "gt_inward_offset_std_px",
    "gt_tangential_rms_px",
    "gt_pattern_mean_side_length_px",
    "gt_reference_mean_side_length_px",
    "gt_reference_minus_pattern_side_length_px",
    "gt_reference_to_pattern_side_length_ratio",
    "pattern_temporal_radial_jitter_px",
    "gt_temporal_radial_jitter_px",
    "gt_combined_uncertainty_px",
    "gt_quality_gate_px",
    "gt_frame_median_vs_consensus_rms_px",
    "gt_cross_model_count_min",
    "gt_cross_model_spread_rms_px",
    "gt_cross_model_spread_max_px",
    "requested_frames_per_video",
    "processed_pattern_frames",
    "valid_pattern_corner_frames",
    "processed_white_frames",
    "valid_white_corner_frames",
    "pattern_corner_coordinate_std_rms_px",
    "white_corner_coordinate_std_rms_px",
    "corner_error_sample_count",
    "pattern_vs_gt_corner_error_mean_px",
    "pattern_vs_gt_corner_error_rms_px",
    "pattern_vs_gt_corner_error_std_px",
    "pattern_vs_gt_corner_error_p95_px",
    "pattern_vs_gt_corner_error_max_px",
    "normalized_corner_error_sample_count",
    "normalized_corner_error_mean_percent",
    "normalized_corner_error_rms_percent",
    "normalized_corner_error_std_percent",
    "normalized_corner_error_p95_percent",
    "normalized_corner_error_max_percent",
    "same_index_corner_error_sample_count",
    "pattern_vs_white_same_index_error_mean_px",
    "pattern_vs_white_same_index_error_rms_px",
    "pattern_vs_white_same_index_error_std_px",
    "white_vs_gt_median_error_mean_px",
    "diagnostic_video",
    "pattern_diagnostic_video",
    "gt_diagnostic_video",
    "diagnostic_roi_x_px",
    "diagnostic_roi_y_px",
    "diagnostic_roi_size_source_px",
    "diagnostic_roi_size_output_px",
    "cross_preview_jpg",
]


DISTANCE_SUMMARY_FIELDS = [
    "distance_cm",
    "pair_count",
    "gt_quality_status",
    "independent_reference_pair_count",
    "unresolved_reference_pair_count",
    "pattern_frame_count",
    "valid_pattern_corner_frame_count",
    "corner_error_sample_count",
    "corner_error_mean_px",
    "corner_error_rms_px",
    "corner_error_std_px",
    "corner_error_p95_px",
    "corner_error_max_px",
    "normalized_corner_error_sample_count",
    "normalized_corner_error_mean_percent",
    "normalized_corner_error_rms_percent",
    "normalized_corner_error_std_percent",
    "normalized_corner_error_p95_percent",
    "normalized_corner_error_max_percent",
    "same_index_corner_error_sample_count",
    "same_index_corner_error_mean_px",
    "same_index_corner_error_rms_px",
]


DISTANCE_CHART_FIELDS = [
    "distance_cm",
    "corner_error_mean_px",
    "corner_error_std_px",
    "normalized_corner_error_mean_percent",
]


BATCH_ERROR_FIELDS = [
    "distance_cm",
    "pattern_video_file",
    "white_video_file",
    "error",
]


DIAGNOSTIC_FIELD_MAP = (
    ("peak_gray", "white_peak_gray"),
    ("background_gray", "white_background_gray"),
    ("contrast_gray", "white_contrast_gray"),
    ("blob_area_px", "white_blob_area_px"),
    (
        "distance_from_search_center_px",
        "white_distance_from_pattern_search_center_px",
    ),
    ("line1_raw_profile_count", "cross_line1_raw_profile_count"),
    ("line2_raw_profile_count", "cross_line2_raw_profile_count"),
    ("line1_support_profiles", "cross_line1_support_profiles"),
    ("line2_support_profiles", "cross_line2_support_profiles"),
    ("line1_negative_profiles", "cross_line1_negative_arm_profiles"),
    ("line1_positive_profiles", "cross_line1_positive_arm_profiles"),
    ("line2_negative_profiles", "cross_line2_negative_arm_profiles"),
    ("line2_positive_profiles", "cross_line2_positive_arm_profiles"),
    ("line1_fit_rms_px", "cross_line1_fit_rms_px"),
    ("line2_fit_rms_px", "cross_line2_fit_rms_px"),
    ("line1_width_median_px", "cross_line1_width_median_px"),
    ("line2_width_median_px", "cross_line2_width_median_px"),
    ("line1_contrast_median_gray", "cross_line1_contrast_median_gray"),
    ("line2_contrast_median_gray", "cross_line2_contrast_median_gray"),
    ("line1_direction_delta_deg", "cross_line1_direction_delta_deg"),
    ("line2_direction_delta_deg", "cross_line2_direction_delta_deg"),
    ("line_fit_rms_px", "cross_line_fit_rms_px"),
)


def parse_edge_levels(value: str) -> tuple[float, ...]:
    try:
        levels = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "edge levels must be comma-separated numbers"
        ) from exc
    if len(set(levels)) < 2 or any(not 0.05 <= level <= 0.95 for level in levels):
        raise argparse.ArgumentTypeError(
            "provide at least two unique edge levels between 0.05 and 0.95"
        )
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare monocular ArUco subpixel corners with a white-pixel or "
            "line-intersection GT. INPUT may be a folder of paired "
            "*_pattern/*_white videos, or one pattern video followed by its "
            "white video."
        )
    )
    parser.add_argument("input_path", help="Input folder, or one pattern video")
    parser.add_argument(
        "white_video",
        nargs="?",
        help="White/cross GT video when INPUT is one explicit pattern video",
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--marker-id", type=int)
    parser.add_argument(
        "--aruco-initial-refinement",
        choices=("none", "subpix", "contour", "apriltag"),
        default="contour",
        help="OpenCV refinement used before adaptive cornerSubPix (default: contour)",
    )
    parser.add_argument(
        "--subpixel-stability-max-raw-px",
        type=float,
        default=DEFAULT_SUBPIX_STABILITY_MAX_RAW_PX,
    )
    parser.add_argument("--white-search-radius-px", type=int, default=20)
    parser.add_argument("--white-threshold-ratio", type=float, default=0.5)
    parser.add_argument("--white-min-contrast-gray", type=float, default=30.0)
    parser.add_argument("--white-min-area-px", type=int, default=2)
    parser.add_argument("--white-max-area-px", type=int, default=300)
    parser.add_argument(
        "--gt-mode",
        choices=("white_blob_centroid", "cross_intersection"),
        default=GT_DETECTION_MODE,
    )
    parser.add_argument(
        "--cross-core-exclusion-px",
        type=float,
        default=DEFAULT_CROSS_CORE_EXCLUSION_PX,
        help=(
            "Distance from the crossing omitted from line profiles; 11 px "
            "avoids the strongest crossing-core PSF contamination in these "
            "videos (default: 11)"
        ),
    )
    parser.add_argument(
        "--cross-ensemble-edge-levels",
        type=parse_edge_levels,
        default=DEFAULT_CROSS_ENSEMBLE_LEVELS,
        help=(
            "Comma-separated contrast levels used on the temporal-mean GT "
            "frame (default: 0.30,0.40,0.50,0.60)"
        ),
    )
    parser.add_argument(
        "--cross-model-spread-max-px",
        type=float,
        default=DEFAULT_CROSS_MODEL_SPREAD_MAX_PX,
        help="Reject GT when edge-level models disagree beyond this many pixels",
    )
    parser.add_argument(
        "--cross-frame-consensus-max-px",
        type=float,
        default=DEFAULT_CROSS_FRAME_CONSENSUS_MAX_PX,
        help=(
            "Reject the temporal reference when it differs from the accepted-"
            "frame median by more than this RMS distance"
        ),
    )
    parser.add_argument(
        "--gt-reference-provenance",
        choices=("unverified", "externally_validated"),
        default="unverified",
        help=(
            "Whether target geometry was independently validated. Unverified "
            "references remain diagnostic and are not accuracy ground truth."
        ),
    )
    parser.add_argument(
        "--diagnostic-roi-radius-px",
        type=int,
        default=DIAGNOSTIC_ROI_RADIUS_PX,
        help=(
            "Source-pixel margin added on each side of the stable Pattern "
            "marker bounding box (default: 10; a 50 px marker becomes an "
            "approximately 70 px square crop)"
        ),
    )
    parser.add_argument(
        "--cross-arm-px",
        type=int,
        default=5,
        help="Half-length of the red marker in diagnostics",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--output", help="Output .xlsx path")
    parser.add_argument(
        "--diagnostic-video-dir",
        help="Folder for aligned Pattern/GT ROI AVI and cross-preview JPG files",
    )
    parser.add_argument(
        "--no-diagnostic-video",
        action="store_true",
        help="Skip aligned ROI AVI generation; cross preview JPG is still saved",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.subpixel_stability_max_raw_px <= 0:
        parser.error("--subpixel-stability-max-raw-px must be positive")
    if args.white_search_radius_px < 3:
        parser.error("--white-search-radius-px must be at least 3")
    if not 0.0 < args.cross_core_exclusion_px < CROSS_PROFILE_MAX_ARM_PX:
        parser.error(
            "--cross-core-exclusion-px must be positive and below "
            f"{CROSS_PROFILE_MAX_ARM_PX:g}"
        )
    if args.gt_mode == "cross_intersection":
        maximum_arm = min(
            CROSS_PROFILE_MAX_ARM_PX,
            args.white_search_radius_px - 2.0,
        )
        positions_per_arm = (
            int(
                math.floor(
                    (maximum_arm - args.cross_core_exclusion_px)
                    / CROSS_PROFILE_ALONG_STEP_PX
                )
            )
            + 1
            if maximum_arm >= args.cross_core_exclusion_px
            else 0
        )
        if (
            positions_per_arm < CROSS_MIN_SIDE_PROFILE_COUNT
            or 2 * positions_per_arm < CROSS_MIN_PROFILE_COUNT
        ):
            parser.error(
                "cross settings leave too few off-core line profiles; "
                "increase --white-search-radius-px or reduce "
                "--cross-core-exclusion-px (defaults require radius >= 16)"
            )
    if args.cross_model_spread_max_px <= 0:
        parser.error("--cross-model-spread-max-px must be positive")
    if args.cross_frame_consensus_max_px <= 0:
        parser.error("--cross-frame-consensus-max-px must be positive")
    if not 0.05 <= args.white_threshold_ratio <= 0.95:
        parser.error("--white-threshold-ratio must be between 0.05 and 0.95")
    if args.white_min_contrast_gray <= 0:
        parser.error("--white-min-contrast-gray must be positive")
    if args.white_min_area_px <= 0 or args.white_max_area_px < args.white_min_area_px:
        parser.error("white blob area limits are invalid")
    if args.diagnostic_roi_radius_px < 3:
        parser.error("--diagnostic-roi-radius-px must be at least 3")
    if args.cross_arm_px < 1:
        parser.error("--cross-arm-px must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    return args


def blank_row(fields: list[str]) -> dict:
    return {field: None for field in fields}


def add_points(row: dict, points, prefix: str = "") -> None:
    if points is None:
        return
    stem = f"{prefix}_" if prefix else ""
    for corner_index, (x, y) in enumerate(
        np.asarray(points, dtype=np.float64).reshape(4, 2)
    ):
        row[f"{stem}c{corner_index}_x_px"] = float(x)
        row[f"{stem}c{corner_index}_y_px"] = float(y)


def finite_number(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numeric_values(rows: list[dict], field: str) -> list[float]:
    output = []
    for row in rows:
        value = finite_number(row.get(field))
        if value is not None:
            output.append(value)
    return output


def describe(values) -> dict:
    clean = [
        number
        for value in values
        if (number := finite_number(value)) is not None
    ]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "rms": None,
            "std": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "std": float(np.std(array, ddof=1)) if array.size >= 2 else None,
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def marker_mean_side_length(points) -> float | None:
    if points is None:
        return None
    values = np.asarray(points, dtype=np.float64).reshape(4, 2)
    lengths = np.linalg.norm(np.roll(values, -1, axis=0) - values, axis=1)
    return float(np.mean(lengths))


def discover_marker_id(
    pattern_video: Path,
    detector: AdaptiveArucoDetector,
    requested_frames: int,
) -> int:
    capture = cv2.VideoCapture(str(pattern_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open pattern video: {pattern_video}")
    counts = Counter()
    try:
        for _frame_index in range(min(requested_frames, 30)):
            ok, frame = capture.read()
            if not ok:
                break
            markers, _diagnostics = detector.detect(frame)
            counts.update(markers.keys())
    finally:
        capture.release()
    if not counts:
        raise RuntimeError("No stable ArUco ID was found in the pattern video")
    marker_id, count = counts.most_common(1)[0]
    print(f"Automatically selected ArUco ID {marker_id} ({count} discovery frames)")
    return int(marker_id)


def process_pattern_video(
    video: Path,
    requested_frames: int,
    marker_id: int,
    detector: AdaptiveArucoDetector,
    progress_every: int,
) -> tuple[list[dict], dict[int, np.ndarray], np.ndarray, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open pattern video: {video}")
    rows = []
    corners_by_frame = {}
    frame_size = None
    distance = parse_distance(video)
    try:
        for frame_index in range(requested_frames):
            ok, frame = capture.read()
            if not ok:
                break
            current_size = (int(frame.shape[1]), int(frame.shape[0]))
            if frame_size is None:
                frame_size = current_size
            elif current_size != frame_size:
                raise RuntimeError(
                    f"Pattern frame size changed from {frame_size} to {current_size}"
                )
            started = time.perf_counter()
            row = blank_row(PATTERN_FRAME_FIELDS)
            row.update(
                {
                    "distance_cm": distance,
                    "pattern_video_file": str(video),
                    "frame_index": frame_index,
                    "frame_width_px": current_size[0],
                    "frame_height_px": current_size[1],
                    "marker_id": marker_id,
                    "status": "FAILED",
                    "failure_reason": "",
                }
            )
            try:
                markers, diagnostics = detector.detect(frame)
                info = diagnostics.get(marker_id, {})
                for source, target in (
                    ("half_window", "subpixel_half_window"),
                    ("stability_max_raw_px", "subpixel_stability_max_raw_px"),
                    ("subpixel_shift_rms_px", "subpixel_shift_rms_px"),
                    ("corner_gradient_mean", "corner_gradient_mean"),
                    ("marker_laplacian_variance", "marker_laplacian_variance"),
                    (
                        "estimated_corner_uncertainty_px",
                        "estimated_corner_uncertainty_px",
                    ),
                ):
                    row[target] = info.get(source)
                if marker_id not in markers:
                    if info:
                        raise RuntimeError(
                            f"ArUco ID {marker_id} subpixel corners were rejected"
                        )
                    raise RuntimeError(f"ArUco ID {marker_id} was not detected")
                points = np.asarray(markers[marker_id], dtype=np.float64).reshape(4, 2)
                corners_by_frame[frame_index] = points
                add_points(row, points)
                row["marker_mean_side_length_px"] = marker_mean_side_length(points)
                row["status"] = "OK"
            except Exception as exc:
                row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            row["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
            rows.append(row)
            if progress_every > 0 and (frame_index + 1) % progress_every == 0:
                print(f"  Pattern F{frame_index:03d} complete")
    finally:
        capture.release()
    if not rows or frame_size is None:
        raise RuntimeError("Pattern video had no decodable frame")
    if not corners_by_frame:
        raise RuntimeError("No valid monocular ArUco corners were obtained")
    search_centers = np.median(
        np.stack(list(corners_by_frame.values())), axis=0
    )
    return rows, corners_by_frame, search_centers, frame_size


def corner_panel(
    frame: np.ndarray,
    search_centers: np.ndarray,
    frame_index: int,
    label_prefix: str,
    display_radius: int,
    cross_arm: int,
    points: np.ndarray | None = None,
    results: list[dict] | None = None,
) -> np.ndarray:
    tiles = []
    for corner_index, center in enumerate(search_centers):
        if results is not None:
            result = dict(results[corner_index])
            # Draw the floating-point GT exactly like the ArUco point.  Painting
            # all floor/ceil source pixels produced a large red square after
            # nearest-neighbor enlargement and visually hid the intersection.
            result["marker_style"] = "cross"
        else:
            result = {"accepted": points is not None, "marker_style": "cross"}
            if points is not None:
                result["x"] = float(points[corner_index, 0])
                result["y"] = float(points[corner_index, 1])
        result["display_radius_px"] = display_radius
        result["cross_arm_px"] = cross_arm
        tiles.append(
            crop_diagnostic_tile(
                frame,
                center,
                result,
                f"{label_prefix} C{corner_index} F{frame_index}",
            )
        )
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


def save_cross_intersection_preview(
    frame: np.ndarray,
    points: np.ndarray,
    path: Path,
    arm_px: int = CROSS_PREVIEW_ARM_PX,
    line_width_px: int = CROSS_PREVIEW_LINE_WIDTH_PX,
) -> None:
    preview = frame.copy()
    for x, y in np.asarray(points, dtype=np.float64).reshape(4, 2):
        px, py = int(round(float(x))), int(round(float(y)))
        cv2.line(
            preview,
            (px - arm_px, py),
            (px + arm_px, py),
            (0, 0, 255),
            line_width_px,
            cv2.LINE_8,
        )
        cv2.line(
            preview,
            (px, py - arm_px),
            (px, py + arm_px),
            (0, 0, 255),
            line_width_px,
            cv2.LINE_8,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path),
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, CROSS_PREVIEW_JPEG_QUALITY],
    ):
        raise RuntimeError(f"Could not save cross-intersection preview JPG: {path}")


def process_white_video(
    pattern_video: Path,
    white_video: Path,
    requested_frames: int,
    gt_detector,
    search_centers: np.ndarray,
    pattern_corners: dict[int, np.ndarray],
    expected_frame_size: tuple[int, int],
    pattern_diagnostic_video: Path | None,
    gt_diagnostic_video: Path | None,
    artifact_dir: Path,
    artifact_prefix: str,
    diagnostic_roi_radius: int,
    cross_arm: int,
    frame_consensus_max_px: float,
    progress_every: int,
) -> tuple[
    list[dict],
    dict[int, np.ndarray],
    dict[int, list[dict]],
    np.ndarray,
    dict,
    Path | None,
]:
    white_capture = cv2.VideoCapture(str(white_video))
    if not white_capture.isOpened():
        raise RuntimeError(f"Could not open white video: {white_video}")
    pattern_capture = None
    pattern_writer = None
    gt_writer = None
    roi_spec = build_aligned_roi_spec(search_centers, diagnostic_roi_radius)
    if (pattern_diagnostic_video is None) != (gt_diagnostic_video is None):
        white_capture.release()
        raise ValueError("Pattern and GT diagnostic paths must both be set or both be None")
    if pattern_diagnostic_video is not None and gt_diagnostic_video is not None:
        pattern_capture = cv2.VideoCapture(str(pattern_video))
        if not pattern_capture.isOpened():
            white_capture.release()
            raise RuntimeError(f"Could not reopen pattern video: {pattern_video}")
        diagnostic_fps = float(white_capture.get(cv2.CAP_PROP_FPS))
        pattern_writer = open_aligned_roi_writer(
            pattern_diagnostic_video,
            diagnostic_fps,
            (ALIGNED_ROI_OUTPUT_SIZE_PX, ALIGNED_ROI_OUTPUT_SIZE_PX),
        )
        gt_writer = open_aligned_roi_writer(
            gt_diagnostic_video,
            diagnostic_fps,
            (ALIGNED_ROI_OUTPUT_SIZE_PX, ALIGNED_ROI_OUTPUT_SIZE_PX),
        )

    rows = []
    corners_by_frame = {}
    diagnostics_by_frame = {}
    cross_preview_path = None
    temporal_mean_frame = None
    decoded_frame_count = 0
    distance = parse_distance(white_video)
    detection_mode = getattr(
        gt_detector,
        "detector_mode",
        (
            "cross_intersection"
            if isinstance(gt_detector, CrossIntersectionDetector)
            else "white_blob_centroid"
        ),
    )
    try:
        for frame_index in range(requested_frames):
            white_ok, frame = white_capture.read()
            if not white_ok:
                break
            current_size = (int(frame.shape[1]), int(frame.shape[0]))
            if current_size != expected_frame_size:
                raise RuntimeError(
                    "Pattern/white frame-size mismatch: "
                    f"{expected_frame_size} versus {current_size}"
                )
            frame_float = frame.astype(np.float32)
            decoded_frame_count += 1
            if temporal_mean_frame is None:
                temporal_mean_frame = frame_float.copy()
            else:
                temporal_mean_frame += (
                    frame_float - temporal_mean_frame
                ) / decoded_frame_count
            pattern_ok = False
            pattern_frame = None
            if pattern_capture is not None:
                pattern_ok, pattern_frame = pattern_capture.read()

            started = time.perf_counter()
            row = blank_row(WHITE_FRAME_FIELDS)
            row.update(
                {
                    "distance_cm": distance,
                    "white_video_file": str(white_video),
                    "gt_detection_mode": detection_mode,
                    "frame_index": frame_index,
                    "frame_width_px": current_size[0],
                    "frame_height_px": current_size[1],
                    "status": "FAILED",
                    "failure_reason": "",
                    "detected_corner_count": 0,
                    "pattern_diagnostic_video": (
                        str(pattern_diagnostic_video)
                        if pattern_diagnostic_video is not None
                        else None
                    ),
                    "gt_diagnostic_video": (
                        str(gt_diagnostic_video)
                        if gt_diagnostic_video is not None
                        else None
                    ),
                    "diagnostic_roi_x_px": roi_spec.x0,
                    "diagnostic_roi_y_px": roi_spec.y0,
                    "diagnostic_roi_size_source_px": roi_spec.size_px,
                    "diagnostic_roi_size_output_px": ALIGNED_ROI_OUTPUT_SIZE_PX,
                }
            )
            points = None
            try:
                points, results = gt_detector.detect_four(frame, search_centers)
                diagnostics_by_frame[frame_index] = results
                row["detected_corner_count"] = sum(
                    bool(item.get("accepted")) for item in results
                )
                if points is None:
                    row["failure_reason"] = "; ".join(
                        f"C{index}: {item.get('failure_reason', 'missing')}"
                        for index, item in enumerate(results)
                        if not item.get("accepted")
                    )
                else:
                    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
                    corners_by_frame[frame_index] = points
                    add_points(row, points)
                    row["status"] = "OK"
            except Exception as exc:
                results = [
                    {
                        "accepted": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                    for _ in range(4)
                ]
                diagnostics_by_frame[frame_index] = results
                row["failure_reason"] = f"{type(exc).__name__}: {exc}"

            if pattern_writer is not None and gt_writer is not None:
                display_pattern = (
                    pattern_frame
                    if pattern_ok and pattern_frame is not None
                    else np.zeros_like(frame)
                )
                pattern_roi_frame = make_aligned_roi_frame(
                    display_pattern,
                    roi_spec,
                    pattern_corners.get(frame_index),
                    ALIGNED_ROI_OUTPUT_SIZE_PX,
                    cross_arm,
                )
                gt_roi_frame = make_aligned_roi_frame(
                    frame,
                    roi_spec,
                    points,
                    ALIGNED_ROI_OUTPUT_SIZE_PX,
                    cross_arm,
                )
                pattern_writer.write(pattern_roi_frame)
                gt_writer.write(gt_roi_frame)
            if cross_preview_path is not None:
                row["cross_preview_jpg"] = str(cross_preview_path)
            row["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
            rows.append(row)
            if progress_every > 0 and (frame_index + 1) % progress_every == 0:
                print(f"  White GT F{frame_index:03d} complete")
    finally:
        if pattern_writer is not None:
            pattern_writer.release()
        if gt_writer is not None:
            gt_writer.release()
        if pattern_capture is not None:
            pattern_capture.release()
        white_capture.release()
    if not rows:
        raise RuntimeError("White GT video had no decodable frame")
    frame_median = (
        np.median(np.stack(list(corners_by_frame.values())), axis=0)
        if corners_by_frame
        else None
    )
    reference_info = {
        "estimator": "accepted_frame_coordinate_median",
        "frame_median": frame_median,
        "frame_median_vs_consensus_rms_px": 0.0,
        "diagnostics": [],
    }
    gt_reference = frame_median
    if hasattr(gt_detector, "detect_consensus"):
        if temporal_mean_frame is None:
            raise RuntimeError("Could not form temporal-mean GT frame")
        gt_reference, reference_diagnostics = gt_detector.detect_consensus(
            temporal_mean_frame,
            search_centers,
        )
        if gt_reference is None:
            reasons = "; ".join(
                f"C{index}: {item.get('failure_reason', 'missing')}"
                for index, item in enumerate(reference_diagnostics)
                if not item.get("accepted")
            )
            raise RuntimeError(f"Temporal-mean GT consensus failed: {reasons}")
        gt_reference = np.asarray(gt_reference, dtype=np.float64).reshape(4, 2)
        frame_consensus_rms = (
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            np.square(frame_median - gt_reference), axis=1
                        )
                    )
                )
            )
            if frame_median is not None
            else None
        )
        if (
            frame_consensus_rms is not None
            and frame_consensus_rms > frame_consensus_max_px
        ):
            raise RuntimeError(
                "Temporal-mean GT differs from the accepted-frame median by "
                f"{frame_consensus_rms:.3f} px RMS, exceeding "
                f"{frame_consensus_max_px:.3f} px"
            )
        reference_info = {
            "estimator": (
                "temporal_mean_multi_edge_level_global_side_consensus"
            ),
            "frame_median": frame_median,
            "frame_median_vs_consensus_rms_px": frame_consensus_rms,
            "diagnostics": reference_diagnostics,
        }
        cross_preview_path = artifact_dir / (
            f"{artifact_prefix}_cross_intersection_temporal_consensus.jpg"
        )
        preview_frame = np.clip(
            np.rint(temporal_mean_frame), 0, 255
        ).astype(np.uint8)
        save_cross_intersection_preview(
            preview_frame,
            gt_reference,
            cross_preview_path,
        )
    elif gt_reference is None:
        raise RuntimeError("No GT-video frame had all four accepted points")
    if cross_preview_path is not None:
        for row in rows:
            row["cross_preview_jpg"] = str(cross_preview_path)
    return (
        rows,
        corners_by_frame,
        diagnostics_by_frame,
        gt_reference,
        reference_info,
        cross_preview_path,
    )


def remove_file_if_exists(path: Path | None) -> None:
    """Remove one partial per-pair artifact after that pair fails."""
    if path is not None and path.is_file():
        path.unlink()


def remove_cross_previews(artifact_dir: Path, artifact_prefix: str) -> None:
    """Remove previews made by a pair that subsequently failed."""
    if not artifact_dir.is_dir():
        return
    pattern = f"{artifact_prefix}_cross_intersection*.jpg"
    for path in artifact_dir.glob(pattern):
        if path.is_file() and path.parent == artifact_dir:
            path.unlink()


def errors_between(first, second) -> np.ndarray:
    return np.linalg.norm(
        np.asarray(first, dtype=np.float64).reshape(4, 2)
        - np.asarray(second, dtype=np.float64).reshape(4, 2),
        axis=1,
    )


def put_error_stats(row: dict, prefix: str, errors) -> None:
    summary = describe(errors)
    row[f"{prefix}_mean_px"] = summary["mean"]
    row[f"{prefix}_rms_px"] = summary["rms"]
    row[f"{prefix}_max_px"] = summary["max"]


def build_comparison_rows(
    pattern_video: Path,
    white_video: Path,
    gt_detection_mode: str,
    pattern_rows: list[dict],
    white_rows: list[dict],
    pattern_corners: dict[int, np.ndarray],
    white_corners: dict[int, np.ndarray],
    white_diagnostics: dict[int, list[dict]],
    gt_median: np.ndarray,
    gt_quality: dict,
) -> tuple[list[dict], list[dict]]:
    comparison_rows = []
    corner_rows = []
    white_row_lookup = {int(row["frame_index"]): row for row in white_rows}
    distance = parse_distance(pattern_video)
    for pattern_row in pattern_rows:
        frame_index = int(pattern_row["frame_index"])
        white_row = white_row_lookup.get(frame_index)
        pattern_points = pattern_corners.get(frame_index)
        white_points = white_corners.get(frame_index)
        side_length = marker_mean_side_length(pattern_points)
        frame_row = blank_row(FRAME_COMPARISON_FIELDS)
        frame_row.update(
            {
                "distance_cm": distance,
                "pattern_video_file": str(pattern_video),
                "white_video_file": str(white_video),
                "frame_index": frame_index,
                "pattern_status": pattern_row["status"],
                "white_same_index_status": (
                    white_row["status"] if white_row is not None else None
                ),
                "gt_quality_status": gt_quality["status"],
                "gt_quality_reason": gt_quality["reason"],
                "gt_approved_as_independent_reference": gt_quality[
                    "approved_as_independent_reference"
                ],
                "pattern_mean_side_length_px": side_length,
            }
        )
        if pattern_points is not None:
            gt_errors = errors_between(pattern_points, gt_median)
            put_error_stats(frame_row, "pattern_vs_gt_median_error", gt_errors)
            if side_length is not None and side_length > 0:
                normalized = describe(gt_errors / side_length * 100.0)
                frame_row["normalized_gt_error_mean_percent"] = normalized["mean"]
                frame_row["normalized_gt_error_rms_percent"] = normalized["rms"]
                frame_row["normalized_gt_error_max_percent"] = normalized["max"]
        if pattern_points is not None and white_points is not None:
            put_error_stats(
                frame_row,
                "pattern_vs_white_same_index_error",
                errors_between(pattern_points, white_points),
            )
        comparison_rows.append(frame_row)

        diagnostics = white_diagnostics.get(frame_index, [{} for _ in range(4)])
        for corner_index in range(4):
            corner_row = blank_row(CORNER_COMPARISON_FIELDS)
            corner_row.update(
                {
                    "distance_cm": distance,
                    "pattern_video_file": str(pattern_video),
                    "white_video_file": str(white_video),
                    "gt_detection_mode": gt_detection_mode,
                    "frame_index": frame_index,
                    "corner_index": corner_index,
                    "gt_quality_status": gt_quality["status"],
                    "gt_approved_as_independent_reference": gt_quality[
                        "approved_as_independent_reference"
                    ],
                    "gt_median_x_px": float(gt_median[corner_index, 0]),
                    "gt_median_y_px": float(gt_median[corner_index, 1]),
                }
            )
            if pattern_points is not None:
                pattern_point = pattern_points[corner_index]
                gt_error = float(
                    np.linalg.norm(pattern_point - gt_median[corner_index])
                )
                corner_row["pattern_x_px"] = float(pattern_point[0])
                corner_row["pattern_y_px"] = float(pattern_point[1])
                corner_row["pattern_vs_gt_median_error_px"] = gt_error
                corner_row["pattern_mean_side_length_px"] = side_length
                if side_length is not None and side_length > 0:
                    corner_row["normalized_corner_error_percent"] = (
                        gt_error / side_length * 100.0
                    )
            if white_points is not None:
                white_point = white_points[corner_index]
                corner_row["white_same_index_x_px"] = float(white_point[0])
                corner_row["white_same_index_y_px"] = float(white_point[1])
                corner_row["white_same_index_vs_gt_median_error_px"] = float(
                    np.linalg.norm(white_point - gt_median[corner_index])
                )
                if pattern_points is not None:
                    corner_row["pattern_vs_white_same_index_error_px"] = float(
                        np.linalg.norm(pattern_points[corner_index] - white_point)
                    )
            diagnostic = (
                diagnostics[corner_index]
                if corner_index < len(diagnostics)
                else {}
            )
            for source, target in DIAGNOSTIC_FIELD_MAP:
                corner_row[target] = diagnostic.get(source)
            corner_rows.append(corner_row)
    return comparison_rows, corner_rows


def coordinate_std_rms(corners_by_frame: dict[int, np.ndarray]) -> float | None:
    if len(corners_by_frame) < 2:
        return None
    coordinate_stds = np.std(
        np.stack(list(corners_by_frame.values())),
        axis=0,
        ddof=1,
    ).reshape(-1)
    return float(np.sqrt(np.mean(np.square(coordinate_stds))))


def summarize_gt_reference_info(reference_info: dict) -> dict:
    diagnostics = [
        item
        for item in reference_info.get("diagnostics", [])
        if item.get("accepted")
    ]
    model_counts = [
        int(item["model_count"])
        for item in diagnostics
        if finite_number(item.get("model_count")) is not None
    ]
    spread_rms_values = [
        float(item["model_spread_rms_px"])
        for item in diagnostics
        if finite_number(item.get("model_spread_rms_px")) is not None
    ]
    spread_max_values = [
        float(item["model_spread_max_px"])
        for item in diagnostics
        if finite_number(item.get("model_spread_max_px")) is not None
    ]
    edge_levels = sorted(
        {
            str(item["edge_levels"])
            for item in diagnostics
            if item.get("edge_levels")
        }
    )
    core_values = [
        float(item["core_exclusion_px"])
        for item in diagnostics
        if finite_number(item.get("core_exclusion_px")) is not None
    ]
    return {
        "estimator": reference_info.get("estimator"),
        "frame_median_vs_consensus_rms_px": finite_number(
            reference_info.get("frame_median_vs_consensus_rms_px")
        ),
        "model_count_min": min(model_counts) if model_counts else None,
        "model_spread_rms_px": (
            float(
                np.sqrt(
                    np.mean(np.square(np.asarray(spread_rms_values)))
                )
            )
            if spread_rms_values
            else None
        ),
        "model_spread_max_px": (
            max(spread_max_values) if spread_max_values else None
        ),
        "edge_levels": " | ".join(edge_levels) if edge_levels else None,
        "core_exclusion_px": (
            float(np.mean(core_values)) if core_values else None
        ),
    }


def audit_gt_geometry(
    pattern_reference: np.ndarray,
    pattern_corners: dict[int, np.ndarray],
    gt_reference: np.ndarray,
    white_corners: dict[int, np.ndarray],
    reference_info: dict,
    reference_provenance: str,
) -> dict:
    """Describe pattern/reference disagreement without choosing which is true.

    Pattern corners are used only as an incompatibility check.  They never
    calibrate, translate, or expand the GT reference; doing so would make the
    answer depend circularly on the method being evaluated.
    """
    pattern = np.asarray(pattern_reference, dtype=np.float64).reshape(4, 2)
    gt = np.asarray(gt_reference, dtype=np.float64).reshape(4, 2)
    marker_center = np.mean(pattern, axis=0)
    inward_vectors = marker_center - pattern
    inward_norms = np.linalg.norm(inward_vectors, axis=1)
    if np.any(inward_norms <= 0):
        raise ValueError("Pattern reference has a degenerate corner geometry")
    inward_units = inward_vectors / inward_norms[:, None]
    tangent_units = np.column_stack((-inward_units[:, 1], inward_units[:, 0]))
    offsets = gt - pattern
    inward_offsets = np.sum(offsets * inward_units, axis=1)
    tangential_offsets = np.sum(offsets * tangent_units, axis=1)

    mean_inward = float(np.mean(inward_offsets))
    inward_std = float(np.std(inward_offsets, ddof=1))
    tangential_rms = float(
        np.sqrt(np.mean(np.square(tangential_offsets)))
    )
    pattern_side = marker_mean_side_length(pattern)
    gt_side = marker_mean_side_length(gt)
    reference_summary = summarize_gt_reference_info(reference_info)
    temporal_coordinate_std = coordinate_std_rms(white_corners)
    temporal_radial_jitter = (
        math.sqrt(2.0) * temporal_coordinate_std
        if temporal_coordinate_std is not None
        else None
    )
    pattern_coordinate_std = coordinate_std_rms(pattern_corners)
    pattern_temporal_radial_jitter = (
        math.sqrt(2.0) * pattern_coordinate_std
        if pattern_coordinate_std is not None
        else None
    )
    uncertainty_terms = [
        pattern_temporal_radial_jitter,
        temporal_radial_jitter,
        reference_summary["model_spread_rms_px"],
    ]
    uncertainty_terms = [
        float(value) for value in uncertainty_terms if finite_number(value) is not None
    ]
    combined_uncertainty = float(
        np.sqrt(np.sum(np.square(uncertainty_terms)))
    ) if uncertainty_terms else 0.0
    rejection_threshold = max(
        GT_SYSTEMATIC_OFFSET_MIN_PX,
        GT_SYSTEMATIC_OFFSET_SIGMA_MULTIPLIER * combined_uncertainty,
    )
    inward_count = int(np.sum(inward_offsets > 0.0))
    outward_count = int(np.sum(inward_offsets < 0.0))
    radial_direction_is_coherent = inward_count == 4 or outward_count == 4
    radial_offset_is_large = abs(mean_inward) > rejection_threshold
    radial_dominates_tangent = tangential_rms < abs(mean_inward)
    coherent_radial_disagreement = (
        radial_direction_is_coherent
        and radial_offset_is_large
        and radial_dominates_tangent
    )
    direction = "inward" if mean_inward >= 0 else "outward"
    disagreement_text = (
        f"{max(inward_count, outward_count)}/4 corners have a coherent "
        f"{direction} radial offset; mean={abs(mean_inward):.3f} px "
        f"> gate={rejection_threshold:.3f} px, tangential RMS="
        f"{tangential_rms:.3f} px."
    )
    if reference_provenance == "externally_validated":
        approved_as_independent_reference = True
        if coherent_radial_disagreement:
            status = "EXTERNALLY_VALIDATED_REFERENCE_WITH_PATTERN_DISAGREEMENT"
            reason = (
                disagreement_text
                + " Reference provenance is externally validated, so this "
                "difference remains in accuracy statistics as a measured "
                "pattern/reference disagreement."
            )
        else:
            status = "EXTERNALLY_VALIDATED_REFERENCE"
            reason = (
                "Reference provenance is externally validated. No coherent "
                f"radial disagreement exceeded {rejection_threshold:.3f} px."
            )
    elif coherent_radial_disagreement:
        approved_as_independent_reference = False
        status = "UNRESOLVED_PATTERN_GT_SYSTEMATIC_DISAGREEMENT"
        reason = (
            disagreement_text
            + " Internal video data cannot determine whether pattern or GT "
            "is biased; raw comparisons are diagnostic only."
        )
    else:
        approved_as_independent_reference = False
        status = "UNVERIFIED_REFERENCE_GEOMETRY"
        reason = (
            f"No coherent radial disagreement exceeded {rejection_threshold:.3f} "
            "px, but target geometry has no independent metrology provenance. "
            "Raw comparisons are diagnostic only."
        )
    return {
        "provenance": reference_provenance,
        "status": status,
        "reason": reason,
        "approved_as_independent_reference": approved_as_independent_reference,
        "inward_corner_count": inward_count,
        "mean_inward_offset_px": mean_inward,
        "inward_offset_std_px": inward_std,
        "tangential_rms_px": tangential_rms,
        "pattern_mean_side_length_px": pattern_side,
        "reference_mean_side_length_px": gt_side,
        "reference_minus_pattern_side_length_px": (
            gt_side - pattern_side
            if gt_side is not None and pattern_side is not None
            else None
        ),
        "reference_to_pattern_side_length_ratio": (
            gt_side / pattern_side
            if gt_side is not None
            and pattern_side is not None
            and pattern_side > 0
            else None
        ),
        "pattern_temporal_radial_jitter_px": pattern_temporal_radial_jitter,
        "temporal_radial_jitter_px": temporal_radial_jitter,
        "combined_uncertainty_px": combined_uncertainty,
        "rejection_threshold_px": rejection_threshold,
        **reference_summary,
    }


def build_pair_summary(
    pattern_video: Path,
    white_video: Path,
    marker_id: int,
    gt_detection_mode: str,
    requested_frames: int,
    pattern_rows: list[dict],
    white_rows: list[dict],
    pattern_corners: dict[int, np.ndarray],
    white_corners: dict[int, np.ndarray],
    gt_median: np.ndarray,
    gt_quality: dict,
    corner_rows: list[dict],
    pattern_diagnostic_video: Path | None,
    gt_diagnostic_video: Path | None,
    diagnostic_roi_spec,
    cross_preview_path: Path | None,
) -> dict:
    record = blank_row(PAIR_SUMMARY_FIELDS)
    record.update(
        {
            "distance_cm": parse_distance(pattern_video),
            "pattern_video_file": str(pattern_video),
            "white_video_file": str(white_video),
            "marker_id": marker_id,
            "gt_detection_mode": gt_detection_mode,
            "gt_reference_provenance": gt_quality["provenance"],
            "gt_reference_estimator": gt_quality["estimator"],
            "gt_quality_status": gt_quality["status"],
            "gt_quality_reason": gt_quality["reason"],
            "gt_approved_as_independent_reference": gt_quality[
                "approved_as_independent_reference"
            ],
            "gt_inward_corner_count": gt_quality["inward_corner_count"],
            "gt_mean_inward_offset_px": gt_quality["mean_inward_offset_px"],
            "gt_inward_offset_std_px": gt_quality["inward_offset_std_px"],
            "gt_tangential_rms_px": gt_quality["tangential_rms_px"],
            "gt_pattern_mean_side_length_px": gt_quality[
                "pattern_mean_side_length_px"
            ],
            "gt_reference_mean_side_length_px": gt_quality[
                "reference_mean_side_length_px"
            ],
            "gt_reference_minus_pattern_side_length_px": gt_quality[
                "reference_minus_pattern_side_length_px"
            ],
            "gt_reference_to_pattern_side_length_ratio": gt_quality[
                "reference_to_pattern_side_length_ratio"
            ],
            "pattern_temporal_radial_jitter_px": gt_quality[
                "pattern_temporal_radial_jitter_px"
            ],
            "gt_temporal_radial_jitter_px": gt_quality[
                "temporal_radial_jitter_px"
            ],
            "gt_combined_uncertainty_px": gt_quality[
                "combined_uncertainty_px"
            ],
            "gt_quality_gate_px": gt_quality["rejection_threshold_px"],
            "gt_frame_median_vs_consensus_rms_px": gt_quality[
                "frame_median_vs_consensus_rms_px"
            ],
            "gt_cross_model_count_min": gt_quality["model_count_min"],
            "gt_cross_model_spread_rms_px": gt_quality[
                "model_spread_rms_px"
            ],
            "gt_cross_model_spread_max_px": gt_quality[
                "model_spread_max_px"
            ],
            "requested_frames_per_video": requested_frames,
            "processed_pattern_frames": len(pattern_rows),
            "valid_pattern_corner_frames": len(pattern_corners),
            "processed_white_frames": len(white_rows),
            "valid_white_corner_frames": len(white_corners),
            "pattern_corner_coordinate_std_rms_px": coordinate_std_rms(
                pattern_corners
            ),
            "white_corner_coordinate_std_rms_px": coordinate_std_rms(
                white_corners
            ),
            "diagnostic_video": (
                None
            ),
            "pattern_diagnostic_video": (
                str(pattern_diagnostic_video)
                if pattern_diagnostic_video is not None
                else None
            ),
            "gt_diagnostic_video": (
                str(gt_diagnostic_video)
                if gt_diagnostic_video is not None
                else None
            ),
            "diagnostic_roi_x_px": diagnostic_roi_spec.x0,
            "diagnostic_roi_y_px": diagnostic_roi_spec.y0,
            "diagnostic_roi_size_source_px": diagnostic_roi_spec.size_px,
            "diagnostic_roi_size_output_px": ALIGNED_ROI_OUTPUT_SIZE_PX,
            "cross_preview_jpg": (
                str(cross_preview_path) if cross_preview_path is not None else None
            ),
        }
    )
    error_stats = describe(
        numeric_values(corner_rows, "pattern_vs_gt_median_error_px")
    )
    for source, target in (
        ("count", "corner_error_sample_count"),
        ("mean", "pattern_vs_gt_corner_error_mean_px"),
        ("rms", "pattern_vs_gt_corner_error_rms_px"),
        ("std", "pattern_vs_gt_corner_error_std_px"),
        ("p95", "pattern_vs_gt_corner_error_p95_px"),
        ("max", "pattern_vs_gt_corner_error_max_px"),
    ):
        record[target] = error_stats[source]
    normalized_stats = describe(
        numeric_values(corner_rows, "normalized_corner_error_percent")
    )
    for source, target in (
        ("count", "normalized_corner_error_sample_count"),
        ("mean", "normalized_corner_error_mean_percent"),
        ("rms", "normalized_corner_error_rms_percent"),
        ("std", "normalized_corner_error_std_percent"),
        ("p95", "normalized_corner_error_p95_percent"),
        ("max", "normalized_corner_error_max_percent"),
    ):
        record[target] = normalized_stats[source]
    same_index_stats = describe(
        numeric_values(corner_rows, "pattern_vs_white_same_index_error_px")
    )
    for source, target in (
        ("count", "same_index_corner_error_sample_count"),
        ("mean", "pattern_vs_white_same_index_error_mean_px"),
        ("rms", "pattern_vs_white_same_index_error_rms_px"),
        ("std", "pattern_vs_white_same_index_error_std_px"),
    ):
        record[target] = same_index_stats[source]
    all_white_gt_errors = [
        error
        for points in white_corners.values()
        for error in errors_between(points, gt_median)
    ]
    record["white_vs_gt_median_error_mean_px"] = describe(
        all_white_gt_errors
    )["mean"]
    return record


def build_distance_summary(
    pair_summaries: list[dict],
    corner_rows: list[dict],
) -> list[dict]:
    distances = {finite_number(row.get("distance_cm")) for row in pair_summaries}
    ordered = sorted(
        distances,
        key=lambda value: (
            value is None,
            value if value is not None else math.inf,
        ),
    )
    output = []
    for distance in ordered:
        same_distance = lambda row: finite_number(row.get("distance_cm")) == distance
        pairs = [row for row in pair_summaries if same_distance(row)]
        approved_pairs = [
            row
            for row in pairs
            if bool(row.get("gt_approved_as_independent_reference"))
        ]
        rejected_pairs = [
            row
            for row in pairs
            if not bool(row.get("gt_approved_as_independent_reference"))
        ]
        corners = [
            row
            for row in corner_rows
            if same_distance(row)
        ]
        if approved_pairs and not rejected_pairs:
            approved_statuses = {
                str(row.get("gt_quality_status")) for row in approved_pairs
            }
            quality_status = (
                next(iter(approved_statuses))
                if len(approved_statuses) == 1
                else "MIXED_INDEPENDENT_REFERENCE_STATUS"
            )
        elif approved_pairs:
            quality_status = "MIXED_GT_QUALITY"
        else:
            rejected_statuses = {
                str(row.get("gt_quality_status")) for row in rejected_pairs
            }
            quality_status = (
                next(iter(rejected_statuses))
                if len(rejected_statuses) == 1
                else "UNRESOLVED_REFERENCE_PROVENANCE"
            )
        record = blank_row(DISTANCE_SUMMARY_FIELDS)
        record.update(
            {
                "distance_cm": distance,
                "pair_count": len(pairs),
                "gt_quality_status": quality_status,
                "independent_reference_pair_count": len(approved_pairs),
                "unresolved_reference_pair_count": len(rejected_pairs),
                "pattern_frame_count": sum(
                    int(row["processed_pattern_frames"]) for row in pairs
                ),
                "valid_pattern_corner_frame_count": sum(
                    int(row["valid_pattern_corner_frames"]) for row in pairs
                ),
            }
        )
        error_stats = describe(
            numeric_values(corners, "pattern_vs_gt_median_error_px")
        )
        for source, target in (
            ("count", "corner_error_sample_count"),
            ("mean", "corner_error_mean_px"),
            ("rms", "corner_error_rms_px"),
            ("std", "corner_error_std_px"),
            ("p95", "corner_error_p95_px"),
            ("max", "corner_error_max_px"),
        ):
            record[target] = error_stats[source]
        normalized_stats = describe(
            numeric_values(corners, "normalized_corner_error_percent")
        )
        for source, target in (
            ("count", "normalized_corner_error_sample_count"),
            ("mean", "normalized_corner_error_mean_percent"),
            ("rms", "normalized_corner_error_rms_percent"),
            ("std", "normalized_corner_error_std_percent"),
            ("p95", "normalized_corner_error_p95_percent"),
            ("max", "normalized_corner_error_max_percent"),
        ):
            record[target] = normalized_stats[source]
        same_index_stats = describe(
            numeric_values(corners, "pattern_vs_white_same_index_error_px")
        )
        record["same_index_corner_error_sample_count"] = same_index_stats["count"]
        record["same_index_corner_error_mean_px"] = same_index_stats["mean"]
        record["same_index_corner_error_rms_px"] = same_index_stats["rms"]
        output.append(record)
    return output


def build_distance_chart_rows(distance_summary: list[dict]) -> list[dict]:
    """Build formula-linked chart data from the monocular distance summary."""
    source_fields = {
        "distance_cm": "distance_cm",
        "corner_error_mean_px": "corner_error_mean_px",
        "corner_error_std_px": "corner_error_std_px",
        "normalized_corner_error_mean_percent": (
            "normalized_corner_error_mean_percent"
        ),
    }
    output = []
    for source_row_number, source_row in enumerate(distance_summary, start=2):
        chart_row = {}
        for chart_field, source_field in source_fields.items():
            value = finite_number(source_row.get(source_field))
            if value is None:
                chart_row[chart_field] = None
                continue
            source_column = excel_column(
                DISTANCE_SUMMARY_FIELDS.index(source_field) + 1
            )
            chart_row[chart_field] = ExcelFormula(
                f"'Distance Summary'!${source_column}${source_row_number}",
                value,
            )
        output.append(chart_row)
    return output


def build_gt_median_row(
    pattern_video: Path,
    white_video: Path,
    marker_id: int,
    gt_detection_mode: str,
    gt_median: np.ndarray,
    valid_white_gt_frames: int,
    gt_quality: dict,
) -> dict:
    row = blank_row(GT_MEDIAN_FIELDS)
    row.update(
        {
            "distance_cm": parse_distance(pattern_video),
            "pattern_video_file": str(pattern_video),
            "white_video_file": str(white_video),
            "marker_id": marker_id,
            "gt_detection_mode": gt_detection_mode,
            "gt_reference_provenance": gt_quality["provenance"],
            "gt_reference_estimator": gt_quality["estimator"],
            "gt_quality_status": gt_quality["status"],
            "gt_quality_reason": gt_quality["reason"],
            "gt_approved_as_independent_reference": gt_quality[
                "approved_as_independent_reference"
            ],
            "valid_white_gt_frames": valid_white_gt_frames,
            "frame_median_vs_consensus_rms_px": gt_quality[
                "frame_median_vs_consensus_rms_px"
            ],
            "cross_model_count_min": gt_quality["model_count_min"],
            "cross_model_spread_rms_px": gt_quality[
                "model_spread_rms_px"
            ],
            "cross_model_spread_max_px": gt_quality[
                "model_spread_max_px"
            ],
            "cross_ensemble_edge_levels": gt_quality["edge_levels"],
            "cross_core_exclusion_px": gt_quality["core_exclusion_px"],
        }
    )
    add_points(row, gt_median, "gt_median")
    return row


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    explicit_white = (
        Path(args.white_video).expanduser().resolve()
        if args.white_video
        else None
    )
    pairs = collect_video_pairs(
        input_path,
        explicit_white,
        args.recursive,
        args.max_pairs,
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            input_path / "monocular_pattern_vs_white_pixel_gt_batch.xlsx"
            if input_path.is_dir()
            else input_path.with_name(
                input_path.stem + "_monocular_vs_white_pixel_gt.xlsx"
            )
        )
    )
    if output.suffix.lower() != ".xlsx":
        output = output.with_suffix(".xlsx")
    diagnostic_dir = (
        Path(args.diagnostic_video_dir).expanduser().resolve()
        if args.diagnostic_video_dir
        else output.with_name(output.stem + "_diagnostics")
    )
    detector = AdaptiveArucoDetector(
        args.subpixel_stability_max_raw_px,
        initial_refinement=args.aruco_initial_refinement,
    )
    if args.gt_mode == "cross_intersection":
        gt_detector = CrossIntersectionConsensusDetector(
            args.white_search_radius_px,
            args.cross_core_exclusion_px,
            args.cross_ensemble_edge_levels,
            args.cross_model_spread_max_px,
        )
    else:
        gt_detector = WhiteBlobDetector(
            args.white_search_radius_px,
            args.white_threshold_ratio,
            args.white_min_contrast_gray,
            args.white_min_area_px,
            args.white_max_area_px,
        )
    gt_detection_mode = getattr(
        gt_detector,
        "detector_mode",
        args.gt_mode,
    )

    print(f"Input: {input_path}")
    print(f"Matched monocular video pairs: {len(pairs)}")
    print(f"Frames requested from each video: {args.frames}")
    print(f"ArUco initial refinement: {args.aruco_initial_refinement}")
    print(f"GT detection mode: {gt_detection_mode}")
    print(f"GT reference provenance: {args.gt_reference_provenance}")
    started = time.perf_counter()
    all_pattern_rows = []
    all_white_rows = []
    all_comparison_rows = []
    all_corner_rows = []
    all_gt_median_rows = []
    pair_summaries = []
    batch_errors = []
    diagnostic_paths = []
    cross_preview_paths = []
    for pair_index, (pattern_video, white_video) in enumerate(pairs, start=1):
        print("")
        print(f"[{pair_index}/{len(pairs)}] Pattern: {pattern_video.name}")
        print(f"[{pair_index}/{len(pairs)}] White GT: {white_video.name}")
        artifact_prefix = f"pair_{pair_index:03d}_{white_video.stem}"
        pattern_diagnostic_video = None
        gt_diagnostic_video = None
        try:
            pattern_distance = parse_distance(pattern_video)
            white_distance = parse_distance(white_video)
            if (
                pattern_distance is not None
                and white_distance is not None
                and not math.isclose(
                    pattern_distance,
                    white_distance,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "Pattern/white filename distance mismatch: "
                    f"{pattern_distance:g} cm versus {white_distance:g} cm"
                )
            marker_id = (
                int(args.marker_id)
                if args.marker_id is not None
                else discover_marker_id(pattern_video, detector, args.frames)
            )
            print(f"ArUco ID: {marker_id}")
            print("Phase 1/2: monocular ArUco subpixel corners...")
            pattern_rows, pattern_corners, search_centers, frame_size = (
                process_pattern_video(
                    pattern_video,
                    args.frames,
                    marker_id,
                    detector,
                    args.progress_every,
                )
            )
            if not args.no_diagnostic_video:
                base_name = f"pair_{pair_index:03d}_{pattern_video.stem}"
                pattern_diagnostic_video = diagnostic_dir / (
                    base_name + "_pattern_aligned_roi.avi"
                )
                gt_diagnostic_video = diagnostic_dir / (
                    base_name + "_gt_aligned_roi.avi"
                )
            diagnostic_roi_spec = build_aligned_roi_spec(
                search_centers,
                args.diagnostic_roi_radius_px,
            )
            print(f"Phase 2/2: monocular {args.gt_mode} GT...")
            (
                white_rows,
                white_corners,
                white_diagnostics,
                gt_median,
                gt_reference_info,
                cross_preview_path,
            ) = process_white_video(
                pattern_video,
                white_video,
                args.frames,
                gt_detector,
                search_centers,
                pattern_corners,
                frame_size,
                pattern_diagnostic_video,
                gt_diagnostic_video,
                diagnostic_dir,
                artifact_prefix,
                args.diagnostic_roi_radius_px,
                args.cross_arm_px,
                args.cross_frame_consensus_max_px,
                args.progress_every,
            )
            gt_quality = audit_gt_geometry(
                search_centers,
                pattern_corners,
                gt_median,
                white_corners,
                gt_reference_info,
                args.gt_reference_provenance,
            )
            print(f"GT quality gate: {gt_quality['status']}")
            if not gt_quality["approved_as_independent_reference"]:
                print(f"WARNING: {gt_quality['reason']}")
            comparison_rows, corner_rows = build_comparison_rows(
                pattern_video,
                white_video,
                gt_detection_mode,
                pattern_rows,
                white_rows,
                pattern_corners,
                white_corners,
                white_diagnostics,
                gt_median,
                gt_quality,
            )
            pair_summary = build_pair_summary(
                pattern_video,
                white_video,
                marker_id,
                gt_detection_mode,
                args.frames,
                pattern_rows,
                white_rows,
                pattern_corners,
                white_corners,
                gt_median,
                gt_quality,
                corner_rows,
                pattern_diagnostic_video,
                gt_diagnostic_video,
                diagnostic_roi_spec,
                cross_preview_path,
            )
            all_pattern_rows.extend(pattern_rows)
            all_white_rows.extend(white_rows)
            all_comparison_rows.extend(comparison_rows)
            all_corner_rows.extend(corner_rows)
            all_gt_median_rows.append(
                build_gt_median_row(
                    pattern_video,
                    white_video,
                    marker_id,
                    gt_detection_mode,
                    gt_median,
                    len(white_corners),
                    gt_quality,
                )
            )
            pair_summaries.append(pair_summary)
            for diagnostic_path in (
                pattern_diagnostic_video,
                gt_diagnostic_video,
            ):
                if diagnostic_path is not None:
                    diagnostic_paths.append(diagnostic_path)
            if cross_preview_path is not None:
                cross_preview_paths.append(cross_preview_path)
            print(
                f"Valid pattern corners: {len(pattern_corners)}/{len(pattern_rows)} | "
                f"valid white GT corners: {len(white_corners)}/{len(white_rows)}"
            )
        except Exception as exc:
            remove_file_if_exists(pattern_diagnostic_video)
            remove_file_if_exists(gt_diagnostic_video)
            remove_cross_previews(diagnostic_dir, artifact_prefix)
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"ERROR, pair skipped: {error_text}")
            batch_errors.append(
                {
                    "distance_cm": parse_distance(pattern_video),
                    "pattern_video_file": str(pattern_video),
                    "white_video_file": str(white_video),
                    "error": error_text,
                }
            )
    if not pair_summaries:
        raise RuntimeError("All video pairs failed; no Excel report was written")

    distance_summary = build_distance_summary(pair_summaries, all_corner_rows)
    distance_chart_rows = build_distance_chart_rows(distance_summary)
    approved_gt_count = sum(
        bool(row.get("gt_approved_as_independent_reference"))
        for row in pair_summaries
    )
    distance_chart_sheet_rows = [
        *distance_chart_rows,
        {
            "distance_cm": "REFERENCE USE",
            "corner_error_mean_px": (
                f"{len(pair_summaries)}/{len(pair_summaries)} cross GT plotted"
            ),
            "corner_error_std_px": (
                f"{approved_gt_count}/{len(pair_summaries)} independently validated"
            ),
            "normalized_corner_error_mean_percent": (
                "See Pair Summary for GT quality status"
            ),
        },
    ]
    batch_error_sheet_rows = (
        batch_errors
        if batch_errors
        else [
            {
                "distance_cm": None,
                "pattern_video_file": "(none)",
                "white_video_file": "(none)",
                "error": "No batch errors. All matched video pairs completed successfully.",
            }
        ]
    )
    settings = [
        {"parameter": "input_path", "value": str(input_path)},
        {"parameter": "matched_video_pair_count", "value": len(pairs)},
        {"parameter": "successful_video_pair_count", "value": len(pair_summaries)},
        {"parameter": "frames_requested_per_video", "value": args.frames},
        {"parameter": "gt_requested_mode", "value": args.gt_mode},
        {"parameter": "gt_detection_mode", "value": gt_detection_mode},
        {
            "parameter": "gt_reference_provenance",
            "value": args.gt_reference_provenance,
        },
        {
            "parameter": "independent_reference_pair_count",
            "value": sum(
                bool(row.get("gt_approved_as_independent_reference"))
                for row in pair_summaries
            ),
        },
        {
            "parameter": "unresolved_reference_pair_count",
            "value": sum(
                not bool(row.get("gt_approved_as_independent_reference"))
                for row in pair_summaries
            ),
        },
        {
            "parameter": "cross_reference_pair_count_used_in_statistics",
            "value": len(pair_summaries),
        },
        {
            "parameter": "marker_id",
            "value": args.marker_id if args.marker_id is not None else "auto per pair",
        },
        {
            "parameter": "aruco_initial_refinement",
            "value": args.aruco_initial_refinement,
        },
        {
            "parameter": "subpixel_stability_max_raw_px",
            "value": args.subpixel_stability_max_raw_px,
        },
        {"parameter": "white_search_radius_px", "value": args.white_search_radius_px},
        {"parameter": "white_threshold_ratio", "value": args.white_threshold_ratio},
        {
            "parameter": "white_min_contrast_gray",
            "value": args.white_min_contrast_gray,
        },
        {"parameter": "white_min_area_px", "value": args.white_min_area_px},
        {"parameter": "white_max_area_px", "value": args.white_max_area_px},
        {
            "parameter": "cross_core_exclusion_px",
            "value": args.cross_core_exclusion_px,
        },
        {
            "parameter": "cross_ensemble_edge_levels",
            "value": ",".join(
                f"{value:g}" for value in args.cross_ensemble_edge_levels
            ),
        },
        {
            "parameter": "cross_model_spread_max_px",
            "value": args.cross_model_spread_max_px,
        },
        {
            "parameter": "cross_frame_consensus_max_px",
            "value": args.cross_frame_consensus_max_px,
        },
        {
            "parameter": "diagnostic_video_enabled",
            "value": not args.no_diagnostic_video,
        },
        {
            "parameter": "diagnostic_roi_margin_each_side_source_px",
            "value": args.diagnostic_roi_radius_px,
        },
        {"parameter": "diagnostic_cross_arm_px", "value": args.cross_arm_px},
        {
            "parameter": "diagnostic_video_width_px",
            "value": ALIGNED_ROI_OUTPUT_SIZE_PX,
        },
        {
            "parameter": "diagnostic_video_height_px",
            "value": ALIGNED_ROI_OUTPUT_SIZE_PX,
        },
        {"parameter": "diagnostic_video_container", "value": "AVI"},
        {"parameter": "diagnostic_video_preferred_codec", "value": "FFV1 lossless"},
        {"parameter": "diagnostic_resize_interpolation", "value": "Lanczos4"},
        {"parameter": "diagnostic_bilateral_diameter_px", "value": BILATERAL_DIAMETER_PX},
        {"parameter": "diagnostic_bilateral_sigma_color", "value": BILATERAL_SIGMA_COLOR},
        {"parameter": "diagnostic_bilateral_sigma_space", "value": BILATERAL_SIGMA_SPACE},
    ]
    protocol = [
        {
            "parameter": "generated_at",
            "value": datetime.now(timezone.utc).astimezone().isoformat(),
        },
        {
            "parameter": "input image",
            "value": "one complete monocular frame; no side-by-side split",
        },
        {
            "parameter": "batch pairing",
            "value": "*_pattern is paired with *_white in the same folder",
        },
        {
            "parameter": "temporal pairing",
            "value": (
                "videos need not be synchronized; cross GT is estimated on "
                "the temporal-mean white frame and checked against the "
                "accepted-frame coordinate median"
            ),
        },
        {
            "parameter": "aligned diagnostic ROI",
            "value": (
                "Pattern and GT are written to separate AVI files using the "
                "same fixed source-pixel crop derived from the temporal Pattern "
                "corner median; no labels or panel layout alter the pixels"
            ),
        },
        {
            "parameter": "diagnostic enlargement",
            "value": (
                "source ROI is bilateral-filtered, enlarged to 320x320 with "
                "Lanczos4, then the one-pixel red subpixel crosses are drawn"
            ),
        },
        {
            "parameter": "temporal stability requirement",
            "value": (
                "camera and target must remain fixed within each video; when "
                "an accepted-frame median exists, temporal-mean consensus "
                f"must agree within {args.cross_frame_consensus_max_px:g} px RMS"
            ),
        },
        {
            "parameter": "ArUco corner",
            "value": (
                "CLAHE initial detection plus adaptive cornerSubPix on the "
                "original grayscale monocular frame"
            ),
        },
        {
            "parameter": "GT detector",
            "value": (
                "cross mode uses several contrast edge levels and takes their "
                "coordinate median; model spread is recorded and gated"
            ),
        },
        {
            "parameter": "cross core exclusion",
            "value": (
                "profiles begin 11 px from the crossing by default to reduce "
                "crossing-core PSF contamination in these videos"
            ),
        },
        {
            "parameter": "pattern/reference disagreement flag",
            "value": (
                "a coherent radial shift beyond max(0.25 px, 3 x combined "
                "pattern/GT temporal and model uncertainty) is reported as "
                "UNRESOLVED_PATTERN_GT_SYSTEMATIC_DISAGREEMENT; it does not "
                "assign fault to either detector"
            ),
        },
        {
            "parameter": "reference provenance handling",
            "value": (
                "all accepted cross references populate Distance Summary and "
                "charts as requested; provenance and disagreement flags remain "
                "visible and do not modify the measured coordinates"
            ),
        },
        {
            "parameter": "GT traceability limitation",
            "value": (
                "internal video consistency cannot establish absolute truth; "
                "absolute GT requires an independently calibrated target or "
                "a same-frame hybrid target"
            ),
        },
        {
            "parameter": "coordinates",
            "value": "raw distorted camera pixels shared by pattern and GT videos",
        },
        {
            "parameter": "normalized error",
            "value": (
                "pattern-to-GT-reference pixel error divided by that frame's "
                "four-edge mean marker length, multiplied by 100"
            ),
        },
        {
            "parameter": "legacy field names",
            "value": (
                "gt_median column names are retained for CSV compatibility; "
                "gt_reference_estimator states the actual estimator"
            ),
        },
        {
            "parameter": "stereo-only fields",
            "value": (
                "RT, baseline, JSON extrinsics, and left/right-eye fields are "
                "not applicable and are intentionally omitted"
            ),
        },
        {
            "parameter": "corner order",
            "value": (
                "OpenCV ArUco order: C0 top-left, C1 top-right, "
                "C2 bottom-right, C3 bottom-left"
            ),
        },
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    csv_specs = (
        ("pattern_frames", PATTERN_FRAME_FIELDS, all_pattern_rows),
        ("white_frames", WHITE_FRAME_FIELDS, all_white_rows),
        ("frame_comparison", FRAME_COMPARISON_FIELDS, all_comparison_rows),
        ("corner_comparison", CORNER_COMPARISON_FIELDS, all_corner_rows),
        ("gt_median", GT_MEDIAN_FIELDS, all_gt_median_rows),
        ("distance_summary", DISTANCE_SUMMARY_FIELDS, distance_summary),
        ("pair_summary", PAIR_SUMMARY_FIELDS, pair_summaries),
    )
    for suffix, fields, rows in csv_specs:
        write_csv(
            prefix.with_name(prefix.name + f"_{suffix}.csv"),
            fields,
            rows,
        )
    write_xlsx(
        output,
        [
            ("Settings", ["parameter", "value"], settings),
            (
                "Distance Charts",
                DISTANCE_CHART_FIELDS,
                distance_chart_sheet_rows,
            ),
            ("Distance Summary", DISTANCE_SUMMARY_FIELDS, distance_summary),
            ("Pair Summary", PAIR_SUMMARY_FIELDS, pair_summaries),
            ("GT Reference", GT_MEDIAN_FIELDS, all_gt_median_rows),
            ("Frame Comparison", FRAME_COMPARISON_FIELDS, all_comparison_rows),
            ("Corner Comparison", CORNER_COMPARISON_FIELDS, all_corner_rows),
            ("Pattern Frames", PATTERN_FRAME_FIELDS, all_pattern_rows),
            ("White GT Frames", WHITE_FRAME_FIELDS, all_white_rows),
            ("Batch Errors", BATCH_ERROR_FIELDS, batch_error_sheet_rows),
            ("Protocol", ["parameter", "value"], protocol),
        ],
    )
    add_distance_charts_to_xlsx(
        output,
        [
            {
                **row,
                "combined_corner_error_mean_px": row.get("corner_error_mean_px"),
                "combined_corner_error_std_px": row.get("corner_error_std_px"),
            }
            for row in distance_summary
        ],
        sheet_index=2,
    )
    print(f"Excel: {output}")
    print("Charts sheet: Distance Charts (active when Excel opens)")
    print(f"Diagnostic folder: {diagnostic_dir}")
    print(f"Aligned Pattern/GT ROI AVI files written: {len(diagnostic_paths)}")
    if args.gt_mode == "cross_intersection":
        print(f"Cross preview JPG files written: {len(cross_preview_paths)}")
    print(
        "Independent reference provenance: "
        f"{approved_gt_count}/{len(pair_summaries)} pairs"
    )
    print(
        "Cross reference used in summary/charts: "
        f"{len(pair_summaries)}/{len(pair_summaries)} pairs"
    )
    if batch_errors:
        print(f"Pairs skipped after errors: {len(batch_errors)}")
    else:
        print("Batch Errors: none")
    print(f"Elapsed: {(time.perf_counter() - started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted")
        raise SystemExit(130)
