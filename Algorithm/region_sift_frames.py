"""SIFT frames at fixed coordinates, using the same Gaussian layers as OpenCV.

This is deliberately dense: no anchor is moved or suppressed, and uncertain
frames still produce descriptors. A flat patch is useful evidence in a group.
Requested automatic sizes are snapped to realizable nonnegative octave/layer
scales. Explicit fixed-frame control runs retain the caller's requested size.

The full-image pyramid makes estimates independent of which other anchors are
in a batch. ``support_radius_px`` includes the descriptor, gradient stencil,
and finite Gaussian-filter dependencies, so warp holes cannot supply evidence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np
from Algorithm.region_sift_profiling import DetailTimer, timed


def _get(config, name, default):
    return getattr(config, name, default)


def _parameters(config):
    layers = int(_get(config, "sift_n_octave_layers", 3))
    sigma = float(_get(config, "sift_sigma", 1.6))
    if layers < 1 or not np.isfinite(sigma) or sigma <= 0.5:
        raise ValueError("SIFT layers must be positive and sigma must exceed 0.5")
    return layers, sigma


def _gaussian_radius(sigma):
    # OpenCV's auto kernel size for floating-point GaussianBlur is 8*sigma+1.
    return ((int(np.rint(float(sigma) * 8.0 + 1.0)) | 1) - 1) // 2


def _snap_size(size, layers, sigma):
    exponent = max(0, int(np.rint(layers * np.log2(float(size) / (2 * sigma)))))
    octave, layer = divmod(exponent, layers)
    actual = 2 * sigma * 2 ** (exponent / layers)
    return float(actual), int(octave), int(layer)


def _nominal_radius(size, octave):
    factor = float(2 ** octave)
    return float(np.rint(3 * (size / factor / 2) * np.sqrt(2) * 2.5) * factor)


def _scale_specs(config):
    layers, sigma = _parameters(config)
    automatic = bool(_get(config, "auto_scale_orientation", True))
    sizes = (_get(config, "scale_keypoint_sizes_px", (3.2, 4, 5, 6.4, 8, 10, 12, 16))
             if automatic else (_get(config, "keypoint_size_px", 10.0),))
    cap = _get(config, "descriptor_max_support_radius_px", 40.0)
    specs = []
    for index, requested in enumerate(sizes):
        if not np.isfinite(requested) or requested <= 0:
            raise ValueError("SIFT sizes must be finite and positive")
        actual, octave, layer = _snap_size(requested, layers, sigma)
        size = actual if automatic else float(requested)
        if automatic and cap is not None and _nominal_radius(size, octave) > float(cap):
            continue
        if not any(item[1:3] == (octave, layer) for item in specs):
            specs.append((size, octave, layer, index if automatic else -1))
    if not specs:
        raise ValueError("No SIFT scale fits descriptor_max_support_radius_px")
    return sorted(specs, key=lambda item: item[0])


@timed('pyramid')
def create_frame_context(image_gray: np.ndarray, config=None) -> Dict[str, Any]:
    """Build reusable image-only scale maps; do not reuse after image mutation.

    Octave zero starts from the original-resolution image, as SIFT.compute
    does when every supplied keypoint has a nonnegative octave. This avoids
    the implicit doubled base image used by detector mode.
    """
    detail = DetailTimer('pyramid')
    gray = np.asarray(image_gray)
    if gray.ndim != 2 or gray.size == 0 or gray.dtype != np.uint8:
        raise ValueError("Dense SIFT requires a nonempty uint8 grayscale image")
    layers, sigma = _parameters(config)
    specs = _scale_specs(config)
    k = 2 ** (1.0 / layers)
    ratio = float(_get(config, "scale_dog_ratio", k))
    if not np.isfinite(ratio) or ratio <= 1:
        raise ValueError("scale_dog_ratio must exceed one")
    # An integer layer stride keeps both DoG images on the descriptor pyramid.
    stride = max(1, int(np.rint(np.log2(ratio) * layers)))
    max_octave = max(spec[1] for spec in specs)
    max_layer = layers + max(2, stride)
    pyramid, halo = {}, {}
    sigma0 = float(np.float32(sigma))
    initial_blur = float(np.sqrt(np.float32(max(sigma0 * sigma0 - 0.25, 0.01))))
    detail.mark('參數與尺度規格準備')
    base = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), initial_blur)
    detail.mark('初始影像轉換與模糊')
    for octave in range(max_octave + 1):
        factor = 2 ** octave
        for layer in range(max_layer + 1):
            key = (octave, layer)
            if octave == 0 and layer == 0:
                pyramid[key], halo[key] = base, float(_gaussian_radius(initial_blur))
            elif layer == 0:
                previous = pyramid[(octave - 1, layers)]
                if min(previous.shape) < 2:
                    raise ValueError("Image is too small for the configured SIFT octaves")
                pyramid[key] = cv2.resize(previous, (previous.shape[1] // 2,
                                                     previous.shape[0] // 2),
                                          interpolation=cv2.INTER_NEAREST)
                # With odd source dimensions, OpenCV's floor-sized nearest
                # resize can shift sampling by one source pixel from 2*x.
                halo[key] = halo[(octave - 1, layers)] + 2 ** (octave - 1)
            else:
                before = sigma * k ** (layer - 1)
                added = np.sqrt((before * k) ** 2 - before ** 2)
                pyramid[key] = cv2.GaussianBlur(pyramid[(octave, layer - 1)],
                                                (0, 0), float(added))
                halo[key] = halo[(octave, layer - 1)] + _gaussian_radius(added) * factor

    detail.mark('Gaussian 各 octave/layer 建立')
    detail.count('gaussian_layers', len(pyramid))
    detail.count('active_scales', len(specs))
    entries = []
    pool_factor = float(_get(config, "scale_response_pool_sigma_factor", 0.75))
    for size, octave, layer, original_index in specs:
        factor = float(2 ** octave)
        sigma_layer = sigma * k ** layer
        current = pyramid[(octave, layer)]
        next_image = pyramid[(octave, layer + stride)]
        response = np.abs(next_image - current) / 255.0
        pool_sigma = sigma_layer * pool_factor
        pool_radius = 0
        if pool_sigma > 0:
            response = cv2.GaussianBlur(response, (0, 0), pool_sigma)
            pool_radius = _gaussian_radius(pool_sigma)
        # Derivative signs follow image coordinates: +x right and +y down.
        detail.mark('DoG 響應與響應平滑')
        gx = cv2.sepFilter2D(current, cv2.CV_32F, np.array([-1, 0, 1], np.float32),
                            np.ones(1, np.float32))
        gy = cv2.sepFilter2D(current, cv2.CV_32F, np.ones(1, np.float32),
                            np.array([-1, 0, 1], np.float32))
        detail.mark('梯度圖 gx/gy')
        orientation_sigma = sigma_layer * float(_get(config, "orientation_sigma_factor", 1.5))
        orientation_radius = int(np.rint(orientation_sigma * float(
            _get(config, "orientation_radius_factor", 3.0))))
        nominal = _nominal_radius(size, octave)
        descriptor_support = nominal + halo[(octave, layer)] + 1.5 * factor
        orientation_support = ((orientation_radius + 1.5) * factor
                               + halo[(octave, layer)])
        response_support = halo[(octave, layer + stride)] + (pool_radius + 1) * factor
        entries.append({
            "size": size, "octave": octave, "layer": layer,
            "packed_octave": octave | (layer << 8), "scale_index": original_index,
            "response": response, "gx": gx, "gy": gy,
            "orientation_sigma": orientation_sigma,
            "orientation_radius": orientation_radius,
            "nominal_radius": nominal,
            "support": float(np.ceil(max(descriptor_support, orientation_support,
                                          response_support))),
        })
        detail.mark('Support 半徑與尺度資料整理')
    # Only layers referenced by entries survive; full intermediate pyramid can
    # be freed after this function returns.
    return {"image_shape": gray.shape, "entries": entries, "layers": layers,
            "sigma": sigma, "effective_dog_ratio": k ** stride,
            "active_sizes_px": np.array([entry["size"] for entry in entries], np.float32)}


def _sample(image, points):
    points = np.asarray(points, np.float32)
    x = np.clip(points[:, 0], 0, image.shape[1] - 1)
    y = np.clip(points[:, 1], 0, image.shape[0] - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, image.shape[1] - 1), np.minimum(y0 + 1, image.shape[0] - 1)
    wx, wy = x - x0, y - y0
    return ((1 - wx) * (1 - wy) * image[y0, x0] + wx * (1 - wy) * image[y0, x1]
            + (1 - wx) * wy * image[y1, x0] + wx * wy * image[y1, x1])


def _valid_support(points, radius, shape, bad_integral, min_valid_ratio=1.0):
    x0 = np.floor(points[:, 0] - radius).astype(int)
    y0 = np.floor(points[:, 1] - radius).astype(int)
    x1 = np.ceil(points[:, 0] + radius).astype(int)
    y1 = np.ceil(points[:, 1] + radius).astype(int)
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < shape[1]) & (y1 < shape[0])
    if bad_integral is not None and np.any(valid):
        idx = np.flatnonzero(valid)
        xa, ya, xb, yb = x0[idx], y0[idx], x1[idx] + 1, y1[idx] + 1
        bad = (bad_integral[yb, xb] - bad_integral[ya, xb]
               - bad_integral[yb, xa] + bad_integral[ya, xa])
        area = (xb - xa) * (yb - ya)
        valid[idx] &= bad <= (1.0 - min_valid_ratio) * area + 1e-9
    return valid


@timed('angle')
def _orientation(entry, point, config):
    detail = DetailTimer('angle')
    factor = float(2 ** entry["octave"])
    center = point / factor
    radius = entry["orientation_radius"]
    # Estimate on the same rounded octave pixel used by OpenCV's descriptor.
    cx, cy = np.rint(center).astype(int)
    gx, gy = entry["gx"], entry["gy"]
    x0, x1 = max(0, cx - radius), min(gx.shape[1], cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(gx.shape[0], cy + radius + 1)
    detail.mark('中心座標與視窗界限')
    xx, yy = np.meshgrid(np.arange(x0, x1) - cx, np.arange(y0, y1) - cy)
    weight = np.exp(-(xx * xx + yy * yy) / (2 * entry["orientation_sigma"] ** 2))
    detail.mark('座標網格與 Gaussian 權重')
    detail.count('window_pixels', (x1-x0)*(y1-y0))
    local_x, local_y = gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
    magnitude = np.hypot(local_x, local_y) * weight
    bins = int(_get(config, "orientation_bins", 36))
    coordinates = np.mod(np.degrees(np.arctan2(local_y, local_x)), 360) * bins / 360
    lower = np.floor(coordinates).astype(int)
    fraction = coordinates - lower
    detail.mark('梯度強度、atan2 與 bin 座標')
    histogram = (np.bincount((lower % bins).ravel(), weights=(magnitude * (1 - fraction)).ravel(), minlength=bins)
                 + np.bincount(((lower + 1) % bins).ravel(), weights=(magnitude * fraction).ravel(), minlength=bins))
    detail.mark('方向直方圖累加')
    for _ in range(int(_get(config, "orientation_hist_smooth_passes", 2))):
        histogram = (np.roll(histogram, 1) + 2 * histogram + np.roll(histogram, -1)) / 4
    detail.mark('方向直方圖平滑')
    peak = int(np.argmax(histogram))
    strength = float(histogram[peak])
    if strength <= 1e-8:
        detail.mark('主峰、次峰與角度可靠性')
        return float(_get(config, "keypoint_angle_deg", 0.0)) % 360, 0.0, 0.0
    before, after = histogram[(peak - 1) % bins], histogram[(peak + 1) % bins]
    denominator = before - 2 * strength + after
    offset = 0.5 * (before - after) / denominator if abs(denominator) > 1e-12 else 0.0
    angle = ((peak + np.clip(offset, -0.5, 0.5)) * 360 / bins) % 360
    peaks = (histogram >= np.roll(histogram, 1)) & (histogram > np.roll(histogram, -1))
    peaks[peak] = False
    second = float(np.max(histogram[peaks])) if np.any(peaks) else 0.0
    confidence = max(0.0, (strength - second) / strength)
    detail.mark('主峰、次峰與角度可靠性')
    return float(angle), strength, confidence


@timed('frames')
def estimate_dense_sift_frames(image_gray, points, config=None, context=None, valid_mask=None):
    """Return one frame per input coordinate, including validity/reliability.

    ``valid`` tests complete selected-scale image support; it is not a texture
    threshold. Invalid coordinates retain finite fallback metadata and are
    marked false so the caller can reject only affected search candidates.
    ``scale_confidence`` and ``orientation_confidence`` are peak-separation
    measures, not calibrated correctness probabilities.

    DoG response and ``scale_response_floor`` use image intensities divided by
    255; the default 1e-4 therefore means 0.0255 input gray levels. Orientation
    strength is the dominant histogram bin's weighted gradient mass in input
    intensity units, not its fraction of total histogram mass. Neither value
    adds a texture-state feature to the descriptor or drops any anchor.
    """
    detail = DetailTimer('frames')
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if context is None:
        context = create_frame_context(image_gray, config)
    if context["image_shape"] != np.asarray(image_gray).shape:
        raise ValueError("Frame context/image shapes differ")
    count = len(pts)
    detail.count('input_points', count)
    result = {name: np.zeros(count, np.float32) for name in (
        "size_px", "angle_deg", "scale_response", "orientation_strength", "scale_confidence",
        "orientation_confidence", "support_radius_px", "nominal_support_radius_px")}
    result.update({name: np.zeros(count, np.int32) for name in ("scale_index", "octave")})
    result.update({name: np.zeros(count, bool) for name in (
        "valid", "scale_reliable", "orientation_reliable")})
    if count == 0:
        return result
    detail.mark('輸入準備與結果配置（含缺省 context 建立）')
    entries = context["entries"]
    bad_integral = None
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != context["image_shape"]:
            raise ValueError("Warp validity mask shape differs from image")
        bad_integral = cv2.integral((mask <= 0).astype(np.uint8), sdepth=cv2.CV_64F)
    min_valid_ratio = float(_get(config, "min_valid_warp_ratio", 1.0))
    if not 0 <= min_valid_ratio <= 1:
        raise ValueError("min_valid_warp_ratio must lie in [0,1]")
    detail.mark('有效遮罩積分圖與參數檢查')
    finite = np.all(np.isfinite(pts), axis=1)
    safe_points = np.where(np.isfinite(pts), pts, 0)
    quantization = float(_get(config, "frame_coordinate_quantization_px", 0.0))
    if quantization > 0:
        safe_points = np.rint(safe_points / quantization) * quantization
    unique, inverse = np.unique(safe_points, axis=0, return_inverse=True)
    detail.mark('座標準備與 np.unique 去重')
    detail.count('unique_points', len(unique))
    detail.count('duplicate_points', count-len(unique))
    detail.count('scale_checks', len(unique)*len(entries))
    responses = np.column_stack([_sample(entry["response"], unique / (2 ** entry["octave"]))
                                 for entry in entries])
    detail.mark('各尺度響應圖取樣')
    valid = np.column_stack([_valid_support(unique, entry["support"] + quantization,
                                            context["image_shape"], bad_integral, min_valid_ratio)
                             for entry in entries])
    # Masked contexts contain only clean gradients/responses. A reflected
    # center is not by itself an invalid frame: estimate from its neighborhood
    # (or the normal flat-scale fallback), then test descriptor coverage.
    detail.mark('各尺度 support 有效性檢查')
    responses[~valid] = -np.inf
    best = np.argmax(responses, axis=1)
    any_valid = np.any(valid, axis=1)
    best_response = responses[np.arange(len(unique)), best]
    automatic = bool(_get(config, "auto_scale_orientation", True))
    floor = float(_get(config, "scale_response_floor", 1e-4))
    flat = (best_response < floor) | ~np.isfinite(best_response)
    requested_fallback = float(_get(config, "flat_keypoint_size_px", 3.2))
    fallback_order = np.argsort([abs(entry["size"] - requested_fallback) for entry in entries])
    for row in np.flatnonzero(flat):
        fits = [index for index in fallback_order if valid[row, index]]
        best[row] = fits[0] if fits else int(fallback_order[0])
    confidence = np.zeros(len(unique), np.float32)
    if len(entries) > 1:
        sorted_response = np.sort(responses, axis=1)
        usable = np.isfinite(sorted_response[:, -2]) & (sorted_response[:, -1] > floor)
        confidence[usable] = ((sorted_response[usable, -1] - sorted_response[usable, -2])
                              / sorted_response[usable, -1])
    detail.mark('尺度選擇、平坦回退與尺度信心')
    detail.count('valid_unique_points', np.count_nonzero(any_valid))
    unique_frames = []
    for row, choice in enumerate(best):
        entry = entries[int(choice)]
        detail.mark('逐點資料、可靠性與迴圈開銷')
        if any_valid[row] and automatic:
            angle, strength, orientation_confidence = _orientation(entry, unique[row], config)
            detail.mark('角度估計（子項見 angle）')
        else:
            angle = float(_get(config, "keypoint_angle_deg", 0.0)) % 360
            strength, orientation_confidence = 0.0, 0.0
        response = float(responses[row, choice]) if any_valid[row] else 0.0
        scale_reliable = (automatic and not flat[row] and confidence[row] >= float(
            _get(config, "scale_min_confidence", 0.05)))
        if not _get(config, "scale_boundary_is_reliable", False):
            allowed = np.flatnonzero(valid[row])
            if len(allowed) == 0 or choice in (allowed[0], allowed[-1]):
                # An edge/warp hole may reduce the available scale interval,
                # too. Its new endpoint is not evidence of a scale maximum.
                scale_reliable = False
        orientation_reliable = (automatic and strength > 1e-8 and orientation_confidence >= float(
            _get(config, "orientation_min_confidence", 0.10)))
        unique_frames.append({
            "size_px": entry["size"], "angle_deg": angle,
            "scale_response": max(0.0, response), "orientation_strength": strength,
            "scale_index": entry["scale_index"], "octave": entry["packed_octave"],
            "scale_confidence": confidence[row], "orientation_confidence": orientation_confidence,
            "scale_reliable": scale_reliable, "orientation_reliable": orientation_reliable,
            "support_radius_px": entry["support"] + quantization,
            "nominal_support_radius_px": entry["nominal_radius"], "valid": any_valid[row],
        })
    detail.mark('逐點資料、可靠性與迴圈開銷')
    for name in result:
        result[name][:] = np.asarray([frame[name] for frame in unique_frames])[inverse]
    result["valid"] &= finite
    detail.mark('去重結果還原與輸出組裝')
    return result


@timed('descriptor')
def compute_descriptors_at_points(image_gray, points, sift=None, config=None,
                                  sizes_px=None, angles_deg=None, octaves=None):
    """Compute aligned Nx128 descriptors without dropping low-information rows.

    A supplied OpenCV SIFT object is reused when its pyramid parameters match.
    When octaves are omitted, nearest Gaussian layers are inferred from sizes;
    automatic callers should pass all three arrays from frame estimation.
    """
    detail = DetailTimer('descriptor')
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    count = len(pts)
    if count == 0:
        return np.empty((0, 128), np.float32)
    layers, sigma = _parameters(config)
    sizes = np.full(count, _get(config, "keypoint_size_px", 10.0), np.float32) if sizes_px is None else np.asarray(sizes_px, np.float32).reshape(-1)
    angles = np.full(count, _get(config, "keypoint_angle_deg", 0.0), np.float32) if angles_deg is None else np.asarray(angles_deg, np.float32).reshape(-1)
    if len(sizes) != count or len(angles) != count or not np.all(np.isfinite(pts)):
        raise ValueError("SIFT coordinates and frame arrays must be finite and aligned")
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0) or not np.all(np.isfinite(angles)):
        raise ValueError("SIFT sizes must be positive and sizes/angles finite")
    if octaves is None:
        values = [_snap_size(size, layers, sigma) for size in sizes]
        packed = np.array([octave | (layer << 8) for _, octave, layer in values], np.int32)
    else:
        packed = np.asarray(octaves, np.int32).reshape(-1)
    if len(packed) != count or np.any((packed & 255) > 127) or np.any(((packed >> 8) & 255) > layers + 2):
        raise ValueError("SIFT octave/layer arrays must be aligned and nonnegative")
    if sift is None:
        sift = cv2.SIFT_create(nOctaveLayers=layers, sigma=sigma)
    elif ((hasattr(sift, "getNOctaveLayers") and sift.getNOctaveLayers() != layers)
          or (hasattr(sift, "getSigma") and abs(sift.getSigma() - sigma) > 1e-6)):
        sift = cv2.SIFT_create(nOctaveLayers=layers, sigma=sigma)
    detail.mark('陣列檢查、octave 準備與 SIFT 物件')
    detail.count('rows', count)
    keypoints = [cv2.KeyPoint(float(point[0]), float(point[1]), float(size),
                              float(angle % 360), 0.0, int(octave), index)
                 for index, (point, size, angle, octave) in enumerate(zip(pts, sizes, angles, packed))]
    detail.mark('建立 cv2.KeyPoint 列表')
    returned, descriptors = sift.compute(image_gray, keypoints)
    detail.mark('OpenCV SIFT.compute（含內部前置）')
    if descriptors is None or descriptors.shape != (count, 128) or len(returned) != count:
        raise ValueError("SIFT must preserve exactly one descriptor for every anchor")
    if any(keypoint.class_id != index for index, keypoint in enumerate(returned)):
        raise ValueError("SIFT changed fixed-anchor descriptor ordering")
    descriptors = np.asarray(descriptors, np.float32)
    if _get(config, "normalize_descriptors", False):
        descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12)
    detail.mark('descriptor 檢查與正規化')
    return descriptors
