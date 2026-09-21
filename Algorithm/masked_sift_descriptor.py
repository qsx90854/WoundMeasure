"""Mask-aware dense SIFT-like descriptors (not a bitwise OpenCV replacement).

Inputs must be raw grayscale, not CLAHE output. Nonzero specular pixels are
missing observations, never black texture. All scale maps use normalized
convolution of ORIGINAL valid observations; gradients, DoG and orientation
exclude invalid samples. Fixed-mask changes to excluded pixels cannot affect
these outputs. Detection/undistortion performed before this module is outside
that guarantee. Scales are evaluated at full resolution (no octave decimation).
"""
import cv2
import numpy as np

from Algorithm import region_sift_frames as frames
from Algorithm.region_sift_profiling import timed, DetailTimer


def validate_config(config):
    for name in ('masked_min_blur_weight', 'masked_min_valid_fraction',
                 'masked_min_cell_fraction', 'masked_min_common_fraction',
                 'masked_descriptor_clip'):
        value = getattr(config, name)
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'{name} must be in (0, 1]')
    value = config.masked_extra_margin_px
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('masked_extra_margin_px must be a nonnegative integer')
    if not np.isfinite(config.masked_min_group_fraction) or not 0 < config.masked_min_group_fraction <= 1:
        raise ValueError('masked_min_group_fraction must be in (0, 1]')
    value = config.masked_min_points_per_cell
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= config.points_per_cell:
        raise ValueError('masked_min_points_per_cell must be between 1 and points_per_cell')
    value = config.masked_max_deficient_cells
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('masked_max_deficient_cells must be a nonnegative integer')
    if not np.isfinite(config.masked_missing_penalty_weight) or not 0 <= config.masked_missing_penalty_weight <= 1:
        raise ValueError('masked_missing_penalty_weight must be in [0, 1]')


def exclusion_mask(mask, config):
    bad = (np.asarray(mask) > 0).astype(np.uint8)
    r = config.masked_extra_margin_px
    return cv2.dilate(bad, np.ones((2*r+1, 2*r+1), np.uint8)) if r else bad


def _smooth(image, valid, sigma):
    weight = cv2.GaussianBlur(valid.astype(np.float64), (0, 0), float(sigma),
                              borderType=cv2.BORDER_CONSTANT)
    numerator = cv2.GaussianBlur(np.where(valid, image, 0).astype(np.float64),
                                 (0, 0), float(sigma), borderType=cv2.BORDER_CONSTANT)
    return numerator / np.maximum(weight, 1e-12), weight


@timed('pyramid')
def create_context(image_gray, specular_mask, config, valid_mask=None):
    validate_config(config)
    image = np.asarray(image_gray)
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError('Masked SIFT requires nonempty raw uint8 grayscale')
    mask = np.asarray(specular_mask)
    if mask.shape != image.shape or not np.all(np.isfinite(mask)):
        raise ValueError('Masked SIFT requires an aligned finite specular mask')
    bad = exclusion_mask(mask, config)
    valid = bad == 0
    if valid_mask is not None:
        if np.asarray(valid_mask).shape != image.shape:
            raise ValueError('Warp validity mask shape mismatch')
        valid &= np.asarray(valid_mask) > 0
    entries = []
    ratio = float(config.scale_dog_ratio)
    stencil = np.ones((3, 3), np.uint8)
    for size, octave, layer, scale_index in frames._scale_specs(config):
        sigma = size / 2.
        current, weight = _smooth(image, valid, sigma)
        next_image, next_weight = _smooth(image, valid, sigma * ratio)
        good = valid & (weight >= config.masked_min_blur_weight)
        gradient_valid = cv2.erode(good.astype(np.uint8), stencil,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        gx = cv2.sepFilter2D(current, cv2.CV_64F, np.array([-1, 0, 1], float), np.ones(1))
        gy = cv2.sepFilter2D(current, cv2.CV_64F, np.ones(1), np.array([-1, 0, 1], float))
        # Do not normalize roundoff on a constant valid surface into texture.
        gx[np.abs(gx) < 1e-10] = 0
        gy[np.abs(gy) < 1e-10] = 0
        gx[~gradient_valid] = 0
        gy[~gradient_valid] = 0
        response_valid = good & (next_weight >= config.masked_min_blur_weight)
        response = np.abs(next_image-current) / 255.
        pool = sigma * config.scale_response_pool_sigma_factor
        if pool > 0:
            response, response_weight = _smooth(response, response_valid, pool)
            # A missing center can still have a valid neighborhood response.
            # No excluded intensity contributes to this pooled response.
            response_valid = response_weight >= config.masked_min_blur_weight
        response[~response_valid] = 0
        orientation_sigma = sigma * config.orientation_sigma_factor
        nominal = float(np.rint(3 * sigma * np.sqrt(2) * 2.5))
        entries.append(dict(size=size, octave=0, layer=0, packed_octave=0,
            source_octave=octave, source_layer=layer, scale_index=scale_index,
            response=response, response_valid=response_valid,
            gx=gx, gy=gy, gradient_valid=gradient_valid,
            orientation_sigma=orientation_sigma,
            orientation_radius=int(np.rint(orientation_sigma * config.orientation_radius_factor)),
            nominal_radius=nominal,
            support=float(np.ceil(max(nominal+1, orientation_sigma*config.orientation_radius_factor+1)
                                 + frames._gaussian_radius(sigma*ratio)
                                 + (frames._gaussian_radius(pool) if pool > 0 else 0)))))
    return dict(image_shape=image.shape, entries=entries, masked=True,
                valid_mask=valid, layers=config.sift_n_octave_layers, sigma=config.sift_sigma,
                active_sizes_px=np.array([e['size'] for e in entries], np.float32))


def _normalize(hist, config):
    values = np.asarray(hist, np.float32).copy()
    values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
    np.minimum(values, config.masked_descriptor_clip, out=values)
    values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
    if not config.normalize_descriptors:
        values *= 512.
    return values


@timed('descriptor')
def compute_descriptors(context, points, sizes_px, angles_deg, config):
    """Preserve N rows; validity is evidence coverage, NOT gradient strength."""
    detail = DetailTimer('descriptor')
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    sizes = np.asarray(sizes_px, float).reshape(-1)
    angles = np.asarray(angles_deg, float).reshape(-1)
    if len(pts) != len(sizes) or len(pts) != len(angles) or not np.all(np.isfinite(pts)):
        raise ValueError('Masked descriptor coordinates/frames must be finite and aligned')
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0) or not np.all(np.isfinite(angles)):
        raise ValueError('Masked descriptor frames must be finite with positive sizes')
    histograms = np.zeros((len(pts), 128), np.float32)
    coverage = np.zeros((len(pts), 16), np.float32)
    fractions = np.zeros(len(pts), np.float32)
    height, width = context['image_shape']
    for i, (point, size, angle) in enumerate(zip(pts, sizes, angles)):
        entry = min(context['entries'], key=lambda e: abs(e['size']-size))
        if not np.isclose(entry['size'], size, rtol=1e-5):
            raise ValueError('Descriptor scale is absent from masked context')
        radius = int(entry['nominal_radius'])
        cx, cy = np.rint(point).astype(int)
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
        theta = np.deg2rad(angle)
        cell_width = 3 * size / 2.
        x = (np.cos(theta)*xx + np.sin(theta)*yy) / cell_width
        y = (-np.sin(theta)*xx + np.cos(theta)*yy) / cell_width
        bx, by = x + 1.5, y + 1.5
        inside = (bx > -1) & (bx < 4) & (by > -1) & (by < 4)
        xs, ys = cx+xx[inside], cy+yy[inside]
        bx, by, x, y = bx[inside], by[inside], x[inside], y[inside]
        in_image = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs, ys = np.clip(xs, 0, width-1), np.clip(ys, 0, height-1)
        good = in_image & entry['gradient_valid'][ys, xs]
        gx, gy = entry['gx'][ys, xs], entry['gy'][ys, xs]
        gaussian = np.exp(-(x*x+y*y)/8.)
        magnitude = np.hypot(gx, gy) * gaussian * good
        # OpenCV descriptor bins use a y-up gradient; keypoint angles use
        # image coordinates (y-down). Keep that bin order for comparability.
        ob = ((angle-np.degrees(np.arctan2(gy, gx))) % 360) / 45.
        ix, iy, io = np.floor(bx).astype(int), np.floor(by).astype(int), np.floor(ob).astype(int)
        fx, fy, fo = bx-ix, by-iy, ob-io
        histogram = np.zeros(128, float)
        total, observed = np.zeros(16, float), np.zeros(16, float)
        for dx in (0, 1):
            for dy in (0, 1):
                sx, sy = ix+dx, iy+dy
                keep = (sx >= 0) & (sx < 4) & (sy >= 0) & (sy < 4)
                cells = (sy[keep]*4+sx[keep])
                spatial = (fx if dx else 1-fx) * (fy if dy else 1-fy)
                total += np.bincount(cells, weights=(spatial*gaussian)[keep], minlength=16)
                observed += np.bincount(cells, weights=(spatial*gaussian*good)[keep], minlength=16)
                for do in (0, 1):
                    bins = cells*8 + ((io[keep]+do) % 8)
                    mass = spatial*magnitude*(fo if do else 1-fo)
                    histogram += np.bincount(bins, weights=mass[keep], minlength=128)
        histograms[i] = histogram
        coverage[i] = observed / np.maximum(total, 1e-12)
        fractions[i] = observed.sum() / max(total.sum(), 1e-12)
    detail.count('custom_rows', len(pts))
    detail.mark('自製遮罩 descriptor 與有效覆蓋累加')
    return dict(descriptors=_normalize(histograms, config), histograms=histograms,
                cell_valid_fraction=coverage, valid_fraction=fractions,
                valid_points=fractions >= config.masked_min_valid_fraction)


def compare_descriptors(left, right, config):
    """Cell-level common evidence; broadcast left N rows over right BxN rows.

    Common fraction uses the conservative intersection lower bound (l+r-1).
    This is not exact pixel-level correspondence of occlusion masks.
    """
    common = np.maximum(0., left['cell_valid_fraction'] + right['cell_valid_fraction'] - 1.)
    usable = common >= config.masked_min_cell_fraction
    common_fraction = np.mean(np.where(usable, common, 0.), axis=-1)
    gate = np.repeat(usable, 8, axis=-1)
    a = _normalize(left['histograms'] * gate, config)
    b = _normalize(right['histograms'] * gate, config)
    distance = np.linalg.norm(a-b, axis=-1)
    valid = (left['valid_points'] & right['valid_points']
             & (common_fraction >= config.masked_min_common_fraction))
    # Fixed N rows: missing evidence never becomes a free zero or disappears
    # from CellAll. Nonnegative unit descriptors have maximum L2 sqrt(2).
    ceiling = np.sqrt(2.) * (1. if config.normalize_descriptors else 512.)
    distance = np.minimum(distance, ceiling)
    distance += config.masked_missing_penalty_weight * (1-common_fraction) * (ceiling-distance)
    distance = np.where(valid, distance, ceiling)
    return distance, valid, common_fraction


def group_validity(valid_points, metadata, config, return_details=False):
    """Require enough point pairs AND evidence in (almost) every sampling cell.

    At least the original best-K must remain valid, so KEEP never needs to
    manufacture a measurement from a missing row. P shares the middle cell.
    Up to masked_max_deficient_cells Region cells may fall below
    masked_min_points_per_cell (e.g. a cell a specular blob fully covers)
    without vetoing the whole candidate; masked_max_deficient_cells=0
    reproduces the original all-cells-required rule exactly. A deficient
    cell is never a free pass: its rows keep the ceiling distance penalty
    from compare_descriptors, so CellAll (score_candidate_groups) still
    feels the missing evidence.
    """
    valid = np.asarray(valid_points, bool)
    if valid.ndim != 2 or valid.shape[1] != len(metadata):
        raise ValueError('Group validity and metadata must align')
    count = valid.shape[1]
    keep = (int(config.keep_best_count) if config.keep_best_count is not None
            else int(round(count * config.keep_best_ratio)))
    required = max(1, min(count, keep), int(np.ceil(count*config.masked_min_group_fraction)))
    valid_count = valid.sum(axis=1)
    enough_points = valid_count >= required
    cells = [(int(m['cell_row']), int(m['cell_col'])) for m in metadata]
    deficient_cell_count = np.zeros(len(valid), int)
    for cell in set(cells):
        selected = np.array([c == cell for c in cells])
        deficient_cell_count += valid[:, selected].sum(axis=1) < config.masked_min_points_per_cell
    enough_cells = deficient_cell_count <= config.masked_max_deficient_cells
    accepted = enough_points & enough_cells
    if return_details:
        return dict(valid=accepted, valid_count=valid_count, required=required,
                    enough_points=enough_points, enough_cells=enough_cells,
                    deficient_cell_count=deficient_cell_count)
    return accepted
