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

try:  # Optional: the NumPy path below stays the reference and the fallback.
    # cache=True writes compiled code next to this file (Algorithm/__pycache__).
    # If that folder is not writable, set NUMBA_CACHE_DIR to a writable path,
    # otherwise the first call can stall while numba tries to save the cache.
    from numba import njit as _njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba installed
    HAVE_NUMBA = False

    def _njit(**_kwargs):
        def wrap(function):
            return function
        return wrap


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


def _entry_support_radius(size, config, ratio):
    """One candidate scale's support radius; a pure function of config, no image."""
    sigma = size / 2.
    orientation_sigma = sigma * config.orientation_sigma_factor
    pool = sigma * config.scale_response_pool_sigma_factor
    nominal = float(np.rint(3 * sigma * np.sqrt(2) * 2.5))
    return float(np.ceil(max(nominal + 1, orientation_sigma * config.orientation_radius_factor + 1)
                         + frames._gaussian_radius(sigma * ratio)
                         + (frames._gaussian_radius(pool) if pool > 0 else 0)))


def max_support_radius(config):
    """Largest entry support radius across all candidate scales, without an image.

    Lets a caller size a safe ROI crop before building the pyramid: any point
    whose full support window (plus masked_extra_margin_px of exclusion-mask
    dilation) stays inside the crop gets bit-identical results to running on
    the uncropped image, since pixels outside the crop are then never read.
    """
    ratio = float(config.scale_dog_ratio)
    return int(max(_entry_support_radius(size, config, ratio)
                    for size, octave, layer, scale_index in frames._scale_specs(config)))


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
            support=_entry_support_radius(size, config, ratio)))
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


def _entry_gradient_polar(entry):
    """Per-pixel gradient magnitude and direction for one entry, cached on it.

    Both depend only on that entry's gx/gy maps, never on which point is being
    described or at what angle, yet the per-point loop below used to recompute
    them over every point's whole window (thousands of arctan2/hypot calls per
    point). Computing each map once and gathering from it is bit-identical:
    these are elementwise functions, so a pixel's value is the same whether it
    is evaluated alone or as part of the full map.
    """
    cached = entry.get('_gradient_polar')
    if cached is None:
        gx, gy = entry['gx'], entry['gy']
        cached = (np.hypot(gx, gy), np.degrees(np.arctan2(gy, gx)))
        entry['_gradient_polar'] = cached
    return cached


def _entry_window_grid(entry):
    """Integer (yy, xx) offsets covering one entry's descriptor window.

    Depends only on the entry's nominal radius, so it is rebuilt once per
    entry instead of per point.
    """
    grid = entry.get('_window_grid')
    if grid is None:
        radius = int(entry['nominal_radius'])
        grid = np.mgrid[-radius:radius+1, -radius:radius+1]
        entry['_window_grid'] = grid
    return grid


@_njit(cache=True, fastmath=False)
def _descriptor_kernel(cx_all, cy_all, cos_all, sin_all, angle_all, cell_width_all, radius,
                       strength_map, direction_map, valid_map,
                       histograms, totals, observeds):
    """Compiled twin of the NumPy per-point loop below; must match it bit for bit.

    Every float operation is kept in the same order and grouping as the NumPy
    version, and fastmath is off, so the compiler may not reassociate. In
    particular the four spatial corners are still accumulated into separate
    temporaries that are added into the running totals one corner at a time,
    exactly as `hist += np.bincount(...)` does per corner, because floating
    point addition is not associative.
    """
    height, width = valid_map.shape
    size = 2 * radius + 1
    capacity = size * size
    # Sample buffers, filled in the window's row-major order -- the same order
    # np.bincount walks its inputs in.
    b_ix = np.empty(capacity, np.int64)
    b_iy = np.empty(capacity, np.int64)
    b_io = np.empty(capacity, np.int64)
    b_fx = np.empty(capacity, np.float64)
    b_fy = np.empty(capacity, np.float64)
    b_fo = np.empty(capacity, np.float64)
    b_gauss = np.empty(capacity, np.float64)
    b_mag = np.empty(capacity, np.float64)
    b_good = np.empty(capacity, np.float64)
    corner16_total = np.empty(16, np.float64)
    corner16_obs = np.empty(16, np.float64)
    corner128_lo = np.empty(128, np.float64)
    corner128_hi = np.empty(128, np.float64)

    for p in range(cx_all.shape[0]):
        cx = cx_all[p]
        cy = cy_all[p]
        ct = cos_all[p]
        st = sin_all[p]
        angle = angle_all[p]
        cell_width = cell_width_all[p]
        count = 0
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                x = (ct * j + st * i) / cell_width
                y = (-st * j + ct * i) / cell_width
                bx = x + 1.5
                by = y + 1.5
                if bx <= -1. or bx >= 4. or by <= -1. or by >= 4.:
                    continue
                xs = cx + j
                ys = cy + i
                in_image = 0 <= xs < width and 0 <= ys < height
                xs_c = min(max(xs, 0), width - 1)
                ys_c = min(max(ys, 0), height - 1)
                good = 1. if (in_image and valid_map[ys_c, xs_c]) else 0.
                gauss = np.exp(-(x * x + y * y) / 8.)
                ob = ((angle - direction_map[ys_c, xs_c]) % 360.) / 45.
                fix = np.floor(bx)
                fiy = np.floor(by)
                fio = np.floor(ob)
                b_ix[count] = np.int64(fix)
                b_iy[count] = np.int64(fiy)
                b_io[count] = np.int64(fio)
                b_fx[count] = bx - fix
                b_fy[count] = by - fiy
                b_fo[count] = ob - fio
                b_gauss[count] = gauss
                b_mag[count] = strength_map[ys_c, xs_c] * gauss * good
                b_good[count] = good
                count += 1

        for dx in range(2):
            for dy in range(2):
                for c in range(16):
                    corner16_total[c] = 0.
                    corner16_obs[c] = 0.
                for c in range(128):
                    corner128_lo[c] = 0.
                    corner128_hi[c] = 0.
                for k in range(count):
                    sx = b_ix[k] + dx
                    sy = b_iy[k] + dy
                    if sx < 0 or sx >= 4 or sy < 0 or sy >= 4:
                        continue
                    cell = sy * 4 + sx
                    fx = b_fx[k] if dx == 1 else 1. - b_fx[k]
                    fy = b_fy[k] if dy == 1 else 1. - b_fy[k]
                    spatial = fx * fy
                    weighted = spatial * b_gauss[k]
                    corner16_total[cell] += weighted
                    corner16_obs[cell] += weighted * b_good[k]
                    mass = spatial * b_mag[k]
                    fo = b_fo[k]
                    io = b_io[k]
                    corner128_lo[cell * 8 + (io % 8)] += mass * (1. - fo)
                    corner128_hi[cell * 8 + ((io + 1) % 8)] += mass * fo
                for c in range(16):
                    totals[p, c] += corner16_total[c]
                    observeds[p, c] += corner16_obs[c]
                for c in range(128):
                    histograms[p, c] += corner128_lo[c]
                for c in range(128):
                    histograms[p, c] += corner128_hi[c]


def _fill_descriptors_numba(context, pts, sizes, angles, histograms, coverage, fractions):
    """Group points by scale, run the compiled kernel, finish in NumPy.

    Points are grouped only so one kernel call can share an entry's radius and
    maps; each point's result depends on nothing but its own inputs, so the
    grouping cannot change any value. The final coverage/fraction reductions
    stay in NumPy so their summation order matches the reference loop.
    """
    height, width = context['image_shape']
    entries = context['entries']
    entry_sizes = np.array([e['size'] for e in entries])
    # Same tie-break as min(entries, key=...): first entry at the smallest gap.
    nearest = np.argmin(np.abs(entry_sizes[None, :] - sizes[:, None]), axis=1)
    cx_all = np.rint(pts[:, 0]).astype(np.int64)
    cy_all = np.rint(pts[:, 1]).astype(np.int64)
    in_bounds = (cx_all >= 0) & (cx_all < width) & (cy_all >= 0) & (cy_all < height)
    for entry_index in np.unique(nearest):
        entry = entries[int(entry_index)]
        matched = nearest == entry_index
        if not np.all(np.isclose(entry['size'], sizes[matched], rtol=1e-5)):
            raise ValueError('Descriptor scale is absent from masked context')
        rows = np.flatnonzero(matched & in_bounds)
        if not len(rows):
            continue
        strength_map, direction_map = _entry_gradient_polar(entry)
        group_angles = angles[rows]
        theta = np.deg2rad(group_angles)
        group_hist = np.zeros((len(rows), 128), np.float64)
        group_total = np.zeros((len(rows), 16), np.float64)
        group_observed = np.zeros((len(rows), 16), np.float64)
        # Per-point size, as the NumPy loop uses: the caller's size may differ
        # from entry['size'] within the isclose tolerance (e.g. float32 frames).
        _descriptor_kernel(
            cx_all[rows], cy_all[rows], np.cos(theta), np.sin(theta), group_angles,
            3 * sizes[rows] / 2., int(entry['nominal_radius']),
            strength_map, direction_map, entry['gradient_valid'],
            group_hist, group_total, group_observed)
        histograms[rows] = group_hist
        coverage[rows] = group_observed / np.maximum(group_total, 1e-12)
        fractions[rows] = (group_observed.sum(axis=1)
                           / np.maximum(group_total.sum(axis=1), 1e-12))


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
    if HAVE_NUMBA:
        _fill_descriptors_numba(context, pts, sizes, angles,
                                histograms, coverage, fractions)
        detail.count('custom_rows', len(pts))
        detail.mark('自製遮罩 descriptor 與有效覆蓋累加')
        return dict(descriptors=_normalize(histograms, config), histograms=histograms,
                    cell_valid_fraction=coverage, valid_fraction=fractions,
                    valid_points=fractions >= config.masked_min_valid_fraction)
    for i, (point, size, angle) in enumerate(zip(pts, sizes, angles)):
        entry = min(context['entries'], key=lambda e: abs(e['size']-size))
        if not np.isclose(entry['size'], size, rtol=1e-5):
            raise ValueError('Descriptor scale is absent from masked context')
        cx, cy = np.rint(point).astype(int)
        if not (0 <= cx < width and 0 <= cy < height):
            continue
        yy, xx = _entry_window_grid(entry)
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
        strength_map, direction_map = _entry_gradient_polar(entry)
        gaussian = np.exp(-(x*x+y*y)/8.)
        magnitude = strength_map[ys, xs] * gaussian * good
        # OpenCV descriptor bins use a y-up gradient; keypoint angles use
        # image coordinates (y-down). Keep that bin order for comparability.
        ob = ((angle-direction_map[ys, xs]) % 360) / 45.
        ix, iy, io = np.floor(bx).astype(int), np.floor(by).astype(int), np.floor(ob).astype(int)
        fx, fy, fo = bx-ix, by-iy, ob-io
        histogram = np.zeros(128, float)
        total, observed = np.zeros(16, float), np.zeros(16, float)
        # Everything below is compacted by `keep` before the arithmetic rather
        # than after it. Elementwise products commute with selection, and the
        # bincount accumulation order per corner is unchanged, so the result
        # is bit-identical -- it just stops multiplying the ~1/3 of samples
        # that the corner then discards, and stops recomputing shared terms.
        for dx in (0, 1):
            for dy in (0, 1):
                sx, sy = ix+dx, iy+dy
                keep = (sx >= 0) & (sx < 4) & (sy >= 0) & (sy < 4)
                cells = (sy[keep]*4+sx[keep])
                spatial = ((fx if dx else 1-fx) * (fy if dy else 1-fy))[keep]
                weighted = spatial*gaussian[keep]
                total += np.bincount(cells, weights=weighted, minlength=16)
                observed += np.bincount(cells, weights=weighted*good[keep], minlength=16)
                io_keep, fo_keep = io[keep], fo[keep]
                cell_bins = cells*8
                mass = spatial*magnitude[keep]
                for do in (0, 1):
                    bins = cell_bins + ((io_keep+do) % 8)
                    histogram += np.bincount(
                        bins, weights=mass*(fo_keep if do else 1-fo_keep), minlength=128)
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
