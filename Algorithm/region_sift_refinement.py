"""Bounded post-search disparity-plane refinement; original diagnostics stay intact.

Only accepted matches are refined. Each anchor moves along the local warped
epipolar tangent by c + ax*dx + ay*dy. This is a local parallel-epipolar
approximation, not a general surface reconstruction or a new global search.
Cost-table interpolation proposes a model; real descriptors verify it before
the center is changed. Missing table samples are never interpolated across.
"""
from itertools import product
import time

import cv2
import numpy as np


def validate_refinement_config(config):
    from Algorithm.Region_SIFT_Matching import RegionSIFTError
    for name in ('refine_radius_px', 'refine_sample_step_px'):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise RegionSIFTError(f'{name} must be positive and finite')
    intervals = 2 * config.refine_radius_px / config.refine_sample_step_px
    if not 2 <= intervals <= 16 or abs(intervals - round(intervals)) > 1e-6 or round(intervals) % 2:
        raise RegionSIFTError('Refinement needs 2..16 even intervals, including zero')
    if (isinstance(config.refine_model_levels, bool)
            or not isinstance(config.refine_model_levels, (int, np.integer))
            or not 1 <= config.refine_model_levels <= 5):
        raise RegionSIFTError('refine_model_levels must be an integer in [1,5]')
    for name in ('refine_slope_penalty', 'refine_min_gain'):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise RegionSIFTError(f'{name} must be finite and nonnegative')


def fit_cost_table(offsets, costs, relative_points, metadata, config):
    """Bounded 3-parameter hierarchical grid, at most 125 * levels models.

Slopes are parameterized as displacements at the normalized ROI edges.
The surrogate uses the existing TrimK + CellAll, not independent point minima.
"""
    offsets = np.asarray(offsets, float)
    costs = np.asarray(costs, float)
    xy = np.asarray(relative_points, float)
    n = len(xy)
    extent = np.maximum(np.max(np.abs(xy), axis=0), 1.)
    design = np.column_stack([np.ones(n), xy / extent])
    complete = np.all(np.isfinite(costs), axis=0)
    informative = complete & (np.ptp(np.where(np.isfinite(costs), costs, 0), axis=0) > 1e-4)
    if informative.sum() < 6 or np.linalg.matrix_rank(design[informative]) < 3:
        return None, 'insufficient spatially distributed informative anchors'
    hull = cv2.convexHull(xy[informative].astype(np.float32))
    if cv2.pointPolygonTest(hull, (0., 0.), False) <= 0:
        return None, 'informative anchors do not surround P'
    cell_keys = [(m['cell_row'], m['cell_col']) for m in metadata]
    cells = [np.array([k == cell for k in cell_keys]) for cell in sorted(set(cell_keys))]
    keep = min(n, max(1, int(config.keep_best_count) if config.keep_best_count is not None
                      else round(n * config.keep_best_ratio)))
    radius = config.refine_radius_px
    step = float(offsets[1] - offsets[0])
    rows = np.arange(n)[None, :]

    def score(params):
        shifts = params @ design.T
        admissible = np.all(np.abs(shifts) <= radius + 1e-7, axis=1)
        z = np.clip((shifts - offsets[0]) / step, 0, len(offsets)-1)
        lo = np.floor(z).astype(int)
        hi = np.minimum(lo+1, len(offsets)-1)
        fraction = z-lo
        low, high = costs[lo, rows], costs[hi, rows]
        # Exact table knots need only that knot, not its invalid neighbor.
        valid = np.isfinite(low) & ((fraction <= 1e-7) | np.isfinite(high))
        values = np.where(np.isfinite(low), low, 0) * (1-fraction)
        values += np.where(np.isfinite(high), high, 0) * fraction
        admissible &= np.all(valid, axis=1)
        trim = np.mean(np.sort(values, axis=1)[:, :keep], axis=1)
        balanced = np.mean([values[:, cell].mean(axis=1) for cell in cells], axis=0)
        penalty = config.refine_slope_penalty * np.sum(params[:, 1:]**2, axis=1)
        value = (1-config.group_balance_weight)*trim + config.group_balance_weight*balanced + penalty
        return np.where(admissible, value, np.inf)

    best = np.zeros(3)
    best_score = score(best[None])[0]
    model_count = 0
    scale = radius
    for _ in range(config.refine_model_levels):
        candidates = best + np.asarray(list(product(np.linspace(-scale, scale, 5), repeat=3)))
        values = score(candidates)
        model_count += len(candidates)
        # Prefer smaller deformation on ties; zero always remains an option.
        order = np.lexsort((np.sum(candidates*candidates, axis=1), values))
        idx = order[0]
        if values[idx] < best_score - 1e-9:
            best, best_score = candidates[idx], float(values[idx])
        scale *= .5
    if not np.isfinite(best_score):
        return None, 'no complete local cost-table model'
    shifts = design @ best
    if np.max(np.abs(shifts)) >= radius - 1e-6:
        return None, 'model reaches local correction boundary'
    return dict(center_shift_px=float(best[0]), slope_x=float(best[1]/extent[0]),
                slope_y=float(best[2]/extent[1]), shifts=shifts,
                penalty=float(config.refine_slope_penalty*np.sum(best[1:]**2)),
                surrogate_score=best_score, model_count=model_count), 'model proposed'


def refine_match(result, point_p, config, *, sift=None, session=None):
    """Mutate final m_pt only after exact validation; debug best/second stay rigid."""
    if not config.local_plane_refine:
        return result
    debug = result.get('region_debug')
    report = dict(status='skipped', reason='original search rejected', applied=False,
                  descriptor_rows=0, descriptor_batches=0)
    result['refinement'] = report
    if debug is not None:
        debug['refinement'] = report
    if result.get('m_pt') is None or result.get('reject_reason') or debug is None:
        return result
    started = time.perf_counter()
    from Algorithm.region_sift_search import evaluate_positions
    from Algorithm.Region_SIFT_Matching import RegionSIFTError, _transform_points
    counts = dict(right_descriptor_rows=0, right_descriptor_batches=0)
    try:
        validate_refinement_config(config)
        if sift is None:
            sift = cv2.SIFT_create(nOctaveLayers=config.sift_n_octave_layers, sigma=config.sift_sigma)
        session = {} if session is None else session
        points = np.asarray(debug['best_points_warp'], np.float32)
        tangent = np.asarray(debug['tangent'], np.float32)
        p = np.asarray(point_p, np.float32)
        relative = np.asarray(debug['left_points'], np.float32) - p
        radius = config.refine_radius_px
        offsets = np.linspace(-radius, radius, int(round(2*radius/config.refine_sample_step_px))+1)
        along = debug['best_along_offset_px']
        half = (config.search_length_px-1)/2
        image = debug['warped_right_gray']
        valid_mask = debug['warped_valid_mask']
        spec = debug['warped_specular_mask']
        report.update(original_center_right=np.asarray(result['m_pt']).copy(),
                      original_center_warp=debug['best_center_warp'].copy(),
                      original_group_score=float(debug['group_score']),
                      original_objective=float(debug['objective_score']))

        def evaluate(positions, center_shifts):
            centers = debug['best_center_warp'][None] + center_shifts[:, None]*tangent
            band = dict(centers_warp=centers,
                        across_offsets=np.full(len(positions), debug['best_across_offset_px']))
            h, w = image.shape
            rounded = np.rint(positions).astype(int)
            x, y = rounded[..., 0], rounded[..., 1]
            inside = np.all((positions[..., 0] >= 0) & (positions[..., 0] <= w-1)
                            & (positions[..., 1] >= 0) & (positions[..., 1] <= h-1), axis=1)
            inside &= np.abs(along + center_shifts) <= half + 1e-6
            anchor_along = along + np.sum((positions-points[None])*tangent, axis=2)
            inside &= np.all(np.abs(anchor_along) <= half + 1e-6, axis=1)
            for j in np.flatnonzero(inside):
                inside[j] &= bool(np.all(valid_mask[y[j], x[j]] > 0))
                if spec is not None and not config.use_masked_sift:
                    inside[j] &= not bool(np.any(spec[y[j], x[j]]))
            indices = np.flatnonzero(inside)
            if not len(indices):
                raise RegionSIFTError('no local candidates pass image/mask/band support')
            # Reuse the existing context and its ROI. Never reuse candidate keys
            # across deformation models: equal centers need not mean equal anchors.
            context = session.get('right_context', {})
            return evaluate_positions(image, positions, indices, valid_mask, spec,
                debug['reject_specular'], band, debug['left_descriptors'], debug['point_metadata'],
                debug['left_frames'], sift, config, session, {}, lambda stage: None,
                counts, left_masked=debug.get('left_masked_packet'),
                right_context_roi=context.get('_roi'))

        positions = points[None] + offsets[:, None, None]*tangent
        table_result = evaluate(positions, offsets)
        table = np.full((len(offsets), len(points)), np.inf)
        for index, values in table_result[5].items():
            table[index] = values
        model, reason = fit_cost_table(offsets, table, relative, debug['point_metadata'], config)
        report.update(reason=reason, table_offsets=offsets, table_distances=table)
        if model is None:
            return result
        report.update(model)
        if abs(model['center_shift_px']) < 1e-8 and np.max(np.abs(model['shifts'])) < 1e-8:
            report['reason'] = 'rigid model already best in local table'
            return result
        # Exact descriptors at the deformed positions, including mask/group/frame gates.
        final_points = points + model['shifts'][:, None]*tangent
        exact = evaluate(final_points[None].astype(np.float32), np.array([model['center_shift_px']]))
        new_group, new_objective = float(exact[0][0]), float(exact[1][0])
        gain = float(debug['objective_score']) - new_objective - model['penalty']
        report.update(refined_group_score=new_group, refined_objective=new_objective,
                      verified_gain=gain)
        if not np.isfinite(gain) or gain < config.refine_min_gain:
            report['reason'] = 'exact Objective improvement below refine_min_gain'
            return result
        if config.max_group_score is not None and new_group > config.max_group_score:
            report['reason'] = 'refined GroupScore exceeds max_group_score'
            return result
        center = debug['best_center_warp'] + model['center_shift_px']*tangent
        right = _transform_points(center.reshape(1, 2), debug['H_lr'])[0]
        if not np.all(np.isfinite(right)):
            raise RegionSIFTError('non-finite refined center')
        report.update(status='applied', applied=True, reason='exact descriptor validation passed',
                      refined_center_warp=center, refined_center_right=right,
                      refined_points_warp=final_points,
                      refined_points_right=_transform_points(final_points, debug['H_lr']),
                      refined_distances=exact[5][0])
        result['m_pt'] = right.astype(np.float32)
        result['method'] += '+LocalPlane'
    except (RegionSIFTError, ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        report['reason'] = 'refinement fallback: ' + str(exc)
    finally:
        report['elapsed_ms'] = (time.perf_counter()-started)*1000
        report['descriptor_rows'] = counts['right_descriptor_rows']
        report['descriptor_batches'] = counts['right_descriptor_batches']
        for key in ('right_descriptor_rows', 'right_descriptor_batches'):
            result.setdefault('timing_counts', {})[key] = result.get('timing_counts', {}).get(key, 0) + counts[key]
    return result
