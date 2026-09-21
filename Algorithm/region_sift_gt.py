"""GT-height projection and exact-position scoring; never changes a match."""
from dataclasses import replace
import numpy as np
from Algorithm.Region_SIFT_Matching import (
    _transform_points, _eroded_valid_mask, estimate_dense_sift_frames,
    compute_descriptors_at_points,
)
from Algorithm.region_sift_scoring import score_candidate_groups


def project_gt(debug, height_mm):
    g = debug['diagnostic_geometry']
    plane = debug['diagnostic_height_plane']
    n, c = np.asarray(plane['n'], float), np.asarray(plane['c'], float)
    if n.shape != (3,) or c.shape != (3,) or not np.all(np.isfinite([n, c])):
        raise ValueError('Height plane is unavailable')
    h = float(height_mm)
    if not np.isfinite(h):
        raise ValueError('GT height must be finite')
    p = np.asarray(debug['left_points'][-1], float)
    ray = np.linalg.solve(np.asarray(g['K_L']), np.r_[p, 1.])
    denom = float(n @ ray)
    if abs(denom) < 1e-10:
        raise ValueError('Left ray is parallel to the GT height plane')
    # Match the application's signed dot(n, X-c) - offset height convention.
    xyz = ray * ((float(n @ c) + h + float(plane['offset'])) / denom)
    right_xyz = np.asarray(g['R']) @ xyz + np.asarray(g['t']).reshape(3)
    if (not np.all(np.isfinite([xyz, right_xyz])) or xyz[2] <= 0 or right_xyz[2] <= 0):
        raise ValueError('GT intersection lies behind an image viewpoint')
    q = np.asarray(g['K_R']) @ right_xyz
    right = q[:2] / q[2]
    warp = _transform_points(right[None], debug['H_rl'])[0]
    delta = warp - debug['seed_on_line']
    along, across = float(delta @ debug['tangent']), float(delta @ debug['normal'])
    cfg = debug['config']
    inside = (abs(along) <= max(0, (cfg.search_length_px-1)/2) + 1e-5
              and abs(across) <= max(0, (cfg.search_width_px-1)/2) + 1e-5)
    centers = np.asarray(debug['candidate_centers_warp'])
    nearest = int(np.argmin(np.linalg.norm(centers-warp, axis=1)))
    return dict(height_mm=h, xyz=xyz, center_right=right, center_warp=warp,
                along=along, across=across, inside_band=inside,
                nearest_index=nearest, nearest_distance_px=float(np.linalg.norm(centers[nearest]-warp)),
                nearest_objective=float(debug['candidate_objective_scores'][nearest]))


def evaluate_gt(debug, height_mm):
    """Score the exact GT shift with the same anchors, frames and validity gates.

    An out-of-band point may still be scored diagnostically. Invalid support
    is never bypassed. The whole region uses the matcher's shared displacement,
    not independent GT-plane reprojection of each anchor.
    """
    out = project_gt(debug, height_mm)
    cfg = debug['config']
    points = np.asarray(debug['left_points']) + (out['center_warp']-debug['left_points'][-1])
    out.update(points_warp=points, scoreable=False, reason='', frames=None)
    image = debug['warped_right_gray']
    height, width = image.shape
    bounds = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) &
              (points[:, 0] <= width-1) & (points[:, 1] >= 0) & (points[:, 1] <= height-1))
    def reject(reason, bad):
        out['reason'] = reason
        out['invalid_anchor_ids'] = (np.flatnonzero(bad)+1).tolist()
        return out
    if not bounds.all():
        return reject('anchor outside warped image', ~bounds)
    xy = np.rint(points).astype(int)
    x, y = xy[:, 0], xy[:, 1]
    valid = debug['warped_valid_mask']
    border = _eroded_valid_mask(valid, int(cfg.descriptor_border_margin_px), float(cfg.min_valid_warp_ratio))
    bad = border[y, x] == 0
    if bad.any():
        return reject('warp/border validity rejected', bad)
    spec = debug.get('warped_specular_mask') if debug.get('reject_specular') else None
    if spec is not None and not cfg.use_masked_sift:
        bad = spec[y, x] != 0
        if bad.any():
            return reject('specular anchor center', bad)
    masked = spec is not None and cfg.specular_check_support and not cfg.use_masked_sift
    context = None
    if cfg.use_masked_sift:
        from Algorithm.masked_sift_descriptor import create_context, compute_descriptors, compare_descriptors, group_validity
        context = create_context(image, spec, cfg, valid)
    frames = estimate_dense_sift_frames(image, points,
        replace(cfg, min_valid_warp_ratio=1.) if masked else cfg,
        context=context,
        valid_mask=np.where(spec > 0, 0, valid).astype(np.uint8) if masked else valid)
    out['frames'] = frames
    if not cfg.use_masked_sift and not frames['valid'].all():
        bad = ~frames['valid']
        reason = 'incomplete image/warp scale support'
        if masked:
            without_spec = estimate_dense_sift_frames(image, points,
                replace(cfg, min_valid_warp_ratio=1.), valid_mask=valid)
            spec_only = bad & without_spec['valid']
            out['specular_support_anchor_ids'] = (np.flatnonzero(spec_only)+1).tolist()
            reason += '; specular-support IDs=' + str(out['specular_support_anchor_ids'])
        return reject(reason, bad)
    if cfg.use_masked_sift:
        packet = compute_descriptors(context, points, frames['size_px'], frames['angle_deg'], cfg)
        packet['valid_points'] &= frames['valid']
        distances, usable, common = compare_descriptors(debug['left_masked_packet'], packet, cfg)
        frames['masked_valid_fraction'] = packet['valid_fraction']
        frames['masked_common_fraction'] = common
        frames['masked_point_valid'] = usable
        frames['scale_reliable'] &= usable
        frames['orientation_reliable'] &= usable
        if not group_validity(usable[None], debug['point_metadata'], cfg)[0]:
            return reject('custom SIFT group/cell valid pair coverage insufficient', ~usable)
    else:
        desc = compute_descriptors_at_points(image, points, config=cfg,
            sizes_px=frames['size_px'], angles_deg=frames['angle_deg'], octaves=frames['octave'])
        distances = np.linalg.norm(desc-debug['left_descriptors'], axis=1)
    scores = score_candidate_groups(distances[None], debug['point_metadata'],
        debug['left_frames'], {k: v[None] for k, v in frames.items()}, cfg)
    group = float(scores['group_score'][0])
    objective = float(scores['objective_without_epi'][0]) + cfg.epipolar_penalty_weight*abs(out['across'])
    out.update(scoreable=True, reason='valid exact-position score', distances=distances,
               keep_mask=scores['keep_mask'][0], group_score=group, objective_score=objective,
               trimmed_score=float(scores['trimmed_mean'][0]),
               balanced_score=float(scores['balanced_mean'][0]),
               frame_penalty=float(scores['frame_penalty'][0]))
    return out
