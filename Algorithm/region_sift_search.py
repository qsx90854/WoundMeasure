"""Coarse-to-fine Region-SIFT search; all caches are scoped to one image pair.

This is an approximate search, not an exhaustive-search equivalence guarantee.
Only same-round descriptors/scores are compared. Missing samples stay missing.
"""
from dataclasses import replace
import time

import numpy as np
import cv2

from Algorithm import region_sift_frames as frames
from Algorithm.region_sift_profiling import on_side, merge_profile


def validate_search_config(config):
    from Algorithm.Region_SIFT_Matching import RegionSIFTError
    for name in ('coarse_along_step_px', 'fine_half_length_px', 'fine_width_px',
                 'fine_step_px', 'valley_shoulder_distance_px'):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise RegionSIFTError(f'{name} must be finite and positive')
    for name in ('adaptive_max_expansions', 'adaptive_cell_increment_px',
                 'adaptive_points_increment', 'valley_min_side_samples'):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RegionSIFTError(f'{name} must be an integer')
    if not 0 <= config.adaptive_max_expansions <= 2:
        raise RegionSIFTError('adaptive_max_expansions must be 0, 1 or 2')
    if min(config.adaptive_cell_increment_px, config.adaptive_points_increment,
           config.valley_min_side_samples) < 1:
        raise RegionSIFTError('Expansion increments and valley side sample count must be positive')
    for name in ('valley_min_relative_depth', 'valley_min_valid_fraction',
                 'valley_max_basin_ratio', 'valley_level_fraction'):
        value = getattr(config, name)
        if not np.isfinite(value) or not 0 < value <= 1:
            raise RegionSIFTError(f'{name} must be in (0, 1]')


def frame_context(image, config, session, side, specular_mask=None, valid_mask=None, roi=None):
    """roi=(x0,y0,x1,y1), masked-SIFT only: crop before building the pyramid.

    context['origin'] is (0,0) when uncropped, else the crop's top-left in the
    original image; callers must subtract it from point coordinates before
    indexing into the context (its own outputs -- sizes, angles, descriptors,
    coverage -- carry no coordinates, so nothing needs shifting back). Safe
    whenever every sampled point's support radius stays inside the crop (see
    masked_sift_descriptor.max_support_radius): pixels outside the crop are
    then never read, so results are identical to running on the full image.
    """
    def build():
        if config.use_masked_sift:
            from Algorithm.masked_sift_descriptor import create_context
            source_image, source_specular, source_valid, origin = image, specular_mask, valid_mask, (0, 0)
            if roi is not None:
                x0, y0, x1, y1 = roi
                source_image = image[y0:y1, x0:x1]
                source_specular = specular_mask[y0:y1, x0:x1] if specular_mask is not None else None
                source_valid = valid_mask[y0:y1, x0:x1] if valid_mask is not None else None
                origin = (x0, y0)
            context = create_context(source_image, source_specular, config, source_valid)
        else:
            context = frames.create_frame_context(image, config)
            origin = (0, 0)
        context['origin'] = origin
        return context
    if session is None:
        return build()
    key = side + '_context'
    cached = session.get(key)
    if cached is None or cached.get('_roi') != roi:
        session[key] = build()
        session[key]['_roi'] = roi
        session[side + '_context_builds'] = session.get(side + '_context_builds', 0) + 1
    return session[key]


@on_side('right')
def evaluate_positions(warped_right, all_positions, valid_indices, warped_valid,
                       warped_specular, reject_specular, band, left_descriptors,
                       point_metadata, left_frames, sift, config, session, cache,
                       timing_next, counts, left_masked=None, funnel=None,
                       right_context_roi=None):
    """Evaluate only new exact-coordinate candidates; cache frames and descriptors.

    The cache belongs to ONE anchor configuration, and is discarded on expansion.
    No quantized coordinate keys: nearby but distinct floating points never alias.
    """
    from Algorithm import Region_SIFT_Matching as matcher
    if cache is None:
        cache = {}
    if funnel is None:
        funnel = {}
    keys = {int(i): tuple(map(float, band['centers_warp'][i])) for i in valid_indices}
    pending = np.array([i for i in valid_indices if keys[int(i)] not in cache], dtype=int)
    counts['candidate_cache_hits'] = len(valid_indices) - len(pending)
    counts['new_candidate_evaluations'] = len(pending)
    funnel['cache_hits'] = int(counts['candidate_cache_hits'])
    funnel['new_candidates'] = int(len(pending))
    funnel.setdefault('new_frame_rejected', 0)
    funnel.setdefault('new_descriptor_group_rejected', 0)
    funnel.setdefault('new_descriptor_group_accepted', 0)
    funnel.setdefault('pair_samples', 0)
    for name in ('left_coverage_invalid_pairs', 'right_coverage_invalid_pairs',
                 'frame_invalid_pairs', 'common_coverage_invalid_pairs'):
        funnel.setdefault(name, 0)
    valid_count_samples, deficient_cell_samples = [], []
    point_count = len(left_descriptors)
    if len(pending):
        timing_next('右圖尺度金字塔與響應圖')
        context = frame_context(warped_right, config, session, 'right', warped_specular, warped_valid,
                                roi=right_context_roi)
        # The context may be a crop; its own outputs carry no coordinates, so
        # only calls that index into it need positions shifted into its local
        # frame (all_positions itself, cached/returned elsewhere, stays
        # absolute). origin is (0,0) whenever uncropped, making every slice
        # below a no-op view of the full array.
        right_origin = np.asarray(context.get('origin', (0, 0)), np.float32)
        _oy, _ox = int(right_origin[1]), int(right_origin[0])
        _ch, _cw = context['image_shape']
        right_context_image = warped_right[_oy:_oy + _ch, _ox:_ox + _cw]
        right_context_valid = warped_valid[_oy:_oy + _ch, _ox:_ox + _cw]
        right_context_specular = (warped_specular[_oy:_oy + _ch, _ox:_ox + _cw]
                                  if warped_specular is not None else None)
        timing_next('右圖尺度選擇與角度估計')
        right_flat = matcher.estimate_dense_sift_frames(
            right_context_image, all_positions[pending].reshape(-1, 2) - right_origin,
            replace(config, min_valid_warp_ratio=1.0)
            if reject_specular and config.specular_check_support and not config.use_masked_sift else config,
            context=context,
            valid_mask=np.where(right_context_specular > 0, 0, right_context_valid).astype(np.uint8)
            if reject_specular and config.specular_check_support and not config.use_masked_sift else right_context_valid)
        timing_next('候選完整support篩選')
        right = {key: np.asarray(value).reshape(len(pending), point_count)
                 for key, value in right_flat.items()}
        complete = np.all(right['valid'], axis=1)
        if config.use_masked_sift:
            from Algorithm.masked_sift_descriptor import group_validity
            complete = group_validity(right['valid'], point_metadata, config)
        counts['incomplete_support_rejected_candidates'] = int(np.count_nonzero(~complete))
        funnel['new_frame_rejected'] += int(np.count_nonzero(~complete))
        funnel['new_frame_accepted'] = funnel.get('new_frame_accepted', 0) + int(np.count_nonzero(complete))
        for i in pending[~complete]:
            cache[keys[int(i)]] = None
        pending = pending[complete]
        right = {key: value[complete] for key, value in right.items()}
        if config.use_masked_sift:
            right['masked_point_valid'] = right['valid'].copy()
        batch_size = max(1, int(config.descriptor_batch_size) // point_count)
        for start in range(0, len(pending), batch_size):
            timing_next('右圖SIFT descriptor（各batch累計）')
            indices = pending[start:start + batch_size]
            rows = np.arange(start, start + len(indices))
            if config.use_masked_sift:
                from Algorithm.masked_sift_descriptor import compute_descriptors, compare_descriptors
                packet = compute_descriptors(context, all_positions[indices].reshape(-1, 2) - right_origin,
                    right['size_px'][rows].reshape(-1), right['angle_deg'][rows].reshape(-1), config)
                descriptors = packet['descriptors']
                packet = {key: value.reshape((len(indices), point_count) + value.shape[1:])
                          for key, value in packet.items()}
                right_coverage_valid = packet['valid_points'].copy()
                right_frame_valid = right['valid'][rows]
                packet['valid_points'] &= right['valid'][rows]
                distances, pair_valid, common = compare_descriptors(left_masked, packet, config)
                gate = group_validity(pair_valid, point_metadata, config, return_details=True)
                group_usable = gate['valid']
                valid_count_samples.extend(np.asarray(gate['valid_count'], int).tolist())
                deficient_cell_samples.extend(np.asarray(gate['deficient_cell_count'], int).tolist())
                funnel['required_valid_points'] = int(gate['required'])
                funnel['pair_samples'] += int(pair_valid.size)
                funnel['left_coverage_invalid_pairs'] += int(
                    len(indices) * np.count_nonzero(~np.asarray(left_masked['valid_points'], bool)))
                funnel['right_coverage_invalid_pairs'] += int(np.count_nonzero(~right_coverage_valid))
                funnel['frame_invalid_pairs'] += int(np.count_nonzero(~right_frame_valid))
                funnel['common_coverage_invalid_pairs'] += int(np.count_nonzero(
                    common < config.masked_min_common_fraction))
                for label, key in (('masked_group_count_rejects', 'enough_points'),
                                   ('masked_cell_count_rejects', 'enough_cells')):
                    counts[label] = counts.get(label, 0) + int(np.count_nonzero(~gate[key]))
                right['masked_point_valid'][rows] = pair_valid
                for field in ('scale_reliable', 'orientation_reliable'):
                    right[field][rows] &= pair_valid
            else:
                descriptors = matcher.compute_descriptors_at_points(
                    warped_right, all_positions[indices].reshape(-1, 2), sift, config,
                    sizes_px=right['size_px'][rows].reshape(-1),
                    angles_deg=right['angle_deg'][rows].reshape(-1),
                    octaves=right['octave'][rows].reshape(-1))
            counts['right_descriptor_batches'] += 1
            counts['right_descriptor_rows'] += len(descriptors)
            if len(descriptors) != len(indices) * point_count:
                raise matcher.RegionSIFTError('right descriptor batch has incomplete rows')
            timing_next('L2距離、群組評分與候選保存（累計）')
            matrices = descriptors.reshape(len(indices), point_count, 128)
            if not config.use_masked_sift:
                distances = np.linalg.norm(matrices - left_descriptors[None, :, :], axis=2)
            components = matcher.score_candidate_groups(distances, point_metadata, left_frames,
                {key: value[rows] for key, value in right.items()}, config)
            for j, i in enumerate(indices):
                if config.use_masked_sift and not group_usable[j]:
                    cache[keys[int(i)]] = None
                    counts['masked_coverage_rejected_candidates'] = counts.get('masked_coverage_rejected_candidates', 0) + 1
                    counts['masked_invalid_point_pairs'] = counts.get('masked_invalid_point_pairs', 0) + int(np.count_nonzero(~pair_valid[j]))
                    continue
                if config.use_masked_sift:
                    funnel['new_descriptor_group_accepted'] += 1
                cache[keys[int(i)]] = dict(
                    group=float(components['group_score'][j]),
                    trimmed=float(components['trimmed_mean'][j]),
                    balanced=float(components['balanced_mean'][j]),
                    penalty=float(components['frame_penalty'][j]),
                    objective=float(components['objective_without_epi'][j]
                                    + config.epipolar_penalty_weight * abs(band['across_offsets'][i])),
                    distances=distances[j].astype(np.float32),
                    descriptors=matrices[j].astype(np.float32),
                    frames={key: value[start+j].copy() for key, value in right.items()})
                if config.use_masked_sift:
                    cache[keys[int(i)]]['frames']['masked_valid_fraction'] = packet['valid_fraction'][j].copy()
                    cache[keys[int(i)]]['frames']['masked_common_fraction'] = common[j].copy()
                    cache[keys[int(i)]]['frames']['masked_point_valid'] = pair_valid[j].copy()
    timing_next('候選快取整理')
    valid_indices = np.array([i for i in valid_indices if cache[keys[int(i)]] is not None], dtype=int)
    funnel['new_descriptor_group_rejected'] = int(
        counts.get('masked_coverage_rejected_candidates', 0))
    funnel['final_scoreable_candidates'] = int(len(valid_indices))
    if valid_count_samples:
        values = np.asarray(valid_count_samples, int)
        funnel.update(valid_points_min=int(values.min()),
                      valid_points_median=float(np.median(values)),
                      valid_points_max=int(values.max()))
    if deficient_cell_samples:
        values = np.asarray(deficient_cell_samples, int)
        funnel.update(deficient_cells_min=int(values.min()),
                      deficient_cells_median=float(np.median(values)),
                      deficient_cells_max=int(values.max()))
    if not len(valid_indices):
        raise matcher.RegionSIFTError('匹配失敗: 自製 SIFT 無候選具足夠有效／共同覆蓋或影像 support'
            if config.use_masked_sift else '匹配失敗: no search candidate has complete reflection-free per-scale SIFT support')
    records = [cache[keys[int(i)]] for i in valid_indices]
    arrays = []
    for name in ('group', 'objective', 'trimmed', 'balanced', 'penalty'):
        values = np.full(len(all_positions), np.inf, dtype=np.float64)
        values[valid_indices] = [row[name] for row in records]
        arrays.append(values)
    right = {key: np.stack([row['frames'][key] for row in records]) for key in records[0]['frames']}
    return (*arrays, {int(i): row['distances'] for i, row in zip(valid_indices, records)},
            {int(i): row['descriptors'] for i, row in zip(valid_indices, records)},
            right, {int(i): row for row, i in enumerate(valid_indices)})


def _local_minima(values):
    """Collapse exactly flat bottoms; edges are competitors, never trusted winners."""
    result, i = [], 0
    while i < len(values):
        if not np.isfinite(values[i]):
            i += 1
            continue
        end = i
        while end+1 < len(values) and values[end+1] == values[i]:
            end += 1
        left = values[i-1] if i else np.inf
        right = values[end+1] if end+1 < len(values) else np.inf
        if values[i] <= left and values[i] <= right:
            result.append((i+end)//2)
        i = end+1
    return result


def analyze_valley(debug, config):
    """Select by Objective; require a two-sided GroupScore depression and uniqueness."""
    along = np.asarray(debug['candidate_along_offsets'], float)
    x = np.unique(along)
    objective = np.full(len(x), np.inf)
    group = np.full(len(x), np.inf)
    across = np.zeros(len(x))
    for j, value in enumerate(x):
        ids = np.flatnonzero(along == value)
        index = ids[np.argmin(np.asarray(debug['candidate_objective_scores'])[ids])]
        objective[j] = debug['candidate_objective_scores'][index]
        group[j] = debug['candidate_group_scores'][index]
        across[j] = debug['candidate_across_offsets'][index]
    finite = np.isfinite(objective) & np.isfinite(group)
    valid_count = int(np.count_nonzero(finite))
    required_count = int(np.ceil(len(x) * config.valley_min_valid_fraction))
    out = dict(clear=False, reason='no valid scores', x=x, group=group, objective=objective,
               valid_fraction=float(np.mean(finite)), valid_count=valid_count,
               sample_count=int(len(x)), required_valid_count=required_count,
               relative_depth=0., basin_ratio=float('nan'))
    if not finite.any():
        return out
    # Same deterministic tie preference as the matching engine.
    best = min(np.flatnonzero(finite), key=lambda j: (objective[j], np.hypot(x[j], across[j]), j))
    out.update(along=float(x[best]), across=float(across[best]), best_index=int(best))
    if np.mean(finite) < config.valley_min_valid_fraction:
        out['reason'] = 'insufficient valid coarse coverage'
        return out
    start = end = best
    while start > 0 and finite[start-1]:
        start -= 1
    while end+1 < len(x) and finite[end+1]:
        end += 1
    distance = config.valley_shoulder_distance_px
    left = np.arange(start, best)[x[best]-x[start:best] <= distance+1e-6]
    right = np.arange(best+1, end+1)[x[best+1:end+1]-x[best] <= distance+1e-6]
    if min(len(left), len(right)) < config.valley_min_side_samples:
        out['reason'] = 'valley at search boundary or next to invalid samples'
        return out
    shoulder = min(float(np.max(group[left])), float(np.max(group[right])))
    depth = max(0., shoulder-group[best]) / max(abs(shoulder), 1e-12)
    out['relative_depth'] = depth
    if depth < config.valley_min_relative_depth:
        out['reason'] = 'GroupScore valley is too shallow or flat'
        return out
    level = group[best] + config.valley_level_fraction * (shoulder-group[best])
    lo = hi = best
    while lo > start and group[lo-1] < level:
        lo -= 1
    while hi < end and group[hi+1] < level:
        hi += 1
    out['basin_bounds'] = (float(x[lo]), float(x[hi]))
    if lo == start or hi == end:
        out['reason'] = 'valley bottom is truncated by boundary or invalid samples'
        return out
    if max(x[best]-x[lo], x[hi]-x[best]) > config.fine_half_length_px:
        out['reason'] = 'valley bottom exceeds fine search window'
        return out
    competitors = [j for j in _local_minima(objective) if j < lo or j > hi]
    if competitors:
        other = min(competitors, key=lambda j: objective[j])
        ratio = objective[best]/objective[other] if objective[other] > 1e-12 else 1.
        out.update(basin_ratio=float(ratio), competitor_along=float(x[other]))
        if ratio > config.valley_max_basin_ratio:
            out['reason'] = 'competing separated valleys have similar Objective'
            return out
    out.update(clear=True, reason='clear bounded valley')
    return out


def _union_offsets(*sets):
    return np.array(sorted(set(tuple(map(float, pair)) for points in sets for pair in points)),
                    dtype=np.float32).reshape(-1, 2)


def _stage_thresholds(config, point_count):
    keep = (int(config.keep_best_count) if config.keep_best_count is not None
            else int(round(point_count * config.keep_best_ratio)))
    required = max(1, min(point_count, keep),
                   int(np.ceil(point_count * config.masked_min_group_fraction)))
    return dict(use_masked_sift=bool(config.use_masked_sift), required_valid_points=required,
        masked_min_valid_fraction=float(config.masked_min_valid_fraction),
        masked_min_cell_fraction=float(config.masked_min_cell_fraction),
        masked_min_common_fraction=float(config.masked_min_common_fraction),
        masked_min_group_fraction=float(config.masked_min_group_fraction),
        keep_best_ratio=float(config.keep_best_ratio), keep_best_count=config.keep_best_count,
        masked_min_points_per_cell=int(config.masked_min_points_per_cell),
        masked_max_deficient_cells=int(config.masked_max_deficient_cells),
        valley_min_valid_fraction=float(config.valley_min_valid_fraction),
        valley_min_relative_depth=float(config.valley_min_relative_depth),
        valley_max_basin_ratio=float(config.valley_max_basin_ratio),
        fine_half_length_px=float(config.fine_half_length_px),
        search_length_px=float(config.search_length_px))


def format_stage_funnel(stage):
    """Human-readable per-stage rejection funnel and parameter pointers."""
    f = stage.get('candidate_funnel') or {}
    t = stage.get('thresholds') or {}
    lines = []
    requested = int(f.get('requested_candidates', stage.get('candidate_count', 0)))
    in_bounds = int(f.get('in_bounds_candidates', requested))
    support = int(f.get('warp_support_candidates', in_bounds))
    center = int(f.get('center_mask_candidates', support))
    new = int(f.get('new_candidates', stage.get('new_candidates', 0)))
    cached = int(f.get('cache_hits', stage.get('cache_hits', 0)))
    frame_ok = int(f.get('new_frame_accepted', new))
    group_ok = int(f.get('new_descriptor_group_accepted', max(0, frame_ok-int(
        f.get('new_descriptor_group_rejected', 0)))))
    scoreable = int(f.get('final_scoreable_candidates', stage.get('valid_candidate_count', 0)))
    lines.append(f"    漏斗: requested={requested} -> inBounds={in_bounds} -> warpSupport={support} "
                 f"-> centerMask={center} | cache={cached}, new={new} -> frameOK={frame_ok} "
                 f"-> newGroupOK={group_ok} -> scoreable(total)={scoreable}")
    if t.get('use_masked_sift'):
        required = int(f.get('required_valid_points', t.get('required_valid_points', 0)))
        if 'valid_points_min' in f:
            lines.append(f"    點對: valid/new candidate min/median/max="
                         f"{f['valid_points_min']}/{f['valid_points_median']:.1f}/{f['valid_points_max']} "
                         f"(需要 >= {required}/{stage['point_count']}) | deficient cells "
                         f"min/median/max={f.get('deficient_cells_min', 0)}/"
                         f"{f.get('deficient_cells_median', 0):.1f}/{f.get('deficient_cells_max', 0)} "
                         f"(允許 <= {t['masked_max_deficient_cells']})")
        samples = int(f.get('pair_samples', 0))
        if samples:
            lines.append(f"    點對無效來源(可重疊, /{samples}): leftCoverage="
                         f"{f.get('left_coverage_invalid_pairs', 0)}, rightCoverage="
                         f"{f.get('right_coverage_invalid_pairs', 0)}, frame="
                         f"{f.get('frame_invalid_pairs', 0)}, commonCoverage="
                         f"{f.get('common_coverage_invalid_pairs', 0)}")
        lines.append(f"    當前門檻: point L/R valid>={t['masked_min_valid_fraction']:.2f}, "
                     f"SIFT-subcell common>={t['masked_min_cell_fraction']:.2f}, "
                     f"point common>={t['masked_min_common_fraction']:.2f}; group valid>={required}, "
                     f"cell points>={t['masked_min_points_per_cell']}, "
                     f"deficient cells<={t['masked_max_deficient_cells']}")
    valley = stage.get('valley') or {}
    if valley:
        lines.append(f"    低谷輸入: valid along={valley.get('valid_count', 0)}/"
                     f"{valley.get('sample_count', 0)} ({100*valley.get('valid_fraction', 0):.1f}%), "
                     f"需要 >= {valley.get('required_valid_count', 0)} "
                     f"({100*t.get('valley_min_valid_fraction', 0):.1f}%)")
    reason = stage.get('reason', '')
    hints = []
    if in_bounds < requested:
        hints.append('座標越界：檢查幾何預測；對應 search_length_px/search_width_px 與點擊位置邊界')
    if support < in_bounds:
        hints.append('warp support 不足：對應 descriptor_border_margin_px、min_valid_warp_ratio')
    if int(f.get('new_descriptor_group_rejected', 0)):
        hints.append('有效點不足：先檢查反光遮罩；點對門檻對應 masked_min_valid_fraction / '
                     'masked_min_cell_fraction / masked_min_common_fraction，群組總數對應 '
                     'masked_min_group_fraction（同時受 keep_best_ratio 下限約束）')
    if (int(f.get('new_descriptor_group_rejected', 0))
            and f.get('deficient_cells_max', 0) > t.get('masked_max_deficient_cells', 0)):
        hints.append('cell 分布不足：對應 masked_min_points_per_cell / masked_max_deficient_cells')
    if reason == 'insufficient valid coarse coverage':
        hints.append('沿線有效率不足：對應 valley_min_valid_fraction；若上游 group 拒絕很多，先處理遮罩/覆蓋門檻')
    elif reason == 'GroupScore valley is too shallow or flat':
        hints.append('低谷太淺：對應 valley_min_relative_depth；降低會增加模糊解風險')
    elif reason in ('valley at search boundary or next to invalid samples',
                    'valley bottom is truncated by boundary or invalid samples'):
        hints.append('低谷被邊界/無效點截斷：對應 search_length_px 或先改善候選有效覆蓋')
    elif reason == 'valley bottom exceeds fine search window':
        hints.append('低谷寬於細搜範圍：對應 fine_half_length_px')
    elif reason == 'competing separated valleys have similar Objective':
        hints.append('存在相近雙低谷：對應 valley_max_basin_ratio；提高會增加錯配風險')
    if hints:
        lines.extend('    對應參數: ' + hint for hint in hints)
    return lines


def run_two_stage(engine, left, right, point, cand, K_L, *, sift, config, left_cache, **kwargs):
    from Algorithm.Region_SIFT_Matching import _validate_config, _centered_offsets, RegionSIFTError
    started = time.perf_counter()
    history, totals, counts = [], {}, {'right_descriptor_rows': 0, 'right_descriptor_batches': 0,
        'candidate_cache_hits': 0, 'specular_center_rejected_candidates': 0,
        'incomplete_support_rejected_candidates': 0, 'new_candidate_evaluations': 0,
        'masked_coverage_rejected_candidates': 0, 'masked_invalid_point_pairs': 0,
        'masked_group_count_rejects': 0, 'masked_cell_count_rejects': 0}
    detail_profile = {}
    session = {}
    result = dict(m_pt=None, method='', region_debug=None, reject_reason=None)
    initial_cache = left_cache if left_cache is not None else {}
    fine_bounds = None
    effective = config

    def evaluate(offsets, cache, round_cache, label, round_index):
        nonlocal result
        result = engine(left, right, point, cand, K_L, sift=sift, config=effective,
                        left_cache=round_cache, _session=session, _offsets=offsets,
                        _candidate_cache=cache, **kwargs)
        session.setdefault('initial_left_cache_hit', bool(result.get('timing_counts', {}).get('left_cache_hit', False)))
        merge_profile(detail_profile, result.get('detail_profile', {}))
        for key, value in result.get('timing_ms', {}).items():
            totals[key] = totals.get(key, 0.) + value
        for key in counts:
            counts[key] += int(result.get('timing_counts', {}).get(key, 0))
        stage = dict(stage=label, round=round_index, cell_width=effective.cell_width_px,
            cell_height=effective.cell_height_px, points_per_cell=effective.points_per_cell,
            point_count=effective.grid_rows*effective.grid_cols*effective.points_per_cell+1,
            elapsed_ms=result.get('elapsed_ms', 0.), candidate_count=len(offsets),
            descriptor_rows=result.get('timing_counts', {}).get('right_descriptor_rows', 0),
            cache_hits=result.get('timing_counts', {}).get('candidate_cache_hits', 0),
            new_candidates=result.get('timing_counts', {}).get('new_candidate_evaluations', 0),
            detail_profile=result.get('detail_profile', {}),
            reason=result.get('reject_reason') or 'scored', fine_bounds=fine_bounds,
            candidate_funnel=dict(result.get('candidate_funnel', {})),
            thresholds=_stage_thresholds(effective,
                effective.grid_rows*effective.grid_cols*effective.points_per_cell+1))
        debug = result.get('region_debug')
        if debug is not None:
            for key in ('candidate_along_offsets', 'candidate_across_offsets',
                        'candidate_group_scores', 'candidate_objective_scores'):
                stage[key] = debug[key].copy()
            stage['valid_candidate_count'] = debug['valid_candidate_count']
        history.append(stage)
        return stage, debug

    try:
        _validate_config(config)
        if sift is None:
            sift = cv2.SIFT_create(nOctaveLayers=config.sift_n_octave_layers, sigma=config.sift_sigma)
        along = _centered_offsets(config.search_length_px, config.coarse_along_step_px)
        half = (config.search_length_px-1.)/2.
        width_half = max(0., (config.search_width_px-1.)/2.)
        for iteration in range(config.adaptive_max_expansions+1):
            effective = replace(config,
                cell_width_px=config.cell_width_px+iteration*config.adaptive_cell_increment_px,
                cell_height_px=config.cell_height_px+iteration*config.adaptive_cell_increment_px,
                points_per_cell=config.points_per_cell+iteration*config.adaptive_points_increment)
            cache = {}
            round_cache = initial_cache.setdefault('two_stage_rounds', {}).setdefault(iteration, {})
            offsets = np.column_stack([along, np.zeros(len(along))])
            stage, debug = evaluate(offsets, cache, round_cache, 'coarse', iteration)
            if debug is None:
                break  # Geometry/support failures are not shallow-score valleys.
            valley = analyze_valley(debug, effective)
            stage.update(valley=valley, reason=valley['reason'])
            if not valley['clear'] and config.coarse_side_rescue and width_half > 0:
                sides = [(a, b) for a in along for b in (-width_half, width_half)]
                offsets = _union_offsets(offsets, sides)
                stage, debug = evaluate(offsets, cache, round_cache, 'coarse+sides', iteration)
                if debug is None:
                    break
                valley = analyze_valley(debug, effective)
                stage.update(valley=valley, reason=valley['reason'])
            if not valley['clear']:
                result.update(m_pt=None, method='', reject_reason='匹配失敗: coarse localization unclear: '+valley['reason'])
                continue
            center = valley['along']
            delta = _centered_offsets(2*config.fine_half_length_px+1, config.fine_step_px)
            fine_along = center+delta
            fine_along = fine_along[(fine_along >= -half-1e-6) & (fine_along <= half+1e-6)]
            # A local segment of the original epipolar band, not a shifted wider band.
            fine_across = _centered_offsets(min(config.fine_width_px, config.search_width_px), config.fine_step_px)
            fine_bounds = (float(fine_along.min()), float(fine_along.max()))
            fine = [(a, b) for a in fine_along for b in fine_across]
            offsets = _union_offsets(offsets, fine)
            stage, debug = evaluate(offsets, cache, round_cache, 'fine+global-coarse', iteration)
            if debug is None:
                break
            stage['fine_candidate_count'] = len(fine)
            best = debug['best_along_offset_px']
            final_valley = analyze_valley(debug, effective)
            stage['valley'] = final_valley
            if best <= fine_bounds[0]+1e-6 or best >= fine_bounds[1]-1e-6:
                result.update(m_pt=None, method='', reject_reason='匹配失敗: fine optimum reaches/outside local search boundary')
            elif not final_valley['clear']:
                result.update(m_pt=None, method='', reject_reason='匹配失敗: final valley unclear: '+final_valley['reason'])
            stage['reason'] = result.get('reject_reason') or 'accepted after fine search'
            if result.get('m_pt') is not None:
                result['method'] += f'+CoarseFine(expand={iteration})'
            break
    except (RegionSIFTError, ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        result.update(m_pt=None, method='', reject_reason=f'匹配失敗: two-stage search: {exc}')
    elapsed = (time.perf_counter()-started)*1000.
    totals['兩階段控制與低谷分析'] = max(0., elapsed-sum(totals.values()))
    counts.update(reject_specular=bool(kwargs.get('reject_specular') or config.use_masked_sift),
        specular_check_support=bool(config.specular_check_support and not config.use_masked_sift),
        left_cache_hit=session.get('initial_left_cache_hit', False),
        left_context_builds=session.get('left_context_builds', 0),
        right_context_builds=session.get('right_context_builds', 0),
        warp_builds=session.get('warp_builds', 0), search_stages=len(history))
    result.update(search_history=history, effective_config=effective,
                  timing_ms=totals, timing_counts=counts, elapsed_ms=elapsed,
                  detail_profile=detail_profile)
    if result.get('region_debug') is not None:
        result['region_debug'].update(search_history=history, fine_bounds=fine_bounds,
            search_status=result.get('reject_reason') or 'accepted', requested_config=config,
            timing_ms=totals, timing_counts=counts, elapsed_ms=elapsed, detail_profile=detail_profile)
    return result
