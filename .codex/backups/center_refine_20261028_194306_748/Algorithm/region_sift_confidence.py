"""Read-only evidence analysis, with explicit opt-in decision gates.

No SIFT calls, image pyramids, plots or descriptor recomputation occur here.
Intervals describe sampled candidates, not calibrated confidence intervals.
"""
import time
import numpy as np


def validate_config(config):
    if config.confidence_mode not in ('legacy', 'shadow', 'active'):
        raise ValueError('confidence_mode must be legacy, shadow or active')
    for name in ('confidence_score_band_relative', 'confidence_basin_prominence_relative',
                 'confidence_min_cell_stability'):
        if not np.isfinite(getattr(config, name)) or not 0 < getattr(config, name) <= 1:
            raise ValueError(f'{name} must be in (0, 1]')
    if not np.isfinite(config.confidence_max_width_px) or config.confidence_max_width_px <= 0:
        raise ValueError('confidence_max_width_px must be positive')
    value = config.confidence_height_span_max_mm
    if value is not None and (not np.isfinite(value) or value <= 0):
        raise ValueError('confidence_height_span_max_mm must be positive or None')


def analyze_profile(debug, config):
    """Group significant 1-D basins after minimizing across the epipolar width.

    Plateaus form a single minimum; invalid samples and large physical gaps
    split components. Shallow ripples do not automatically create competitors.
    """
    along = np.asarray(debug['candidate_along_offsets'], float)
    obj = np.asarray(debug['candidate_objective_scores'], float)
    x = np.unique(along)
    y = np.array([np.min(obj[along == a]) for a in x])
    valid = np.isfinite(y)
    labels = np.full(len(x), -1, int)
    result = dict(status='unscorable', reason_codes=['no_valid_candidates'],
                  plausible_candidate_indices=np.array([], int), localization_width_px=float('nan'),
                  profile_x=x, profile_objective=y, basin_labels=labels, basins=[])
    if not valid.any():
        return result
    gap = max(config.coarse_along_step_px if config.two_stage_search else config.search_along_step_px,
              config.fine_step_px)
    starts = [i for i in range(len(x)) if valid[i] and
              (i == 0 or not valid[i-1] or x[i]-x[i-1] > gap+1e-5)]
    basins = []
    for start in starts:
        end = start
        while end+1 < len(x) and valid[end+1] and x[end+1]-x[end] <= gap+1e-5:
            end += 1
        minima = []
        i = start
        while i <= end:
            j = i
            while j < end and y[j+1] == y[i]:
                j += 1
            if (i == start or y[i] < y[i-1]) and (j == end or y[j] < y[j+1]):
                minima.append((i+j)//2)
            i = j+1
        # Remove the shallowest adjacent ripple first, then recompute the ridge
        # between the surviving minima over their ENTIRE interval. Pairwise
        # cuts on the original minima let a chain of tiny ripples hide a large
        # ridge and incorrectly merge two genuinely competing valleys.
        while len(minima) > 1:
            links = []
            for j, (a, b) in enumerate(zip(minima, minima[1:])):
                ridge = float(np.max(y[a:b+1]))
                required = config.confidence_basin_prominence_relative * max(abs(y[a]), abs(y[b]), 1e-12)
                links.append(((ridge-max(y[a], y[b]))/required, j))
            prominence, j = min(links)
            if prominence >= 1.:
                break
            a, b = minima[j], minima[j+1]
            # Keep the deeper minimum; equal minima use the same spatial
            # preference as candidate selection rather than a scan-direction bias.
            remove = j+1 if (y[a], abs(x[a]), a) <= (y[b], abs(x[b]), b) else j
            minima.pop(remove)
        cuts = []
        for a, b in zip(minima, minima[1:]):
            peak = a + int(np.argmax(y[a:b+1]))
            required = config.confidence_basin_prominence_relative * max(abs(y[a]), abs(y[b]), 1e-12)
            if y[peak]-max(y[a], y[b]) >= required:
                cuts.append(peak)
        lo = start
        for hi in cuts + [end]:
            if hi < lo:
                continue
            bid = len(basins)
            labels[lo:hi+1] = bid
            k = lo + int(np.argmin(y[lo:hi+1]))
            basins.append(dict(id=bid, start=lo, end=hi, minimum_index=k,
                               along=float(x[k]), objective=float(y[k]),
                               component_start=start, component_end=end))
            lo = hi+1
    best_candidate = int(debug.get('best_candidate_index', np.nanargmin(obj)))
    if not np.isfinite(obj[best_candidate]):
        best_candidate = int(np.nanargmin(obj))
    best_x = int(np.searchsorted(x, along[best_candidate]))
    bid = int(labels[best_x])
    basin = basins[bid]
    level = float(obj[best_candidate] + config.confidence_score_band_relative * max(abs(obj[best_candidate]), 1e-12))
    lo = hi = best_x
    while lo > basin['start'] and y[lo-1] <= level:
        lo -= 1
    while hi < basin['end'] and y[hi+1] <= level:
        hi += 1
    nearby = np.flatnonzero(np.isfinite(obj) & (obj <= level) & (along >= x[lo]) & (along <= x[hi]))
    competitors = [b for b in basins if b['id'] != bid]
    other = min(competitors, key=lambda b: b['objective']) if competitors else None
    ratio = (float(obj[best_candidate]/other['objective']) if other and other['objective'] > 1e-12
             else (1. if other else float('nan')))
    # Reaching an invalid segment endpoint is also an unbounded interval.
    bounded = lo > basin['component_start'] and hi < basin['component_end']
    required_valid = int(np.ceil(len(x)*config.valley_min_valid_fraction))
    reasons = []
    if int(valid.sum()) < required_valid:
        reasons.append('insufficient_profile_coverage')
    if np.ptp(y[valid]) <= config.flat_score_relative_tolerance * (1. if config.normalize_descriptors else 512.):
        reasons.append('flat_objective')
    if not bounded:
        reasons.append('localization_unbounded')
    if other and ratio > config.valley_max_basin_ratio:
        reasons.append('competing_basins')
    result.update(status='scored_uncertain' if reasons else 'accepted', reason_codes=reasons,
                  best_basin=bid, competing_basin=other, basin_ratio=ratio, basins=basins,
                  plausible_candidate_indices=nearby, localization_width_px=float(x[hi]-x[lo]),
                  localization_bounds_px=(float(x[lo]), float(x[hi])), bounded=bounded,
                  score_band_limit=level, best_candidate_index=best_candidate,
                  interval_profile_indices=(lo, hi), valid_fraction=float(valid.mean()))
    return result


def search_valley(debug, config, legacy):
    """Cheap active-mode coarse routing; no cell or height work per round."""
    if config.confidence_mode != 'active':
        return legacy
    report = analyze_profile(debug, config)
    out = dict(legacy)
    out.update(clear=report['status'] == 'accepted',
               reason='bounded Objective basin' if report['status'] == 'accepted'
               else ', '.join(report['reason_codes']), confidence=report)
    out['legacy_basin_ratio'] = legacy.get('basin_ratio', float('nan'))
    out['basin_ratio'] = report.get('basin_ratio', float('nan'))
    return out


def _aggregate(values, valid, cells, config, keep_mask=None):
    n = values.shape[1]
    k = min(n, max(1, int(config.keep_best_count) if config.keep_best_count is not None
                   else int(round(n*config.keep_best_ratio))))
    if keep_mask is None:
        order = np.argsort(np.where(valid, values, np.inf), axis=1, kind='stable')
        keep_mask = np.zeros_like(valid)
        np.put_along_axis(keep_mask, order[:, :k], True, axis=1)
    trim = np.sum(np.where(keep_mask, values, 0.), axis=1)/np.maximum(keep_mask.sum(axis=1), 1)
    balanced = np.mean(np.stack([values[:, cells == c].mean(axis=1) for c in np.unique(cells)]), axis=0)
    return (1-config.group_balance_weight)*trim+config.group_balance_weight*balanced, keep_mask


def analyze_evidence(debug, config, profile):
    ids = np.asarray(debug['evidence_candidate_indices'], int)
    costs = np.asarray(debug['evidence_distances'], float)
    valid = np.asarray(debug['evidence_pair_valid'], bool)
    raw = np.asarray(debug['evidence_texture_distances'], float)
    missing = np.asarray(debug['evidence_missing_penalties'], float)
    common = np.asarray(debug['evidence_common_fractions'], float)
    metadata = debug['point_metadata']
    cells = np.array([int(m['cell_row'])*config.grid_cols+int(m['cell_col']) for m in metadata])
    _, keep = _aggregate(costs, valid, cells, config)
    texture, _ = _aggregate(np.where(valid, raw, 0.), valid, cells, config, keep)
    penalty, _ = _aggregate(missing, valid, cells, config, keep)
    # Attribution uses the same KEEP and cell weights as the original score.
    debug['candidate_texture_contributions'] = np.full(len(debug['candidate_objective_scores']), np.nan)
    debug['candidate_missing_contributions'] = debug['candidate_texture_contributions'].copy()
    debug['candidate_texture_contributions'][ids] = texture
    debug['candidate_missing_contributions'][ids] = penalty
    def summary(index):
        row = int(np.flatnonzero(ids == index)[0])
        return dict(index=int(index), texture_contribution=float(texture[row]),
                    missing_contribution=float(penalty[row]), valid_pairs=int(valid[row].sum()),
                    total_pairs=costs.shape[1], common_coverage=float(common[row].mean()),
                    frame_penalty=float(debug['candidate_frame_penalties'][index]))
    best = profile['best_candidate_index']
    out = dict(best_evidence=summary(best))
    second = debug.get('second_candidate_index')
    if second is not None and second in ids:
        out['legacy_second_evidence'] = summary(second)
        out['legacy_second_same_basin'] = bool(
            profile['basin_labels'][np.searchsorted(profile['profile_x'], debug['candidate_along_offsets'][second])]
            == profile['best_basin'])
    other = profile.get('competing_basin')
    if other:
        along = np.asarray(debug['candidate_along_offsets'])
        candidates = ids[along[ids] == other['along']]
        index = min(candidates, key=lambda i: debug['candidate_objective_scores'][i])
        out['competitor_evidence'] = summary(index)
    if not config.confidence_cell_stability:
        return out
    started = time.perf_counter()
    decisions = []
    along = np.asarray(debug['candidate_along_offsets'])
    for cell in np.unique(cells):
        selected = cells != cell
        if not selected.any():
            continue
        v = valid[:, selected]
        n = int(selected.sum())
        k = min(n, max(1, int(config.keep_best_count) if config.keep_best_count is not None
                       else int(round(n*config.keep_best_ratio))))
        required = max(k, int(np.ceil(n*config.masked_min_group_fraction))) if config.use_masked_sift else n
        good = v.sum(axis=1) >= required
        if config.use_masked_sift:
            deficient = sum((v[:, cells[selected] == c].sum(axis=1) < config.masked_min_points_per_cell)
                            for c in np.unique(cells[selected]))
            good &= deficient <= config.masked_max_deficient_cells
        scores, _ = _aggregate(costs[:, selected], v, cells[selected], config)
        scores = np.where(good, scores, np.inf)
        if not np.isfinite(scores).any():
            decisions.append(dict(cell=int(cell), stable=False, reason='insufficient_remaining_evidence'))
            continue
        row = min(np.flatnonzero(np.isfinite(scores)), key=lambda j: (scores[j], abs(along[ids[j]]), j))
        index = int(ids[row])
        basin_id = int(profile['basin_labels'][np.searchsorted(profile['profile_x'], along[index])])
        decisions.append(dict(cell=int(cell), stable=basin_id == profile['best_basin'],
                              index=index, along=float(along[index]), basin=basin_id))
    out.update(cell_trials=decisions,
               cell_stability=float(np.mean([d['stable'] for d in decisions])) if decisions else float('nan'),
               cell_stability_ms=(time.perf_counter()-started)*1000.)
    return out


def finalize_confidence(result, config, *, coarse_only=False):
    """Run once per search. Shadow records a proposal; active can only veto.

    Hard geometry/coverage/quality failures are never rescued here. Active
    bypasses the old ratio in the matcher and routes by basins in search.
    """
    started = time.perf_counter()
    debug = result.get('region_debug')
    if config.confidence_mode == 'legacy':
        return
    legacy_status = 'accepted' if result.get('m_pt') is not None else 'rejected'
    if debug is None:
        result['candidate_status'] = 'unscorable'
        result['confidence'] = dict(mode=config.confidence_mode, status='unscorable',
                                    reason_codes=['no_scoreable_candidates'], legacy_status=legacy_status,
                                    elapsed_ms=(time.perf_counter()-started)*1000.)
        return
    report = analyze_profile(debug, config)
    if report['status'] == 'unscorable':
        report.update(mode=config.confidence_mode, legacy_status=legacy_status,
                      elapsed_ms=(time.perf_counter()-started)*1000.)
        result['confidence'] = debug['confidence'] = report
        result['candidate_status'] = 'unscorable'
        if config.confidence_mode == 'active':
            result.update(m_pt=None, method='', reject_reason='匹配失敗: confidence: no_valid_candidates')
        return
    report.update(analyze_evidence(debug, config, report))
    reasons = report['reason_codes']
    if coarse_only:
        reasons.append('coarse_only_not_refined')
    if report['localization_width_px'] > config.confidence_max_width_px:
        reasons.append('localization_too_wide')
    lo, hi = report['interval_profile_indices']
    x = report['profile_x']
    start, end = max(0, lo-1), min(len(x)-1, hi+1)
    expected_step = config.fine_step_px if config.two_stage_search else config.search_along_step_px
    if np.any(np.diff(x[start:end+1]) > expected_step+1e-5):
        reasons.append('localization_undersampled')
    if config.confidence_cell_stability and report.get('cell_stability', 0.) < config.confidence_min_cell_stability:
        reasons.append('cell_unstable')
    # Hard legacy quality checks apply to both proposals; ratio/valley are the
    # rules being compared, not evidence that the new proposal is impossible.
    hard = result.get('reject_reason')
    if hard and not any(tag in hard for tag in ('ObjRatio', 'valley unclear', 'localization unclear')):
        reasons.append('upstream_rejection')
    report.update(mode=config.confidence_mode, legacy_status=legacy_status,
                  upstream_reason=hard, status='scored_uncertain' if reasons else 'accepted',
                  elapsed_ms=(time.perf_counter()-started)*1000., geometry_model='rigid candidates')
    result['confidence'] = debug['confidence'] = report
    result['scored_candidate'] = np.asarray(debug['best_center_right']).copy()
    result['candidate_status'] = report['status']
    if config.confidence_mode == 'active' and reasons:
        reason = hard if 'upstream_rejection' in reasons else '匹配失敗: confidence: '+', '.join(reasons)
        result.update(m_pt=None, method='', reject_reason=reason)


def attach_height_span(result, triangulate, normal, center, *, offset=0., label='', max_depth=np.inf):
    """Use the app's exact triangulation callback, before fusion/smoothing.

    Height gating applies only to a bounded sampled rigid solution. A LocalPlane
    result has different geometry and is explicitly not covered by this span.
    """
    report = result.get('confidence')
    debug = result.get('region_debug')
    if report is None or debug is None:
        return
    started = time.perf_counter()
    reason = None
    heights = []
    if normal is None or center is None or not np.all(np.isfinite(normal)) or not np.all(np.isfinite(center)):
        reason = 'height_plane_unavailable'
    elif result.get('refinement', {}).get('applied'):
        reason = 'local_plane_result_not_covered'
    else:
        for index in report.get('plausible_candidate_indices', []):
            p = np.append(debug['candidate_centers_warp'][index], 1.)
            q = np.asarray(debug['H_lr']) @ p
            try:
                if not np.all(np.isfinite(q)) or abs(q[2]) < 1e-12:
                    raise ValueError('invalid homography')
                xyz = triangulate(q[:2]/q[2])
                if xyz is None or not np.all(np.isfinite(xyz)) or not 0 < xyz[2] <= max_depth:
                    raise ValueError('invalid triangulation')
                value = float(np.dot(normal, xyz-center))-offset
                if not np.isfinite(value):
                    raise ValueError('invalid plane height')
                heights.append(value)
            except (ValueError, FloatingPointError):
                reason = 'invalid_candidate_geometry'
    report.update(height_plane_label=label, height_count=len(heights), height_reason=reason,
                  height_range_mm=(min(heights), max(heights)) if heights else None,
                  height_span_mm=max(heights)-min(heights) if heights else None,
                  height_elapsed_ms=(time.perf_counter()-started)*1000.)
    cfg = debug['config']
    if cfg.confidence_height_span_max_mm is not None:
        failed = reason is not None or not heights or report['height_span_mm'] > cfg.confidence_height_span_max_mm
        if failed:
            report['reason_codes'].append(reason or 'height_span_too_wide')
            report['status'] = result['candidate_status'] = 'scored_uncertain'
            if cfg.confidence_mode == 'active':
                result.update(m_pt=None, method='', reject_reason='匹配失敗: confidence: '+', '.join(report['reason_codes']))


def format_confidence(report):
    if not report:
        return []
    lines = [f"[Region-SIFT Confidence] mode={report['mode']} | proposal={report['status']} | "
             f"input={report['legacy_status']} | reasons={','.join(report['reason_codes']) or 'none'} | "
             f"{report['elapsed_ms']:.1f}ms"]
    if report.get('upstream_reason'):
        lines.append(f"  input rejection: {report['upstream_reason']}")
    for key, label in (('best_evidence', 'best'), ('legacy_second_evidence', 'legacy second'),
                       ('competitor_evidence', 'other basin')):
        if key in report:
            d = report[key]
            lines.append(f"  {label}: texture contribution={d['texture_contribution']:.3f} + missing={d['missing_contribution']:.3f}; "
                         f"frame={d['frame_penalty']:.3f}, valid={d['valid_pairs']}/{d['total_pairs']}, common={d['common_coverage']:.3f}")
    if 'legacy_second_same_basin' in report:
        lines.append(f"  legacy second belongs to same basin: {report['legacy_second_same_basin']}")
    if 'localization_width_px' in report:
        lines.append(f"  same-basin span={report['localization_width_px']:.2f}px | bounded={report.get('bounded')} | "
                     f"other-basin ratio={report.get('basin_ratio', float('nan')):.4f} | "
                     f"cell stability={report.get('cell_stability', float('nan')):.3f} ({report.get('cell_stability_ms', 0):.2f}ms)")
    if 'height_range_mm' in report:
        lines.append(f"  sampled height range={report['height_range_mm']} mm | {report['height_plane_label']} | "
                     f"reason={report['height_reason']} | {report['height_elapsed_ms']:.2f}ms; no temporal/multiframe fusion, not a confidence interval")
    return lines
