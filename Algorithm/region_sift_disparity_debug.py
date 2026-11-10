"""Opt-in, read-only per-anchor disparity diagnostic. No matching decisions change.

Uses the selected round's anchors, descriptors, masks and geometry. Local frames
are re-estimated just like the matcher (not a frozen-frame photometric test).
The table retains valid points even when their whole group fails coverage.
"""
from dataclasses import replace
import time

import cv2
import numpy as np

# Display/evidence heuristics only; never acceptance thresholds for the matcher.
FLAT_RELATIVE_RANGE = 0.02
NEAR_MIN_RELATIVE_BAND = 0.03
MIN_CURVE_VALID_FRACTION = 0.75


def summarize_table(offsets, costs):
    """costs is offsets x anchors; invalid observations remain NaN."""
    offsets, costs = np.asarray(offsets, float), np.asarray(costs, float)
    if costs.ndim != 2 or costs.shape[0] != len(offsets):
        raise ValueError('Table must be offsets x anchors')
    finite = np.isfinite(costs)
    delta = np.full_like(costs, np.nan)
    preferred = np.full(costs.shape[1], np.nan)
    reliable = np.zeros(costs.shape[1], bool)
    boundary = np.zeros(costs.shape[1], bool)
    reasons = []
    for j in range(costs.shape[1]):
        ids = np.flatnonzero(finite[:, j])
        if not len(ids):
            reasons.append('no valid samples')
            continue
        values = costs[ids, j]
        minimum = values.min()
        delta[ids, j] = values-minimum  # No row-wise contrast stretching.
        ties = ids[np.isclose(values, minimum, rtol=1e-7, atol=1e-8)]
        best = ties[np.argmin(np.abs(offsets[ties]))]
        preferred[j] = offsets[best]
        boundary[j] = best in (0, len(offsets)-1)
        near = finite[:, j] & (costs[:, j] <= minimum + NEAR_MIN_RELATIVE_BAND*max(abs(minimum), 1e-8))
        near_ids = np.flatnonzero(near)
        components = 1 + np.count_nonzero(np.diff(near_ids) > 1)
        if not np.all(finite[:, j]):
            reason = 'incomplete support'
        elif np.ptp(values) <= FLAT_RELATIVE_RANGE*max(abs(np.median(values)), 1e-8):
            reason = 'flat'
        elif components > 1:
            reason = 'multiple minima'
        elif offsets[near_ids[-1]]-offsets[near_ids[0]] >= .75*np.ptp(offsets):
            reason = 'broad minimum'
        elif boundary[j]:
            reason = 'boundary minimum'
        else:
            reason = 'distinct interior minimum'
            reliable[j] = True
        reasons.append(reason)
    return dict(delta=delta, preferred=preferred, reliable=reliable,
                boundary=boundary, reasons=reasons)


def subset_curve(costs, subset):
    subset = np.asarray(subset, bool)
    values = np.asarray(costs, float)[:, subset]
    counts = np.isfinite(values).sum(axis=1)
    required = max(1, int(np.ceil(subset.sum()*MIN_CURVE_VALID_FRACTION)))
    means = np.divide(np.nansum(values, axis=1), counts,
                      out=np.full(len(values), np.nan), where=counts >= required)
    return means, counts, required


def collect_disparity_audit(result, point_p, *, enabled=False):
    """Return a detached report; disabled performs no array work or descriptors.

Does not mutate result, score arrays, configs, caches or accepted coordinates.
Both retained candidates are examined, even if the final match was rejected.
"""
    if not enabled:
        return None
    started = time.perf_counter()
    report = dict(status='unavailable', candidates=[], descriptor_rows=0,
                  reason='no scored candidate', elapsed_ms=0.)
    try:
        debug = result.get('region_debug')
        if not debug:
            return report
        from Algorithm import Region_SIFT_Matching as matcher
        from Algorithm import masked_sift_descriptor as masked
        from Algorithm.region_sift_search import frame_context
        from Algorithm.region_sift_refinement import validate_refinement_config
        cfg = debug['config']
        validate_refinement_config(cfg)
        p = np.asarray(point_p, np.float32)
        anchors = np.asarray(debug['left_points'], np.float32)
        n = len(anchors)
        radius, step = cfg.refine_radius_px, cfg.refine_sample_step_px
        offsets = np.linspace(-radius, radius, int(round(2*radius/step))+1)
        tangent = np.asarray(debug['tangent'], np.float32)
        ranked = [('Top 1', int(debug['best_candidate_index']))]
        second = debug.get('second_candidate_index')
        if second is not None:
            ranked.append(('Top 2', int(second)))
        centers = np.asarray([debug['candidate_centers_warp'][i] for _, i in ranked])
        positions = (centers[:, None, None, :] + (anchors-p)[None, None, :, :]
                     + offsets[None, :, None, None]*tangent)
        positions = positions.reshape(-1, n, 2).astype(np.float32)
        image = debug['warped_right_gray']
        warp_valid = np.asarray(debug['warped_valid_mask'])
        spec = debug.get('warped_specular_mask')
        h, w = image.shape
        roi = None
        if cfg.use_masked_sift and cfg.frame_coordinate_quantization_px == 0:
            margin = masked.max_support_radius(cfg) + int(cfg.masked_extra_margin_px) + 2
            lo = np.floor(positions.min(axis=(0, 1))-margin).astype(int)
            hi = np.ceil(positions.max(axis=(0, 1))+margin).astype(int)+1
            roi = (max(0, lo[0]), max(0, lo[1]), min(w, hi[0]), min(h, hi[1]))
        context = frame_context(image, cfg, None, 'right', spec, warp_valid, roi=roi)
        origin = np.asarray(context.get('origin', (0, 0)), np.float32)
        ox, oy = origin.astype(int)
        ch, cw = context['image_shape']
        frame_valid_mask = warp_valid.copy()
        full_mask_check = bool(debug.get('reject_specular') and cfg.specular_check_support and not cfg.use_masked_sift)
        if full_mask_check and spec is not None:
            frame_valid_mask[spec > 0] = 0
        frame_cfg = replace(cfg, min_valid_warp_ratio=1.) if full_mask_check else cfg
        flat = positions.reshape(-1, 2)
        frames = matcher.estimate_dense_sift_frames(image[oy:oy+ch, ox:ox+cw], flat-origin,
            frame_cfg, context=context, valid_mask=frame_valid_mask[oy:oy+ch, ox:ox+cw])
        frame_ok = frames['valid'].copy()
        xy = np.rint(flat).astype(int)
        inside = (flat[:, 0] >= 0) & (flat[:, 0] <= w-1) & (flat[:, 1] >= 0) & (flat[:, 1] <= h-1)
        safe = np.column_stack((np.clip(xy[:, 0], 0, w-1), np.clip(xy[:, 1], 0, h-1)))
        frame_ok &= inside & (warp_valid[safe[:, 1], safe[:, 0]] > 0)
        if debug.get('reject_specular') and not cfg.use_masked_sift and spec is not None:
            frame_ok &= spec[safe[:, 1], safe[:, 0]] == 0
        along = np.concatenate([debug['candidate_along_offsets'][i]+offsets for _, i in ranked])
        band_ok = np.abs(along) <= (cfg.search_length_px-1)/2 + 1e-6
        frame_ok &= np.repeat(band_ok, n)
        ids = np.flatnonzero(frame_ok)
        descriptors = np.zeros((len(flat), 128), np.float32)
        batch_size = max(1, int(cfg.descriptor_batch_size))
        if cfg.use_masked_sift:
            packet = dict(descriptors=descriptors, histograms=np.zeros_like(descriptors),
                cell_valid_fraction=np.zeros((len(flat), 16), np.float32),
                valid_fraction=np.zeros(len(flat), np.float32), valid_points=np.zeros(len(flat), bool))
        sift = None if cfg.use_masked_sift else cv2.SIFT_create(nOctaveLayers=cfg.sift_n_octave_layers, sigma=cfg.sift_sigma)
        for start in range(0, len(ids), batch_size):
            ix = ids[start:start+batch_size]
            if cfg.use_masked_sift:
                part = masked.compute_descriptors(context, flat[ix]-origin,
                    frames['size_px'][ix], frames['angle_deg'][ix], cfg)
                for key in packet:
                    packet[key][ix] = part[key]
            else:
                descriptors[ix] = matcher.compute_descriptors_at_points(image, flat[ix], sift, cfg,
                    sizes_px=frames['size_px'][ix], angles_deg=frames['angle_deg'][ix], octaves=frames['octave'][ix])
            report['descriptor_rows'] += len(ix)
        right_frames = {key: value.reshape(-1, n) for key, value in frames.items()}
        if cfg.use_masked_sift:
            packet = {key: value.reshape((-1, n)+value.shape[1:]) for key, value in packet.items()}
            distances, pair_ok, _ = masked.compare_descriptors(debug['left_masked_packet'], packet, cfg)
            pair_ok &= frame_ok.reshape(-1, n)
            group_ok = masked.group_validity(pair_ok, debug['point_metadata'], cfg)
            right_frames['masked_point_valid'] = pair_ok
            for key in ('scale_reliable', 'orientation_reliable'):
                right_frames[key] &= pair_ok
        else:
            distances = np.linalg.norm(descriptors.reshape(-1, n, 128)-debug['left_descriptors'][None], axis=2)
            pair_ok = frame_ok.reshape(-1, n)
            group_ok = np.all(pair_ok, axis=1)
        components = matcher.score_candidate_groups(distances, debug['point_metadata'],
            debug['left_frames'], right_frames, cfg)
        across = np.repeat([debug['candidate_across_offsets'][i] for _, i in ranked], len(offsets))
        objective = components['objective_without_epi'] + cfg.epipolar_penalty_weight*np.abs(across)
        objective = np.where(group_ok, objective, np.nan)
        costs = np.where(pair_ok, distances, np.nan)
        center_mask = np.array([m['cell_row'] == cfg.grid_rows//2 and m['cell_col'] == cfg.grid_cols//2
                               for m in debug['point_metadata']])
        report.update(status='ok', reason='', offsets=offsets, anchors=anchors.copy(), point=p.copy(),
            metadata=debug['point_metadata'], center_mask=center_mask, left_roi=debug['left_roi'],
            click_index=int(np.argmin(np.linalg.norm(anchors-p, axis=1))), config=cfg,
            accepted=result.get('m_pt') is not None, rejection=result.get('reject_reason'),
            reference='rigid candidates before any LocalPlane refinement')
        for k, (label, index) in enumerate(ranked):
            sl = slice(k*len(offsets), (k+1)*len(offsets))
            curve = costs[sl].copy()
            report['candidates'].append(dict(label=label, index=index, costs=curve,
                objective=objective[sl].copy(), valid_counts=pair_ok[sl].sum(axis=1),
                along=float(debug['candidate_along_offsets'][index]),
                across=float(debug['candidate_across_offsets'][index]),
                summary=summarize_table(offsets, curve)))
        return report
    except (ValueError, KeyError, TypeError, cv2.error) as exc:
        report.update(status='unavailable', reason=str(exc))
        return report
    finally:
        report['elapsed_ms'] = (time.perf_counter()-started)*1000


class DisparityAuditViewer:
    """UI-thread-only viewer; rank switching never recomputes descriptors."""
    def __init__(self):
        self.figure = None
        self.button = None
        self.rank = 0

    def close(self):
        import matplotlib.pyplot as plt
        if self.button is not None:
            self.button.disconnect_events()
            self.button = None
        figure, self.figure = self.figure, None
        if figure is not None:
            plt.close(figure)

    def show(self, result, *, enabled=False):
        if not enabled:
            if self.figure is not None:
                self.close()
            return
        self.report = (result.get('region_debug') or {}).get('disparity_audit') or dict(
            status='unavailable', reason='No audit stored. Enable the switch and measure again.')
        self.image = result.get('debug_left_gray')
        # CenterRefine is measured relative to rigid Top 1 only. Reuse its table;
        # never recompute descriptors just to draw these additional anchors.
        self.center_report = (result.get('region_debug') or {}).get('center_refinement')
        self.rank = 0
        self._draw()

    def _switch(self, event):
        self.rank = (self.rank+1) % len(self.report['candidates'])
        self._draw()

    def _draw(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from matplotlib.patches import Rectangle
        from matplotlib.colors import Normalize
        if self.button is not None:
            self.button.disconnect_events()
            self.button = None
        if self.figure is None or not plt.fignum_exists(self.figure.number):
            self.figure = plt.figure(figsize=(16, 9))
            self.figure.canvas.manager.set_window_title('Region-SIFT anchor disparity consistency')
        fig = self.figure
        fig.clear()
        report = self.report
        if report['status'] != 'ok':
            fig.text(.08, .5, 'Disparity audit unavailable\n'+report.get('reason', ''), fontsize=12)
            fig.canvas.draw_idle()
            return
        data = report['candidates'][self.rank]
        x, points, p = report['offsets'], report['anchors'], report['point']
        summary = data['summary']
        grid = fig.add_gridspec(2, 3, left=.04, right=.97, bottom=.16, top=.84,
                               width_ratios=(1.05, 1.1, 1.), wspace=.38, hspace=.28)
        spatial, heat = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[:, 1])
        group, subsets = fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])
        radius = max(abs(x[0]), abs(x[-1]))
        center = getattr(self, 'center_report', None)
        center_available = bool(center and 'point_costs' in center and 'points' in center)
        show_center = center_available and self.rank == 0
        if center_available:
            radius = max(radius, float(np.max(np.abs(center['offsets']))))
        norm = Normalize(-radius, radius)
        cmap = plt.get_cmap('bwr')
        if self.image is not None:
            spatial.imshow(self.image, cmap='gray', vmin=0, vmax=255)
        span = np.maximum(np.ptp(points, axis=0), [20, 20])/2+15
        if show_center:
            span = np.maximum(span, np.max(np.abs(center['points']-p), axis=0)+15)
        spatial.set_xlim(p[0]-span[0], p[0]+span[0]); spatial.set_ylim(p[1]+span[1], p[1]-span[1])
        for i, pt in enumerate(points):
            value = summary['preferred'][i]
            if not np.isfinite(value):
                spatial.plot(*pt, marker='x', color='gray', ms=7)
            else:
                color = cmap(norm(value))
                spatial.scatter(*pt, s=55, facecolors=[color] if summary['reliable'][i] else 'none',
                                edgecolors=[color], linewidths=1.5)
            spatial.annotate(str(i), pt, xytext=(3, 3), textcoords='offset points', fontsize=6)
        if show_center:
            center_summary = summarize_table(center['offsets'], center['point_costs'])
            for i, pt in enumerate(center['points']):
                value = center_summary['preferred'][i]
                valid = np.isfinite(value)
                color = cmap(norm(value)) if valid else 'gray'
                spatial.scatter(*pt, marker='D', s=40,
                    facecolors=[color] if valid and center_summary['reliable'][i] else 'none',
                    edgecolors=[color], linewidths=1.2, zorder=4)
                if not valid:
                    spatial.plot(*pt, marker='x', color='gray', ms=7, zorder=5)
                spatial.annotate(f'C{i}', pt, xytext=(-5, -11), ha='right',
                                 textcoords='offset points', fontsize=7, color='yellow', zorder=6)
            rx, ry, rw, rh = center['roi']
            spatial.add_patch(Rectangle((rx, ry), rw, rh, fill=False,
                                       edgecolor='cyan', linestyle=':', linewidth=1.5))
        spatial.plot(*p, marker='*', color='gold', markeredgecolor='black',
                     ms=9 if show_center else 14, zorder=5)
        cfg = report['config']
        # Use the actual sampling ROI (not an approximate P-centered cell).
        x0, y0 = report['left_roi'][:2]
        cx = x0+(cfg.grid_cols//2)*cfg.cell_width_px
        cy = y0+(cfg.grid_rows//2)*cfg.cell_height_px
        spatial.add_patch(Rectangle((cx, cy), cfg.cell_width_px, cfg.cell_height_px,
                                   fill=False, edgecolor='lime', linestyle='--'))
        center_note = ('\nDiamonds C*: center-refine (Top 1)\nCyan box: center sampling ROI' if show_center else
                       '\nCenter-refine anchors: Top 1 only' if center_available else
                       '\nCenter-refine point evidence unavailable' if center else '')
        spatial.set_title('Left anchors: preferred correction\nfilled=distinct; hollow=uncertain; gray=invalid'
                          + center_note, fontsize=10)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=spatial, shrink=.65,
                     label='Along correction (px), NOT height/error')
        distance = np.linalg.norm(points-p, axis=1)
        order = np.argsort(distance, kind='stable')
        # Fixed heat scale across candidates; never independently normalize rows.
        deltas = [c['summary']['delta'] for c in report['candidates']]
        finite_delta = np.concatenate([a[np.isfinite(a)] for a in deltas])
        vmax = max(float(np.max(finite_delta)) if len(finite_delta) else 0., 1e-6)
        hcmap = plt.get_cmap('viridis').copy(); hcmap.set_bad('#aaaaaa')
        heat.imshow(np.ma.masked_invalid(summary['delta'][:, order].T), aspect='auto',
                    interpolation='nearest', cmap=hcmap, vmin=0, vmax=vmax)
        labels = []
        for row, i in enumerate(order):
            meta = report['metadata'][i]
            labels.append(f'{"P " if i == report["click_index"] else ""}{i} c{meta["cell_row"]},{meta["cell_col"]} {distance[i]:.0f}px')
            if np.isfinite(summary['preferred'][i]):
                col = int(np.argmin(abs(x-summary['preferred'][i])))
                heat.plot(col, row, marker='o', ms=4, markeredgecolor='red',
                          markerfacecolor='red' if summary['reliable'][i] else 'none')
                if summary['boundary'][i]:
                    heat.annotate('<' if col == 0 else '>', (col, row), color='red', fontsize=9)
        heat.set_yticks(np.arange(len(order)), labels, fontsize=max(4, min(7, 180/len(order))))
        heat.set_xticks(np.arange(len(x)), [f'{v:+g}' for v in x], fontsize=7)
        heat.set_xlabel('Along correction (px)')
        heat.set_title('Per-anchor distance minus own minimum\ndark=low; gray=invalid; <> = boundary', fontsize=10)
        fig.colorbar(heat.images[0], ax=heat, shrink=.65, label='Delta distance (shared scale)')
        group.plot(x, data['objective'], 'ko-')
        group.set_title('Whole-group Objective (original definition)', fontsize=10)
        group.set_ylabel('Objective')
        for off, score, count in zip(x, data['objective'], data['valid_counts']):
            if np.isfinite(score):
                group.annotate(f'{count}/{len(points)}', (off, score), xytext=(0, 6),
                               textcoords='offset points', fontsize=7, ha='center')
        for mask, color, label in ((report['center_mask'], 'tab:red', 'Center cell'),
                                   (~report['center_mask'], 'tab:blue', 'Outer cells')):
            mean, counts, required = subset_curve(data['costs'], mask)
            subsets.plot(x, mean, 'o-', color=color, label=f'{label}: need {required}/{mask.sum()}')
            for off, value, count in zip(x, mean, counts):
                if np.isfinite(value):
                    subsets.annotate(str(count), (off, value), xytext=(0, 6 if color == 'tab:red' else -12),
                                     textcoords='offset points', fontsize=7, color=color)
        subsets.legend(fontsize=7)
        subsets.set_ylabel('Mean valid-row distance')
        subsets.set_title('Center vs outer (numbers = valid points)', fontsize=10)
        for ax in (group, subsets):
            ax.axvline(0, color='gray', ls='--'); ax.grid(alpha=.2)
            ax.set_xticks(x); ax.set_xlabel('Along correction from this candidate (px)')
        status = 'accepted match' if report['accepted'] else 'REJECTED match: candidate only'
        fig.suptitle(f'Anchor disparity consistency | {data["label"]} | {status}\n'
                     f'base along={data["along"]:+.2f}, across={data["across"]:+.2f} px; '
                     f'{report["descriptor_rows"]} extra rows / {report["elapsed_ms"]:.1f} ms', fontsize=12)
        fig.text(.04, .055, 'Read-only; rigid candidates before LocalPlane. Top 2 is distance-excluded, NOT guaranteed a separate basin.\n'
                 'Frames are re-estimated: disagreement can reflect masks/texture/scale/angle, not only surface height.\n'
                 'Curve gaps = insufficient evidence. Compare minima positions, not absolute heights of differently defined scores.', fontsize=9)
        if len(report['candidates']) > 1:
            axbutton = fig.add_axes([.79, .9, .17, .035])
            self.button = Button(axbutton, 'Switch Top 1 / Top 2', useblit=False)
            self.button.on_clicked(self._switch)
        fig.canvas.draw_idle()
        fig.show()
