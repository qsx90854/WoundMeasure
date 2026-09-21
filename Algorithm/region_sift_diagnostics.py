"""Read-only Region-SIFT ambiguity viewer; no matching or GUI backend selection."""

import textwrap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import CheckButtons, RadioButtons, TextBox, Button
from Algorithm.region_sift_gt import evaluate_gt


def score_grid(debug, key):
    """Preserve invalid candidates as NaN, including irregular/singleton axes."""
    along = np.asarray(debug['candidate_along_offsets'])
    across = np.asarray(debug['candidate_across_offsets'])
    x, y = np.unique(along), np.unique(across)
    grid = np.full((len(y), len(x)), np.nan)
    scores = np.asarray(debug[key], dtype=float)
    grid[np.searchsorted(y, across), np.searchsorted(x, along)] = np.where(
        np.isfinite(scores), scores, np.nan)
    return x, y, grid


class RegionSIFTDiagnostics:
    """One reusable figure. Click an anchor in any crop to inspect all frames."""

    def __init__(self):
        self.figure = None
        self.search_figure = None
        self.options = [False, False, False, False]
        self.score_key = 'candidate_objective_scores'
        self.gt = None
        self.gt_text = ''
        self.gt_error = ''

    def _disconnect_widgets(self):
        """Detach callbacks before clearing axes; TextBox.stop_typing draws."""
        widgets = [getattr(self, name, None)
                   for name in ('checks', 'radio', 'gt_box', 'gt_button')]
        for widget in widgets:
            if widget is not None:
                widget.eventson = False
                widget.active = False
                widget.disconnect_events()
                if widget.canvas.mouse_grabber is widget.ax:
                    widget.canvas.release_mouse(widget.ax)
        box = getattr(self, 'gt_box', None)
        if box is not None and box.capturekeystrokes:
            box.stop_typing()
        for name in ('checks', 'radio', 'gt_box', 'gt_button'):
            setattr(self, name, None)
        self.crops = []
        self.point_sets = []

    def _on_close(self, event):
        if self.figure is not None and event.canvas is self.figure.canvas:
            self._disconnect_widgets()
            self.figure = None
            if self.search_figure is not None:
                plt.close(self.search_figure)
                self.search_figure = None

    def _ensure_window(self):
        if self.figure is None or not plt.fignum_exists(self.figure.number):
            self.figure = plt.figure(figsize=(15, 10))
            self.figure.canvas.manager.set_window_title('Region-SIFT ambiguity diagnostics')
            self.figure.canvas.mpl_connect('button_press_event', self._pick)
            self.figure.canvas.mpl_connect('close_event', self._on_close)

    def _present(self):
        self.figure.show()
        window = getattr(self.figure.canvas.manager, 'window', None)
        if window is not None and hasattr(window, 'deiconify'):
            window.deiconify()
            window.lift()
        self.figure.canvas.draw_idle()

    def show(self, result):
        self.result = result
        self.debug = result.get('region_debug')
        self._show_search_history(result)
        self.gt = None
        self.gt_error = ''
        self._ensure_window()
        if not self.debug or result.get('debug_left_gray') is None:
            self._disconnect_widgets()
            self.figure.clear()
            reason = str(result.get('fail_reason') or 'Region-SIFT unavailable')
            cfg = result.get('region_diagnostic_config')
            settings = '' if cfg is None else (
                f'\n\nSampling: {cfg.grid_rows}x{cfg.grid_cols} cells, '
                f'{cfg.cell_width_px}x{cfg.cell_height_px}px/cell, '
                f'{cfg.points_per_cell} anchors/cell + P\n'
                f'Search: {cfg.search_width_px}x{cfg.search_length_px}px; '
                f'support cap={cfg.descriptor_max_support_radius_px}px; '
                f'specular_check_support={cfg.specular_check_support}')
            self.figure.text(.05, .85, 'No candidate scores for this measurement.\n\n' +
                textwrap.fill(reason, width=110) + settings +
                '\n\nThe matcher stopped before scoring. Check the reason above; '
                'this is not an empty score plot.', va='top')
            self._present()
            return
        self.selected = len(self.debug['left_points']) - 1
        self._build()
        self._present()

    def _show_search_history(self, result):
        from Algorithm.region_sift_search_debug import draw_search_history
        history = result.get('region_search_history') or (self.debug or {}).get('search_history')
        if not history:
            if self.search_figure is not None:
                plt.close(self.search_figure)
                self.search_figure = None
            return
        if self.search_figure is None or not plt.fignum_exists(self.search_figure.number):
            self.search_figure = plt.figure(figsize=(12, 7))
            self.search_figure.canvas.manager.set_window_title('Region-SIFT coarse-to-fine search')
        draw_search_history(self.search_figure, history,
                            result.get('fail_reason') or (self.debug or {}).get('search_status', ''))
        self.search_figure.show()
        self.search_figure.canvas.draw_idle()

    def _build(self):
        fig = self.figure
        self._disconnect_widgets()
        fig.clear()
        layout = fig.add_gridspec(3, 12, left=.055, right=.94, bottom=.29,
                                 top=.88, hspace=.5, wspace=.7,
                                 height_ratios=[1, 1.3, .65])
        self.curve = fig.add_subplot(layout[0, :6])
        self.heat = fig.add_subplot(layout[0, 6:])
        self.crops = [fig.add_subplot(layout[1, i*3:(i+1)*3]) for i in range(4)]
        self.bars = fig.add_subplot(layout[2, :])
        self.color_axis = fig.add_axes([.925, .41, .012, .23])
        self.checks = CheckButtons(fig.add_axes([.055, .035, .16, .12]),
                                  ['IDs', 'L2 colors', 'KEEP / TRIM', 'Selected support'],
                                  self.options)
        self.checks.on_clicked(self._toggle)
        self.radio = RadioButtons(fig.add_axes([.23, .035, .13, .12]),
                                  ['Objective', 'GroupScore'],
                                  active=int(self.score_key == 'candidate_group_scores'))
        self.radio.on_clicked(self._score)
        self.gt_box = TextBox(fig.add_axes([.105, .205, .075, .032]), 'GT mm ', initial=self.gt_text)
        self.gt_button = Button(fig.add_axes([.19, .205, .10, .032]), 'Evaluate GT')
        self.gt_box.on_submit(self._submit_gt)
        self.gt_button.on_clicked(lambda event: self._submit_gt(self.gt_box.text))
        self.gt_status = fig.text(.31, .245, '', fontsize=8, va='top')
        self.detail = fig.text(.38, .11, '', fontsize=9, va='center')
        fig.text(.38, .035, 'Click an anchor in any crop. Filled=KEEP, hollow=TRIM. '
                 'TRIM still contributes to CellAll.\n'
                 'Right crops use warped coordinates; all crops share the same scale. '
                 'Gray heatmap cells are invalid or not evaluated.', fontsize=8)
        self._plots()
        self._images()

    def _submit_gt(self, text):
        self.gt_text = text.strip()
        self.gt = None
        self.gt_error = ''
        if self.gt_text:
            try:
                self.gt = evaluate_gt(self.debug, float(self.gt_text))
            except (ValueError, KeyError, TypeError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
                self.gt_error = str(exc)
        self._plots()
        self._images()
        self.figure.canvas.draw_idle()

    def _score(self, label):
        self.score_key = ('candidate_objective_scores' if label == 'Objective'
                          else 'candidate_group_scores')
        self._plots()
        self.figure.canvas.draw_idle()

    def _toggle(self, label):
        self.options = list(self.checks.get_status())
        self._images()
        self.figure.canvas.draw_idle()

    def _plots(self):
        d = self.debug
        x, y, grid = score_grid(d, self.score_key)
        self.curve.clear()
        self.heat.clear()
        for offset, row in zip(y, grid):
            self.curve.plot(x, row, alpha=.45, lw=1, label=f'across {offset:g}')
        envelope = np.min(np.where(np.isfinite(grid), grid, np.inf), axis=0)
        envelope[~np.isfinite(envelope)] = np.nan
        self.curve.plot(x, envelope, color='black', lw=2, label='min across')
        cmap = plt.get_cmap('YlGnBu_r').copy()
        cmap.set_bad('#999999')
        self.heat.pcolormesh(x, y, np.ma.masked_invalid(grid), cmap=cmap, shading='nearest')
        for key, marker, color, label in [('best_candidate_index', '*', 'red', 'Best'),
                                          ('second_candidate_index', '^', 'magenta', 'Second')]:
            index = d.get(key)
            if index is not None:
                a = d['candidate_along_offsets'][index]
                b = d['candidate_across_offsets'][index]
                value = d[self.score_key][index]
                self.curve.scatter(a, value, marker=marker, c=color, s=90, label=label, zorder=5)
                self.heat.scatter(a, b, marker=marker, c=color, s=90, zorder=5)
        a, b = d['best_along_offset_px'], d['best_across_offset_px']
        radius = d['second_best_exclusion_radius_px']
        self.curve.axvspan(a-radius, a+radius, alpha=.08, color='red')
        limits = self.heat.get_xlim(), self.heat.get_ylim()
        self.heat.add_patch(Circle((a, b), radius, fill=False, color='red', ls='--'))
        self.heat.set_xlim(limits[0]); self.heat.set_ylim(limits[1])
        gt = self.gt
        if gt is not None:
            self.heat.scatter(gt['along'], gt['across'], marker='D', c='lime',
                              edgecolors='black', s=70, zorder=6)
            # Include off-band GT without stretching the colored search samples.
            self.heat.set_xlim(min(limits[0][0], gt['along']-1), max(limits[0][1], gt['along']+1))
            self.heat.set_ylim(min(limits[1][0], gt['across']-1), max(limits[1][1], gt['across']+1))
            self.curve.axvline(gt['along'], color='green', ls=':', label='GT predicted along')
            if gt['scoreable']:
                key = 'objective_score' if self.score_key == 'candidate_objective_scores' else 'group_score'
                self.curve.scatter(gt['along'], gt[key], c='lime', edgecolors='black',
                                   marker='D', s=70, zorder=6, label='GT exact score')
            near = gt['nearest_index']
            gt_status = (f"GT + current geometry: {'INSIDE' if gt['inside_band'] else 'OUTSIDE'} search band; "
                         f"along/across=({gt['along']:.2f}, {gt['across']:.2f})px; "
                         f"R=({gt['center_right'][0]:.2f}, {gt['center_right'][1]:.2f})\n"
                         f"Nearest sample #{near}: distance={gt['nearest_distance_px']:.3f}px, "
                         f"Obj={gt['nearest_objective']:.3f}; exact GT: {gt['reason']}")
            if gt['scoreable']:
                gt_status += (f"\nGT G={gt['group_score']:.3f}, Obj={gt['objective_score']:.3f}, "
                              f"GT-BestObj={gt['objective_score']-d['objective_score']:+.3f} "
                              f"(negative=GT better); Trim/CellAll/Frame="
                              f"{gt['trimmed_score']:.2f}/{gt['balanced_score']:.2f}/{gt['frame_penalty']:.2f}")
            else:
                gt_status += '\nRejected anchor IDs: ' + str(gt.get('invalid_anchor_ids', []))
            self.gt_status.set_text(gt_status)
        else:
            self.gt_status.set_text('GT: ' + (self.gt_error or
                'Enter this block height, then Evaluate GT. Uses current RT/height plane; not independent pixel ground truth.'))
        metric = 'Objective' if self.score_key == 'candidate_objective_scores' else 'GroupScore'
        self.curve.set(xlabel='Along epipolar offset (px)', ylabel=metric,
                       title='Lower is better | shaded: +/- exclusion radius')
        self.curve.grid(alpha=.25)
        self.curve.legend(fontsize=7, ncol=3)
        self.heat.set(xlabel='Along epipolar offset (px)', ylabel='Across (px)',
                      title=f'{metric}: blue=low, yellow=high | dashed: exclusion circle')
        second = d.get('second_candidate_index')
        separation = (np.hypot(a-d['second_along_offset_px'], b-d['second_across_offset_px'])
                      if second is not None else float('nan'))
        heights = d.get('diagnostic_heights_mm', [float('nan'), float('nan')])
        status = textwrap.fill(
            f"Matcher: {d.get('diagnostic_match_status', 'unknown')} | "
            f"Measurement: {self.result.get('fail_reason') or 'no failure reported'}", width=155)
        self.figure.suptitle(
            f"BestObj={d['objective_score']:.2f}  SecondObj={d['second_objective_score']:.2f}  "
            f"ObjRatio={d['objective_score_ratio']:.4f}  separation={separation:.2f}px\n"
            f"Candidate H (single frame, mm): best={heights[0]:.3f}, second={heights[1]:.3f}, "
            f"difference={abs(heights[0]-heights[1]):.3f} | {d.get('diagnostic_plane_label', 'plane unavailable')}\n"
            + status, fontsize=10)

    def _images(self):
        d = self.debug
        second = d.get('second_candidate')
        gt = self.gt
        self.point_sets = [np.asarray(d['left_points']), np.asarray(d['best_points_warp']),
                           None if second is None else np.asarray(second['points_warp'])]
        frames = [d['left_frames'], d['right_frames'], None if second is None else second['frames']]
        distances = [None, np.asarray(d['distances']), None if second is None else second['distances']]
        keeps = [None, d['keep_mask'], None if second is None else second['keep_mask']]
        images = [self.result['debug_left_gray'], d['warped_right_gray'], d['warped_right_gray']]
        self.point_sets.append(None if gt is None else gt['points_warp'])
        frames.append(None if gt is None else gt['frames'])
        distances.append(None if gt is None else gt.get('distances'))
        keeps.append(None if gt is None else gt.get('keep_mask'))
        images.append(d['warped_right_gray'])
        max_distance = max(1., *(float(np.max(a)) for a in distances if a is not None))
        norm = plt.Normalize(0, max_distance)
        cmap = plt.get_cmap('RdYlGn_r')
        left = self.point_sets[0]
        half = max(float(np.ptp(left[:, 0])), float(np.ptp(left[:, 1]))) / 2 + 12
        if self.options[3]:
            half = max(half, *(float(f['support_radius_px'][self.selected]) +
                        float(np.max(np.abs(p[self.selected]-p[-1]))) + 5
                        for p, f in zip(self.point_sets, frames) if p is not None and f is not None))
        details = []
        for ax, pts, frame, values, keep, im, name in zip(
                self.crops, self.point_sets, frames, distances, keeps, images,
                ['Left reference', 'Right best (warped)', 'Right second (warped)', 'GT predicted (warped)']):
            ax.clear()
            ax.set_title(name, fontsize=10)
            if pts is None:
                ax.text(.5, .5, 'Enter GT height' if name.startswith('GT') else 'No distinct second',
                        ha='center', transform=ax.transAxes, fontsize=8)
                continue
            ax.imshow(im, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            center = pts[-1]
            ax.set_xlim(center[0]-half, center[0]+half)
            ax.set_ylim(center[1]+half, center[1]-half)
            colors = cmap(norm(values)) if values is not None and self.options[1] else np.tile([0., .7, 1., 1.], (len(pts), 1))
            faces = colors.copy()
            if keep is not None and self.options[2]:
                faces[~np.asarray(keep, bool), 3] = 0
            ax.scatter(pts[:, 0], pts[:, 1], s=22, facecolors=faces, edgecolors=colors)
            ax.scatter(*center, marker='+', c='red', s=90)
            if name.startswith('GT'):
                ax.scatter(*center, marker='D', facecolors='none', edgecolors='lime', s=110)
            ax.scatter(*pts[self.selected], s=85, facecolors='none', edgecolors='yellow')
            if self.options[0]:
                for i, (u, v) in enumerate(pts):
                    ax.annotate(str(i+1), (u, v), xytext=(3, 3), textcoords='offset points', color='cyan', fontsize=7)
            i = self.selected
            if self.options[3] and frame is not None:
                ax.add_patch(Circle(pts[i], float(frame['support_radius_px'][i]),
                                    fill=False, color='yellow', ls='--'))
            if frame is None:
                details.append(f"{name}: no valid frame; {gt['reason']}")
                continue
            details.append(f"{name.split(' (')[0]}: size={frame['size_px'][i]:.2f}, angle={frame['angle_deg'][i]:.1f}, "
                           f"support={frame['support_radius_px'][i]:.1f}px, "
                           f"reliable S/A={int(frame['scale_reliable'][i])}/{int(frame['orientation_reliable'][i])}"
                           + (f", L2={values[i]:.2f}, {'KEEP' if keep[i] else 'TRIM'}"
                              if values is not None else ''))
        self.color_axis.clear()
        self.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=self.color_axis, label='L2')
        self.detail.set_text(f'Anchor {self.selected+1} | support circle is a conservative bound\n' + '\n'.join(details))
        self.bars.clear()
        ids = np.arange(1, len(left)+1)
        self.bars.bar(ids-.18, distances[1], width=.36, label='Best', color='#2794b8')
        if distances[2] is not None:
            self.bars.bar(ids+.18, distances[2], width=.36, label='Second', color='#df8741')
        if distances[3] is not None:
            self.bars.plot(ids, distances[3], 'D-', color='green', markersize=3, label='GT exact')
        self.bars.set(xlabel='Anchor ID (last=P)',
                      ylabel='L2 + missing penalty' if d['config'].use_masked_sift else 'L2', xticks=ids)
        self.bars.legend(fontsize=8)
        self.bars.grid(axis='y', alpha=.2)

    def _pick(self, event):
        if event.inaxes not in getattr(self, 'crops', []) or event.xdata is None:
            return
        pts = self.point_sets[self.crops.index(event.inaxes)]
        if pts is None:
            return
        screen = event.inaxes.transData.transform(pts)
        distance = np.linalg.norm(screen-[event.x, event.y], axis=1)
        if distance.min() <= 15:
            self.selected = int(np.argmin(distance))
            self._images()
            self.figure.canvas.draw_idle()
