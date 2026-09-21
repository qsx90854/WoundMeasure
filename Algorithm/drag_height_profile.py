"""Drag-sampled height profiles with timer-driven measurement and cached replay."""
from dataclasses import dataclass
import gzip
import pickle
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DragProfileConfig:
    spacing_px: float = 10.0
    max_points: int = 100
    pick_radius_px: float = 12.0  # Screen pixels, independent of zoom.


def sample_drag_path(path, config):
    """Uniform arclength samples in image pixels, rounded like normal clicks."""
    if not np.isfinite(config.spacing_px) or config.spacing_px <= 0 or config.max_points < 2:
        raise ValueError('Profile spacing must be positive and max_points >= 2')
    pts = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(pts) < 2 or not np.isfinite(pts).all():
        raise ValueError('Drag a path with at least two distinct points')
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]]
    if len(pts) < 2:
        raise ValueError('Drag a path with at least two distinct points')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if arc[-1] < 2:
        raise ValueError('Drag at least 2 image pixels')
    step = max(float(config.spacing_px), arc[-1] / (int(config.max_points)-1))
    distances = np.arange(0., arc[-1], step)
    distances = np.r_[distances, arc[-1]]
    if len(distances) > config.max_points:
        distances = np.linspace(0., arc[-1], config.max_points)
    xy = np.column_stack([np.interp(distances, arc, pts[:, axis]) for axis in (0, 1)])
    xy = np.rint(xy).astype(int)
    unique = np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)]
    return xy[unique], distances[unique], step


class ProfileCache:
    """Private, temporary snapshots; load only files created by this instance."""
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='region_sift_profile_')

    def save(self, index, result):
        keys = ('u', 'v', 'pt', 'pt_raw', 'p3d', 'p3d_best', 'height_display_mm',
                'height_reference_source', 'fail_reason', 'region_debug',
                'region_diagnostic_config', 'region_search_history',
                'debug_left_gray', 'debug_right_gray', 'cand_idx')
        snapshot = {key: result[key] for key in keys if key in result}
        with gzip.open(Path(self.directory.name) / f'{index}.pkl.gz', 'wb', compresslevel=1) as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, index):
        with gzip.open(Path(self.directory.name) / f'{index}.pkl.gz', 'rb') as stream:
            return pickle.load(stream)

    def close(self):
        self.directory.cleanup()


class DragHeightProfile:
    """All UI and measurement callbacks run on the main Tk/Matplotlib thread."""
    def __init__(self, figure, left_ax, right_ax, button, config, *,
                 activate, deactivate, measure, replay, notify, refresh, image_shape):
        self.figure, self.axes, self.button = figure, (left_ax, right_ax), button
        self.config = config
        self.activate, self.deactivate = activate, deactivate
        self.measure, self.replay = measure, replay
        self.notify, self.refresh, self.image_shape = notify, refresh, image_shape
        self.mode = 'off'
        self.path, self.samples, self.records = [], np.empty((0, 2)), []
        self.artists = []
        self.cache = None
        self.timer = None
        self.chart = None
        self.selected = None
        self._busy = False
        self._cancel_requested = False
        button.on_clicked(self.toggle)
        figure.canvas.mpl_connect('close_event', self._on_main_close)
        figure.canvas.mpl_connect('key_press_event', self._on_key)

    @property
    def enabled(self):
        return self.mode != 'off'

    def toggle(self, event=None):
        if self.mode == 'running':
            if self._busy:
                self._cancel_requested = True
            else:
                self._finish('cancelled')
            return
        if self.enabled:
            self.close()
            return
        if not self.activate():
            return
        self.mode = 'ready'
        self.button.label.set_text('拖曳剖面: On')
        self.notify('Drag profile: hold LEFT mouse on LEFT image, draw, then release.\n'
                    f'Spacing={self.config.spacing_px:g}px; max={self.config.max_points} points. '
                    'Press this button again or Esc to exit.')
        self.refresh()

    def _on_key(self, event):
        if event.key == 'escape' and self.enabled:
            self.toggle()

    def _on_main_close(self, event):
        if event.canvas is self.figure.canvas:
            self.close(redraw=False)

    def _clear_artists(self):
        for artist in self.artists:
            artist.remove()
        self.artists = []

    def _reset_run(self):
        if self.cache is not None:
            self.cache.close()
        self.cache = None
        self._clear_artists()
        if self.chart is not None:
            plt.close(self.chart)
        self.chart = None
        self.path, self.records = [], []
        self.samples = np.empty((0, 2))
        self.selected = None

    def close(self, redraw=True):
        if self.timer is not None:
            self.timer.stop()
        self.timer = None
        was_enabled = self.enabled
        self.mode = 'off'
        self._reset_run()
        if was_enabled:
            self.deactivate()
        self.button.label.set_text('拖曳高度剖面')
        if redraw:
            self.refresh()

    def _point(self, event):
        if event.inaxes is not self.axes[0] or event.xdata is None or event.ydata is None:
            return None
        height, width = self.image_shape()[:2]
        if not (0 <= event.xdata <= width-1 and 0 <= event.ydata <= height-1):
            return None
        return np.array([event.xdata, event.ydata], float)

    def _nearest(self, ax, event):
        choices = []
        for i, record in enumerate(self.records):
            point = self.samples[i] if ax is self.axes[0] else record['right']
            if point is not None:
                screen = ax.transData.transform(point)
                choices.append((float(np.linalg.norm(screen-[event.x, event.y])), i))
        if not choices:
            return None
        distance, index = min(choices)
        return index if distance <= self.config.pick_radius_px else None

    def on_press(self, event):
        if not self.enabled or event.inaxes not in self.axes or event.button != 1:
            return False
        if self.mode == 'running':
            return True
        if self.mode == 'review':
            picked = self._nearest(event.inaxes, event)
            if picked is not None:
                self.select(picked)
                return True
        if event.inaxes is self.axes[1]:
            return True
        point = self._point(event)
        if point is not None:
            self._reset_run()
            self.path = [point]
            self.mode = 'drawing'
            self.stroke, = self.axes[0].plot([point[0]], [point[1]], color='cyan', lw=1.5,
                                           animated=True, zorder=20)
            self.artists.append(self.stroke)
            self.refresh()
        return True

    def on_motion(self, event):
        if self.mode != 'drawing':
            return self.enabled and event.inaxes in self.axes
        point = self._point(event)
        if point is not None and np.linalg.norm(point-self.path[-1]) >= .5:
            self.path.append(point)
            points = np.asarray(self.path)
            self.stroke.set_data(points[:, 0], points[:, 1])
            self.refresh()
        return True

    def on_release(self, event):
        if self.mode != 'drawing':
            return self.enabled and event.inaxes in self.axes
        if event.button != 1:
            return True
        point = self._point(event)
        if point is not None:
            self.path.append(point)
        try:
            self.samples, self.distances, self.effective_spacing = sample_drag_path(self.path, self.config)
            self.cache = ProfileCache()
        except (ValueError, OSError) as exc:
            self.mode = 'ready'
            self.notify(f'Drag profile: {exc}')
            return True
        self.mode = 'running'
        self._cancel_requested = False
        self._overlay()
        self.notify(f'Drag profile: measuring {len(self.samples)} points; spacing ~{self.effective_spacing:.1f}px.\n'
                    'Cancel between points with the profile button or Esc.')
        self.timer = self.figure.canvas.new_timer(interval=100)
        self.timer.add_callback(self._tick)
        self.timer.start()
        return True

    def _tick(self):
        if self.mode != 'running' or self._busy:
            return
        self._busy = True
        index = len(self.records)
        u, v = map(int, self.samples[index])
        try:
            try:
                result = self.measure(u, v) or dict(fail_reason='No measurement result')
            except Exception as exc:
                result = dict(u=u, v=v, fail_reason=f'{type(exc).__name__}: {exc}')
            if self.mode != 'running':
                return
            result.setdefault('u', u)
            result.setdefault('v', v)
            value = result.get('height_display_mm')
            height = float(value) if value is not None and np.isfinite(value) else float('nan')
            right = result.get('pt')
            trusted = right is not None
            if right is None and isinstance(result.get('region_debug'), dict):
                right = result['region_debug'].get('best_center_right')
            if right is not None:
                right = np.asarray(right, float).reshape(2)
                if not np.isfinite(right).all():
                    right = None
            record = dict(height=height, right=right, trusted=trusted,
                          reason=result.get('fail_reason') or '', cached=False)
            try:
                self.cache.save(index, result)
                record['cached'] = True
            except (OSError, pickle.PickleError) as exc:
                record['reason'] += f'; diagnostic cache failed: {exc}'
            self.records.append(record)
            self._overlay()
            self.button.label.set_text(f'取消剖面 {index+1}/{len(self.samples)}')
            self.notify(f'Drag profile {index+1}/{len(self.samples)} | P{index+1}: '
                        f'Height={height:.3f} mm\n{record["reason"]}')
        finally:
            self._busy = False
        if self._cancel_requested or len(self.records) == len(self.samples):
            self._finish('cancelled' if self._cancel_requested else 'completed')

    def _finish(self, status):
        if self.timer is not None:
            self.timer.stop()
        self.timer = None
        self.mode = 'review'
        self.button.label.set_text('剖面檢視 / 關閉')
        self.notify(f'Profile {status}: {len(self.records)}/{len(self.samples)} measured.\n'
                    'Click a numbered LEFT/RIGHT point or chart point for diagnostics. '
                    'Right x? = rejected candidate, not a valid match.\n'
                    'Drag elsewhere on LEFT to replace profile; button/Esc exits.')
        self._show_chart()
        self.refresh()

    def _overlay(self):
        self._clear_artists()
        if len(self.samples) == 0:
            return
        path = np.asarray(self.path)
        line, = self.axes[0].plot(path[:, 0], path[:, 1], color='cyan', alpha=.6,
                                  lw=1, zorder=20, animated=True)
        self.artists.append(line)
        for i, xy in enumerate(self.samples):
            record = self.records[i] if i < len(self.records) else None
            valid = record is not None and np.isfinite(record['height'])
            color = 'lime' if valid else ('red' if record else '#aaaaaa')
            label = f'{i+1}: {record["height"]:.2f}' if valid else f'{i+1}: ' + ('FAIL' if record else '...')
            self._mark(self.axes[0], xy, label, color, 'o', i == self.selected)
            if record and record['right'] is not None:
                self._mark(self.axes[1], record['right'], str(i+1) + ('' if record['trusted'] else '?'),
                           color if record['trusted'] else 'magenta',
                           'o' if record['trusted'] else 'x', i == self.selected)
        self.refresh()

    def _mark(self, ax, point, label, color, marker, selected):
        dot, = ax.plot(*point, marker=marker, color=color, ms=7 if selected else 4,
                       linestyle='none', zorder=25, animated=True)
        text = ax.annotate(label, point, xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='yellow' if selected else color,
                           zorder=26, animated=True)
        self.artists.extend([dot, text])

    def _show_chart(self):
        if self.chart is None or not plt.fignum_exists(self.chart.number):
            self.chart = plt.figure(figsize=(10, 4.8))
            self.chart.canvas.manager.set_window_title('Wound height profile')
            self.chart.canvas.mpl_connect('button_press_event', self._chart_pick)
            self.chart.canvas.mpl_connect('close_event', self._chart_close)
        self.chart.clear()
        self.chart_ax = self.chart.add_subplot(111)
        ids = np.arange(1, len(self.records)+1)
        values = np.array([r['height'] for r in self.records])
        self.chart_ax.plot(ids, values, 'o-', color='#2585bd', ms=4)
        invalid = ids[~np.isfinite(values)]
        self.chart_ax.plot(invalid, np.full(len(invalid), .04), 'rx',
                           transform=self.chart_ax.get_xaxis_transform(), label='Failed (not zero height)')
        if self.selected is not None and self.selected < len(values) and np.isfinite(values[self.selected]):
            self.chart_ax.plot(self.selected+1, values[self.selected], 'o', color='orange', ms=9)
        self.chart_ax.set(xlabel='Point number (drag order)', ylabel='Wound Height (mm)',
                          title=f'{len(values)} measured / {len(self.samples)} sampled | '
                                f'valid={np.isfinite(values).sum()} | spacing ~{self.effective_spacing:.1f} image px')
        self.chart_ax.set_xlim(.5, max(1.5, len(values)+.5))
        if len(ids) <= 30:
            self.chart_ax.set_xticks(ids)
        self.chart_ax.grid(alpha=.25)
        self.chart_ax.legend(loc='best', fontsize=8)
        self.chart.text(.08, .02, 'Click a point number to inspect its stored Region-SIFT diagnostics. '
                        'Gaps preserve failures; no height interpolation.', fontsize=8)
        self.chart.tight_layout(rect=(0, .06, 1, 1))
        self.chart.show()
        self.chart.canvas.draw_idle()

    def _chart_close(self, event):
        if self.chart is not None and event.canvas is self.chart.canvas:
            self.chart = None

    def _chart_pick(self, event):
        if event.inaxes is self.chart_ax and event.xdata is not None and event.button == 1:
            index = int(round(event.xdata))-1
            if 0 <= index < len(self.records):
                self.select(index)

    def select(self, index):
        if not 0 <= index < len(self.records):
            return
        self.selected = index
        self._overlay()
        self._show_chart()
        record = self.records[index]
        self.notify(f'Profile P{index+1}: ({self.samples[index, 0]}, {self.samples[index, 1]}) | '
                    f'Height={record["height"]:.3f} mm\n{record["reason"]}')
        if record['cached']:
            try:
                self.replay(self.cache.load(index))
            except Exception as exc:
                self.notify(f'Profile P{index+1} diagnostic unavailable: {exc}')

    def draw_overlays(self):
        for artist in self.artists:
            if artist.axes is not None and artist.axes.get_visible():
                artist.axes.draw_artist(artist)
