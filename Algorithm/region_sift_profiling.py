"""Read-only nested timing. Context-local state never enters feature/cache keys."""
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_active = ContextVar('region_sift_detail_profile', default=None)


def begin(profile):
    return _active.set((profile, 'left'))


def end(token):
    _active.reset(token)


def _bucket(category):
    active = _active.get()
    if active is None:
        return None
    profile, side = active
    return profile.setdefault(side + '.' + category, dict(total_ms=0., calls=0, ms={}, counts={}))


def on_side(side):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            active = _active.get()
            if active is None:
                return function(*args, **kwargs)
            token = _active.set((active[0], side))
            try:
                return function(*args, **kwargs)
            finally:
                _active.reset(token)
        return wrapped
    return decorate


def timed(category):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            bucket = _bucket(category)
            if bucket is None:
                return function(*args, **kwargs)
            started = perf_counter()
            bucket['calls'] += 1
            try:
                return function(*args, **kwargs)
            finally:
                bucket['total_ms'] += (perf_counter()-started)*1000.
        return wrapped
    return decorate


class DetailTimer:
    def __init__(self, category):
        self.bucket = _bucket(category)
        self.last = perf_counter() if self.bucket is not None else 0.

    def mark(self, name):
        if self.bucket is not None:
            now = perf_counter()
            values = self.bucket['ms']
            values[name] = values.get(name, 0.) + (now-self.last)*1000.
            self.last = now

    def count(self, name, value):
        if self.bucket is not None:
            counts = self.bucket['counts']
            counts[name] = counts.get(name, 0) + int(value)


def merge_profile(target, source):
    for key, source_bucket in source.items():
        bucket = target.setdefault(key, dict(total_ms=0., calls=0, ms={}, counts={}))
        bucket['total_ms'] += source_bucket['total_ms']
        bucket['calls'] += source_bucket['calls']
        for field in ('ms', 'counts'):
            for name, value in source_bucket[field].items():
                bucket[field][name] = bucket[field].get(name, 0) + value


def format_profile(profile):
    if not profile:
        return ['  (本次未進入特徵計算，或全數使用快取)']
    lines = ['  下列是既有大項的內部分解；不可再加到總耗時。',
             '  angle 是 frames 的子項；unique 為各呼叫去重後加總，不是跨階段全域去重。']
    for side, label in (('left', '左圖'), ('right', '右圖')):
        for category in ('pyramid', 'frames', 'angle', 'descriptor'):
            bucket = profile.get(side+'.'+category)
            if bucket is None:
                continue
            total = bucket['total_ms']
            lines.append(f"  [{label} {category}] {total:.3f} ms | calls={bucket['calls']} | "
                         + ', '.join(f'{k}={v}' for k, v in bucket['counts'].items()))
            for name, value in bucket['ms'].items():
                lines.append(f"    {name}: {value:.3f} ms ({100*value/total if total else 0:.1f}%)")
            remainder = max(0., total-sum(bucket['ms'].values()))
            lines.append(f'    計時／函式進出及未分類: {remainder:.3f} ms')
            if category == 'angle' and bucket['calls']:
                lines.append(f"    平均每次 angle: {total/bucket['calls']:.4f} ms")
            if category == 'frames' and bucket['counts'].get('unique_points', 0):
                lines.append(f"    每 unique 點攤提（含準備／scale／angle）: "
                             f"{total/bucket['counts']['unique_points']:.4f} ms")
            if category == 'descriptor':
                if bucket['counts'].get('custom_rows', 0):
                    rows = bucket['counts']['custom_rows']
                    lines.append(f'    自製算子每 row 攤提: {total/rows:.4f} ms（含覆蓋統計）')
                    continue
                rows = bucket['counts'].get('rows', 0)
                if rows:
                    cv_ms = bucket['ms'].get('OpenCV SIFT.compute（含內部前置）', 0.)
                    lines.append(f'    OpenCV 每 row 攤提: {cv_ms/rows:.4f} ms（非固定單點成本）')
                    if bucket['calls']:
                        lines.append(f"    OpenCV 每 batch 攤提: {cv_ms/bucket['calls']:.3f} ms")
                lines.append('    注意：Python 端無法把 OpenCV 內部金字塔與 descriptor 核心分開計時。')
    return lines
