"""GUI callback entry to completed main-canvas blit; no OS presentation claims."""


class ClickLatency:
    def __init__(self, enabled=True, print_fn=print):
        self.enabled = enabled
        self.print = print_fn
        self.pending = None
        self.sequence = 0

    def begin(self, point, started, release_started=None):
        if not self.enabled:
            self.pending = None
            return None
        if self.pending is not None:
            self.print(f"[Click-to-render #{self.pending['id']}] superseded before main render; no display latency reported")
        self.sequence += 1
        self.pending = dict(id=self.sequence, point=tuple(point), started=started,
                            origin=started if release_started is None else release_started,
                            source='量測入口' if release_started is None else '滑鼠放開 callback 入口',
                            ready=None)
        return self.pending

    def ready(self, ticket, finished):
        if ticket is not None and ticket is self.pending:
            ticket['ready'] = finished

    def complete(self, ticket, render_started, prepared, finished, *, main_drawn):
        # Ignore partial debug-only draws, reentrant/old frames and measurements
        # which have not finished updating their artists yet.
        if (ticket is None or ticket is not self.pending or not main_drawn
                or ticket['ready'] is None or ticket['ready'] > render_started):
            return None
        times = [ticket['origin'], ticket['started'], ticket['ready'],
                 render_started, prepared, finished]
        if any(b < a for a, b in zip(times, times[1:])):
            return None
        self.pending = None
        stages = list(zip((
            ticket['source'] + ' → 開始量測',
            '量測計算＋UI 資料更新',
            '量測結束 → render 開始（含 log 輸出／排程等待）',
            '顯示影像準備',
            '實際 draw＋blit'),
            [(b-a)*1000 for a,b in zip(times,times[1:])]))
        total = (finished-ticket['origin'])*1000
        self.print(f"⏱️ [Click-to-render #{ticket['id']}] P={ticket['point']} | "
                   f"{ticket['source']} → 主畫布 blit 完成: {total:.1f} ms")
        for name, ms in stages:
            self.print(f'   - {name}: {ms:.1f} ms')
        self.print('   不含 OS 事件送達前等待、螢幕合成／刷新；不是實際螢幕亮起時間。')
        return dict(total_ms=total, stages=stages)
