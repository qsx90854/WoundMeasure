"""分階段耗時量測工具：在終端機顯示各步驟耗時與占比（純量測，不影響計算結果）。"""
import time


class StageTimer:
    """
    用法：
        t = StageTimer("啟動流程")
        ... 步驟 1 ...
        t.stage("步驟 1 名稱")
        ... 步驟 2 ...
        t.stage("步驟 2 名稱")
        t.report()

    stage() 記錄「距上一次 stage()（或建立時）」的耗時；report() 印出總表與占比。
    """

    def __init__(self, title):
        self.title = title
        self.t0 = time.perf_counter()
        self.t_last = self.t0
        self.stages = []

    def stage(self, name):
        now = time.perf_counter()
        self.stages.append((name, now - self.t_last))
        self.t_last = now

    def total(self):
        return time.perf_counter() - self.t0

    def report(self, print_fn=print):
        total = self.total()
        print_fn(f"⏱️ [{self.title}] 總耗時 {total:.2f} s")
        for name, dt in self.stages:
            pct = (dt / total * 100.0) if total > 1e-9 else 0.0
            print_fn(f"   - {name:<26s}{dt * 1000.0:9.1f} ms ({pct:5.1f}%)")
