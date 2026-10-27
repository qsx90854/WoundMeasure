# -*- coding: utf-8 -*-
"""Region-SIFT 精簡版深度量測（對外 demo 用）。

只保留 Region-SIFT 量測：選影片 -> 分析 Session -> 點擊左圖 -> 右圖匹配點／距離／高度。
量測演算法全在 regionsift_lite_core.py（與 MAIN 逐位元相同）；本檔只負責 UI。

啟動：
    python depth_measure_regionsift_lite.py [影片檔]      # 未給影片則跳出選檔視窗
    python depth_measure_regionsift_lite.py 影片 --no-record   # 不寫量測紀錄 txt
自測（無 GUI，Agg 後端，存截圖與紀錄到指定資料夾）：
    python depth_measure_regionsift_lite.py --selftest 輸出資料夾 [--video 影片]

操作：左圖點擊量測｜滾輪縮放｜左鍵拖曳平移｜右鍵或 R 鍵重設視圖。
"""
import argparse
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import cv2


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video_pos', nargs='?', default=None, help='影片檔（可省略，改由選檔視窗選）')
    ap.add_argument('--video', default=None, help='影片檔（同位置參數）')
    ap.add_argument('--no-record', action='store_true', help='不寫量測紀錄 txt')
    ap.add_argument('--selftest', metavar='OUT_DIR', default=None,
                    help='自測：Agg 後端建 Session、模擬點擊並存截圖／紀錄到 OUT_DIR')
    ap.add_argument('--click', type=int, nargs=2, default=None, metavar=('U', 'V'), help='自測用點擊座標')
    return ap.parse_args(argv)


_ARGS = _parse_args() if __name__ == '__main__' else None

import matplotlib
matplotlib.use("Agg" if (_ARGS is not None and _ARGS.selftest) else "TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

import regionsift_lite_core as core

BG = '#1E1E1E'
FG = '#E0E0E0'
BTN = '#2D2D2D'
BTN_HOVER = '#444444'
ACCENT = '#00BFFF'


class LiteApp:
    def __init__(self, record_enabled=True, record_path_override=None):
        self.config = core.REGION_SIFT_CONFIG
        self.record_enabled = record_enabled
        self.record_path_override = record_path_override
        self.session = None
        self.video_path = None
        self.busy = False            # 分析或量測進行中
        self.queue = queue.Queue()   # worker -> 主執行緒
        self.pan = {'pressing': False, 'dragged': False, 'x': None, 'y': None, 'ax': None}
        self.last_res = None

        self.fig = plt.figure(figsize=(14, 8.2), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("Region-SIFT Lite")
            self.fig.canvas.toolbar.pack_forget()
        except Exception:
            pass
        self.ax_A = self.fig.add_axes([0.02, 0.33, 0.47, 0.52], facecolor='#101010')
        self.ax_B = self.fig.add_axes([0.51, 0.33, 0.47, 0.52], facecolor='#101010')
        for ax in (self.ax_A, self.ax_B):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('#555555')
        self.ax_A.set_title('左圖（點擊量測）', color=FG, fontsize=11)
        self.ax_B.set_title('右圖（匹配結果）', color=FG, fontsize=11)
        self.im_A = self.im_B = None
        self.scatter_A, = self.ax_A.plot([], [], marker='+', color='#FF3030', markersize=16, mew=2, ls='none', zorder=5)
        self.scatter_B, = self.ax_B.plot([], [], marker='o', mfc='none', mec='#39FF14', markersize=14, mew=2,
                                         ls='none', zorder=5)
        self.epi_line, = self.ax_B.plot([], [], color='#FFD400', lw=1.2, alpha=0.9, zorder=4)

        # 結果文字（HUD）與資訊列
        self.hud = self.fig.text(0.02, 0.975, '', color='white', fontsize=15, fontweight='bold',
                                 va='top', ha='left', linespacing=1.4,
                                 bbox=dict(facecolor='#000000', alpha=0.55, edgecolor='none', pad=6))
        self.info = self.fig.text(0.02, 0.30, '', color=FG, fontsize=9.5, va='top', ha='left', linespacing=1.5)
        self.hint = self.fig.text(0.98, 0.012, '左圖點擊=量測｜滾輪=縮放｜拖曳=平移｜右鍵/R=重設視圖',
                                  color='#888888', fontsize=8.5, va='bottom', ha='right')

        # 進度條（分析時顯示）
        self.ax_prog = self.fig.add_axes([0.02, 0.115, 0.96, 0.022], facecolor='#303030')
        self.ax_prog.set_xlim(0, 100)
        self.ax_prog.set_ylim(0, 1)
        self.ax_prog.set_xticks([])
        self.ax_prog.set_yticks([])
        self.prog_bar = Rectangle((0, 0), 0, 1, color='#28A745')
        self.ax_prog.add_patch(self.prog_bar)
        self.prog_text = self.ax_prog.text(50, 0.5, '', color='white', fontsize=8.5, ha='center', va='center')
        self.ax_prog.set_visible(False)

        # 按鈕列
        def make_button(x, w, label):
            ax = self.fig.add_axes([x, 0.025, w, 0.06])
            btn = Button(ax, label, color=BTN, hovercolor=BTN_HOVER)
            btn.label.set_color(FG)
            btn.label.set_fontsize(10)
            ax.patch.set_edgecolor(ACCENT)
            return btn
        self.btn_open = make_button(0.02, 0.13, '開啟影片')
        self.btn_desc = make_button(0.16, 0.20, self._desc_label())
        self.btn_params = make_button(0.37, 0.13, 'SIFT 參數')
        self.btn_record = make_button(0.51, 0.15, self._record_label())
        self.btn_home = make_button(0.67, 0.12, '重設視圖')
        self.btn_open.on_clicked(lambda e: self.on_open())
        self.btn_desc.on_clicked(lambda e: self.on_toggle_descriptor())
        self.btn_params.on_clicked(lambda e: self.on_params())
        self.btn_record.on_clicked(lambda e: self.on_toggle_record())
        self.btn_home.on_clicked(lambda e: self.reset_view())

        c = self.fig.canvas
        c.mpl_connect('button_press_event', self.on_press)
        c.mpl_connect('button_release_event', self.on_release)
        c.mpl_connect('motion_notify_event', self.on_motion)
        c.mpl_connect('scroll_event', self.on_scroll)
        c.mpl_connect('key_press_event', lambda e: self.reset_view() if e.key in ('r', 'R') else None)
        c.mpl_connect('close_event', lambda e: self._stop_timer())
        self.timer = None
        try:
            self.timer = c.new_timer(interval=80)
            self.timer.add_callback(self.poll)
            self.timer.start()
        except Exception:
            self.timer = None
        self.set_hud('請按「開啟影片」選擇影片')

    # ------------------------------------------------------------ 小工具
    def _stop_timer(self):
        if self.timer is not None:
            try:
                self.timer.stop()
            except Exception:
                pass

    def _desc_label(self):
        return 'Descriptor: 自製(遮罩式)' if self.config.use_masked_sift else 'Descriptor: OpenCV'

    def _record_label(self):
        return '量測紀錄: 開' if self.record_enabled else '量測紀錄: 關'

    def set_hud(self, text):
        self.hud.set_text(text)
        self.fig.canvas.draw_idle()

    def set_info(self, text):
        self.info.set_text(text)

    def set_progress(self, percent, text):
        self.ax_prog.set_visible(True)
        self.prog_bar.set_width(float(np.clip(percent, 0, 100)))
        self.prog_text.set_text(text)

    def hide_progress(self):
        self.ax_prog.set_visible(False)

    def flush(self):
        """強制重繪一次（用於即將進入長時間計算前，讓「計算中」提示先顯示）。"""
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    # ------------------------------------------------------------ 影片載入／Session
    def on_open(self):
        if self.busy:
            return
        import tkinter as tk
        from tkinter import filedialog
        parent = getattr(self.fig.canvas.manager, 'window', None)
        root = None
        if parent is None:
            root = tk.Tk()
            root.withdraw()
        path = filedialog.askopenfilename(
            parent=parent or root, title='選擇影片',
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")])
        if root is not None:
            root.destroy()
        if path:
            self.load_video(path)

    def load_video(self, path, sync=False, analysis_copy_dir=None):
        """分析影片並建立 Session。sync=True 在目前執行緒執行（自測用），否則背景執行緒＋進度條。"""
        if self.busy:
            return
        self.busy = True
        self.video_path = str(path)
        self.set_hud(f'分析影片中…\n{Path(path).name}')
        self.set_progress(0, 'Preparing...')
        self.flush()

        def progress_cb(percent, text):
            self.queue.put(('progress', percent, core.ui_progress_status_english(text)))

        def work():
            try:
                session = core.build_session(path, progress_callback=progress_cb,
                                             analysis_copy_dir=analysis_copy_dir)
                self.queue.put(('session', session))
            except Exception as exc:
                traceback.print_exc()
                self.queue.put(('error', f'{type(exc).__name__}: {exc}'))

        if sync:
            work()
            self.poll()
        else:
            threading.Thread(target=work, daemon=True).start()

    def apply_session(self, session):
        self.session = session
        self.last_res = None
        cand = session['cands'][0]
        left_rgb = session['left_bgr'][:, :, ::-1]
        h, w = left_rgb.shape[:2]
        for ax in (self.ax_A, self.ax_B):
            ax.cla()
            ax.set_xticks([])
            ax.set_yticks([])
        self.im_A = self.ax_A.imshow(left_rgb, interpolation='nearest')
        self.im_B = self.ax_B.imshow(cand['rgb'], interpolation='nearest')
        self.scatter_A, = self.ax_A.plot([], [], marker='+', color='#FF3030', markersize=16, mew=2, ls='none', zorder=5)
        self.scatter_B, = self.ax_B.plot([], [], marker='o', mfc='none', mec='#39FF14', markersize=14, mew=2,
                                         ls='none', zorder=5)
        self.epi_line, = self.ax_B.plot([], [], color='#FFD400', lw=1.2, alpha=0.9, zorder=4)
        self.ax_A.set_title(f"左圖 F{session['left_frame_idx']}（點擊量測）", color=FG, fontsize=11)
        self.ax_B.set_title(
            f"右圖 F{cand['right_frame_idx']}（最優配對，基線 {cand['baseline']:.1f} mm；另有 {len(session['cands']) - 1} 組融合）",
            color=FG, fontsize=11)
        self.reset_view(draw=False)
        self.hide_progress()
        diag = session.get('shared_plane_diag') or {}
        self.set_hud('請在左圖點擊要量測的位置')
        self.set_info(f"影片: {Path(session['video_path']).name}\n"
                      f"高度參考平面: {session['height_plane'][2]}"
                      + ('' if diag.get('available') else f"（Shared Pattern Plane 不可用：{diag.get('reason', '')}）"))
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ 非同步訊息輪詢（主執行緒）
    def poll(self):
        updated = False
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == 'progress':
                    self.set_progress(msg[1], msg[2])
                    updated = True
                elif kind == 'session':
                    self.busy = False
                    self.apply_session(msg[1])
                    updated = True
                elif kind == 'measured':
                    self.busy = False
                    self.apply_result(msg[1])
                    updated = True
                elif kind == 'error':
                    self.busy = False
                    self.hide_progress()
                    self.set_hud(f'失敗\n{msg[1]}')
                    updated = True
        except queue.Empty:
            pass
        if updated:
            self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ 量測
    def request_measure(self, u, v, sync=False):
        if self.session is None or self.busy:
            return
        self.busy = True
        cfg = self.config                      # 不可變快照
        session = self.session
        self.set_hud(f'計算中…  ({u}, {v})\n{"自製遮罩式" if cfg.use_masked_sift else "OpenCV"} SIFT')
        self.scatter_A.set_data([u], [v])
        self.scatter_B.set_data([], [])
        self.epi_line.set_data([], [])
        self.flush()

        def work():
            try:
                t0 = time.perf_counter()
                res = core.measure_click(session, u, v, cfg)
                res['elapsed_s'] = time.perf_counter() - t0
                self.queue.put(('measured', res))
            except Exception as exc:
                traceback.print_exc()
                self.queue.put(('error', f'{type(exc).__name__}: {exc}'))

        if sync:
            work()
            self.poll()
        else:
            threading.Thread(target=work, daemon=True).start()

    def apply_result(self, res):
        self.last_res = res
        sess = self.session
        cand = sess['cands'][0]
        u, v = res['u'], res['v']
        self.scatter_A.set_data([u], [v])
        pt = res.get('pt')
        if pt is not None:
            self.scatter_B.set_data([pt[0]], [pt[1]])
            w = sess['image_size'][0]
            p0, p1 = core.epipolar_line(cand['F'], (u, v), w)
            self.epi_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        else:
            self.scatter_B.set_data([], [])
            self.epi_line.set_data([], [])

        if res.get('d') is not None:
            lines = [f"Camera-to-Selected Position Distance: {res['depth']:.1f} mm"]
            if res.get('height_display_mm') is not None:
                lines.append(f"Wound Height ({res['height_reference_source']}): {res['height_display_mm']:.1f} mm")
            self.set_hud('\n'.join(lines))
        else:
            reason = core.ui_failure_reason_english(res.get('fail_reason') or 'No Valid Depth')
            self.set_hud('No Valid Depth' if reason == 'No Valid Depth' else f'Invalid point ({reason})')

        info = [f"點擊 (u,v)=({u}, {v})   計算 {res.get('elapsed_s', 0.0):.2f} s   "
                f"Descriptor: {'自製(遮罩式)' if self.config.use_masked_sift else 'OpenCV'}"]
        if res.get('d') is not None:
            pw = res.get('p3d_w')
            info.append(f"右圖匹配 ({pt[0]:.2f}, {pt[1]:.2f})   相機座標 [{res['p3d'][0]:.1f}, {res['p3d'][1]:.1f}, {res['p3d'][2]:.1f}] mm"
                        + ('' if pw is None else f"   世界座標 [{pw[0]:.1f}, {pw[1]:.1f}, {pw[2]:.1f}] mm"))
            info.append(f"重投影誤差 {res['error']:.3f} px   極線偏差 {res['d_epi']:.2f} px   "
                        f"ZNCC {res['zncc_score']:.3f}   信心 {res['confidence_score']:.3f}")
            n_ok = len(res.get('multi_res', []))
            info.append(f"多配對融合: 採用 {res.get('fused_count', 0)}/{n_ok} 組成功"
                        + (f"（剔除 F{res['fused_dropped']}）" if res.get('fused_dropped') else ''))
        else:
            info.append(f"失敗原因: {res.get('fail_reason') or '無匹配點'}")
        self.set_info('\n'.join(info))

        if self.record_enabled:
            core.save_measurement_to_txt(sess['video_path'], res, cand, txt_path=self.record_path_override)
        self.fig.canvas.draw_idle()

    def measure_sync(self, u, v):
        """自測用：同步量測並更新畫面。"""
        self.request_measure(int(u), int(v), sync=True)
        return self.last_res

    # ------------------------------------------------------------ 按鈕
    def on_toggle_descriptor(self):
        if self.busy:
            return
        self.apply_region_settings(replace(self.config, use_masked_sift=not self.config.use_masked_sift))

    def apply_region_settings(self, updated):
        if self.busy:
            raise ValueError('計算進行中，請稍後再修改參數')
        previous = self.config
        self.config = updated
        self.btn_desc.label.set_text(self._desc_label())
        for name, value in vars(updated).items():
            if getattr(previous, name) != value:
                print(f'[Region-SIFT Settings] {name}: {getattr(previous, name)} -> {value}')
        self.set_hud('Region-SIFT 參數已套用；請重新點選')
        self.fig.canvas.draw_idle()

    def on_params(self):
        if self.busy:
            return
        from Algorithm.region_sift_settings import show_region_sift_settings
        show_region_sift_settings(self.fig.canvas.manager.window, self.config, self.apply_region_settings)

    def on_toggle_record(self):
        self.record_enabled = not self.record_enabled
        self.btn_record.label.set_text(self._record_label())
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ 滑鼠：點擊／平移／縮放
    def reset_view(self, draw=True):
        if self.session is None:
            return
        h, w = self.session['left_bgr'].shape[:2]
        for ax in (self.ax_A, self.ax_B):
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
        if draw:
            self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes not in (self.ax_A, self.ax_B):
            return
        if event.button == 3:
            self.reset_view()
            return
        if event.button != 1:
            return
        self.pan.update({'pressing': True, 'dragged': False, 'x': event.x, 'y': event.y, 'ax': event.inaxes})

    def on_motion(self, event):
        if not self.pan['pressing'] or event.inaxes != self.pan['ax']:
            return
        dx, dy = event.x - self.pan['x'], event.y - self.pan['y']
        if not self.pan['dragged'] and abs(dx) < 3 and abs(dy) < 3:
            return
        self.pan['dragged'] = True
        ax = self.pan['ax']
        inv = ax.transData.inverted()
        p0, p1 = inv.transform((self.pan['x'], self.pan['y'])), inv.transform((event.x, event.y))
        dx_d, dy_d = p1 - p0
        ax.set_xlim(ax.get_xlim() - dx_d)
        ax.set_ylim(ax.get_ylim() - dy_d)
        self.pan.update({'x': event.x, 'y': event.y})
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if not self.pan['pressing']:
            return
        self.pan['pressing'] = False
        if self.pan['dragged'] or event.xdata is None or event.inaxes is not self.ax_A:
            return
        if self.pan['ax'] is not self.ax_A or self.session is None or self.busy:
            return
        ux, vx = float(event.xdata), float(event.ydata)
        # 自動吸附 ArUco 角點（同 MAIN）；座標一律 int(round())
        ux, vx = core.snap_to_aruco_corner(ux, vx, self.session['cands'][0]['cornersA'])
        u, v = int(round(ux)), int(round(vx))
        w, h = self.session['image_size']
        if not (0 <= u < w and 0 <= v < h):
            return
        self.request_measure(u, v)

    def on_scroll(self, event):
        if event.inaxes not in (self.ax_A, self.ax_B) or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        f = 1.2 if event.button == 'down' else 1 / 1.2
        xl, yl = ax.get_xlim(), ax.get_ylim()
        x, y = event.xdata, event.ydata
        ax.set_xlim([x - (x - xl[0]) * f, x + (xl[1] - x) * f])
        ax.set_ylim([y - (y - yl[0]) * f, y + (yl[1] - y) * f])
        self.fig.canvas.draw_idle()


# ====================================================================== 自測
def run_selftest(out_dir, video, click=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = out / f"{Path(video).stem}_lite_selftest_record.txt"
    if record.exists():
        record.unlink()
    app = LiteApp(record_enabled=True, record_path_override=str(record))
    app.fig.savefig(out / '00_startup.png', dpi=90, facecolor=BG)
    t0 = time.perf_counter()
    app.load_video(video, sync=True, analysis_copy_dir=str(out / '_analysis_copy'))
    assert app.session is not None, 'Session 建立失敗'
    print(f'[selftest] Session 完成 {time.perf_counter() - t0:.1f}s 左圖 F{app.session["left_frame_idx"]}')
    app.fig.savefig(out / '01_session.png', dpi=90, facecolor=BG)
    if click is None:
        pts = np.concatenate([np.asarray(c, np.float64).reshape(-1, 2)
                              for c in app.session['cands'][0]['cornersA'].values()])
        click = tuple(int(round(float(x))) for x in np.median(pts, axis=0))
    results = {}
    for tag, masked in (('opencv', False), ('masked', True)):
        app.config = replace(app.config, use_masked_sift=masked)
        app.btn_desc.label.set_text(app._desc_label())
        res = app.measure_sync(*click)
        app.fig.savefig(out / f'02_click_{tag}.png', dpi=90, facecolor=BG)
        results[tag] = res
        print(f"[selftest] {tag}: click={click} d={res.get('d')} height={res.get('height_display_mm')} "
              f"fail={res.get('fail_reason')!r} elapsed={res.get('elapsed_s', 0):.2f}s")
    print(f'[selftest] 紀錄檔: {record} ({record.stat().st_size if record.exists() else 0} bytes)')
    return results


def main():
    args = _ARGS
    video = args.video or args.video_pos
    if args.selftest:
        if not video:
            video = str(Path(__file__).resolve().parent / 'test_video_Zebra' / 'video_20260929_140621_560744_lossless.avi')
        run_selftest(args.selftest, video, None if args.click is None else tuple(args.click))
        return
    app = LiteApp(record_enabled=not args.no_record)
    if video:
        app.load_video(video)
    else:
        app.on_open()
    plt.show()


if __name__ == '__main__':
    main()
