"""Session-only editor for adaptive spatial specular-mask parameters."""
from dataclasses import fields, replace
import math
from typing import get_type_hints

from Algorithm.specular_detection import (
    BlockSpatialSpecularConfig,
    validate_block_spatial_specular_config,
)


LABELS = {
    'block_width_px': '動態門檻區塊寬度 px；較小更局部，但對雜訊更敏感',
    'block_height_px': '動態門檻區塊高度 px',
    'v_percentile': 'HSV V 動態門檻百分位；降低會選到更多亮區',
    'rgb_percentile': 'RGB 最大值動態門檻百分位；降低會選到更多亮區',
    'local_hot_percentile': '局部亮度差動態門檻百分位；降低會更寬鬆',
    'v_min': 'V 動態門檻下限',
    'v_max': 'V 動態門檻上限',
    'rgb_min': 'RGB 動態門檻下限',
    'rgb_max': 'RGB 動態門檻上限',
    'local_hot_min': '局部亮度差門檻下限；降低會抓更多弱反光',
    'local_hot_max': '局部亮度差門檻上限',
    'gray_min': '局部亮點分支的最低灰階亮度',
    'gray_below_v': '局部亮點可比 V 門檻低多少；提高會更寬鬆',
    's_max': '低飽和亮區的最大 HSV S；提高可接受更多偏色亮區',
    'whiteness_max': 'RGB 最大/最小差上限；提高可接受更多偏色亮區',
    'background_sigma': '估計局部背景的 Gaussian sigma',
    'enable_prominence_gate': '要求亮點相對局部背景突出，抑制大片黃白區誤判',
    'prominence_min': '局部突出門檻下限；降低會更寬鬆',
    'prominence_mad_multiplier': '紋理自適應 MAD 倍數；降低會更寬鬆',
    'enable_strong_highlight_exception': '保留接近飽和且接近白色的強反光例外',
    'strong_v_min': '強反光例外的最低 V',
    'strong_s_max': '強反光例外的最大 S',
    'strong_whiteness_max': '強反光例外的 RGB 色差上限',
    'interpolate_thresholds': '在區塊中心之間平滑內插門檻，減少方格邊界',
    'open_kernel_px': '形態學 opening 核大小，必須為正奇數',
    'close_kernel_px': '形態學 closing 核大小，必須為正奇數',
    'dilate_px': '最終遮罩向外擴張半徑 px；0 不擴張',
}


def format_value(value):
    return str(value)


def _parse_value(text, annotation):
    text = str(text).strip()
    if annotation is bool:
        if text.lower() not in ('true', 'false'):
            raise ValueError('請填 True 或 False')
        return text.lower() == 'true'
    value = annotation(text)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('必須是有限數字')
    return value


def parse_config_values(current, entries):
    """Parse edits atomically and return a validated immutable config."""
    annotations = get_type_hints(BlockSpatialSpecularConfig)
    changes = {}
    for name, text in entries.items():
        if name not in annotations:
            raise ValueError(f'未知參數: {name}')
        try:
            changes[name] = _parse_value(text, annotations[name])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name}: {exc}') from exc
    config = replace(current, **changes)
    validate_block_spatial_specular_config(config)
    return config


def show_specular_settings(parent, current, on_apply):
    """Open a modal adaptive-spatial-mask editor; changes are session-only."""
    import tkinter as tk
    from tkinter import ttk, messagebox

    window = tk.Toplevel(parent)
    window.title('舊版反光參數（本次執行）')
    window.geometry('1050x700')
    window.minsize(820, 450)
    window.transient(parent)
    ttk.Label(
        window,
        text=('修改後按「套用」：立即重算目前左右圖的空間反光遮罩。'
              '套用時切回舊版、開啟 Adaptive Spatial 與 Show Spatial；重新啟動後恢復程式設定。'),
        padding=10,
    ).pack(anchor='w')
    ttk.Label(
        window,
        text=('這裡控制「哪些像素判定為反光」；自製 SIFT 的有效覆蓋與群組門檻仍在「SIFT 參數」。'
              '有有效傷口 AI 遮罩時會走傷口自適應偵測，這組全圖區塊參數不參與。'),
        padding=6,
    ).pack(anchor='w')
    body = ttk.Frame(window)
    body.pack(fill='both', expand=True, padx=10)
    canvas = tk.Canvas(body, highlightthickness=0)
    scroll = ttk.Scrollbar(body, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scroll.set)
    scroll.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)
    form = ttk.Frame(canvas)
    item = canvas.create_window((0, 0), window=form, anchor='nw')
    form.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>', lambda event: canvas.itemconfigure(item, width=event.width))
    window.bind('<MouseWheel>', lambda event: canvas.yview_scroll(-int(event.delta / 120), 'units'))
    variables = {}
    form.columnconfigure(1, weight=1)
    for row, field in enumerate(fields(current)):
        name, value = field.name, getattr(current, field.name)
        ttk.Label(form, text=name).grid(row=row, column=0, sticky='w', padx=6, pady=4)
        variable = tk.StringVar(window, value=format_value(value))
        variables[name] = variable
        if isinstance(value, bool):
            editor = ttk.Combobox(
                form, textvariable=variable, values=('True', 'False'), state='readonly')
        else:
            editor = ttk.Entry(form, textvariable=variable)
        editor.grid(row=row, column=1, sticky='ew', padx=6, pady=4)
        ttk.Label(form, text=LABELS.get(name, '')).grid(
            row=row, column=2, sticky='w', padx=6)

    def apply():
        try:
            updated = parse_config_values(
                current, {name: variable.get() for name, variable in variables.items()})
            on_apply(updated)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror('反光參數未套用', str(exc), parent=window)
            return
        window.destroy()

    def restore():
        for name, variable in variables.items():
            variable.set(format_value(getattr(current, name)))

    footer = ttk.Frame(window, padding=10)
    footer.pack(fill='x')
    ttk.Button(footer, text='還原開啟時的數值', command=restore).pack(side='left')
    ttk.Button(footer, text='取消', command=window.destroy).pack(side='right', padx=4)
    ttk.Button(footer, text='套用並重算', command=apply).pack(side='right', padx=4)
    window.bind('<Escape>', lambda event: window.destroy())
    window.grab_set()
    window.focus_set()
    return window
