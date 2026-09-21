"""Independent session-only parameter editor for Specular Model V2."""
from dataclasses import fields, replace
from typing import get_type_hints

from Algorithm.specular_model_v2 import SpecularModelV2Config, validate_specular_model_v2_config


V2_LABELS = {
    'v2_input_gamma': '輸入轉近似線性 RGB 的指數；已線性輸入填 1',
    'v2_light_rgb': '光源 RGB 顏色比例，逗號分隔；白光 1,1,1',
    'v2_reference_radii_px': '同類組織參考取樣半徑，递增整數，逗號分隔',
    'v2_reference_hue_sigma': '扣除光源方向後的色調容許差；越小越避免混入異色組織',
    'v2_reference_chroma_power': '參考點色度權重指數；提高偏重較少白光污染的參考',
    'v2_min_reference_affinity': '最低同類組織色調相容度；提高更避免借用異色組織',
    'v2_min_reference_samples': '最低有效參考樣本數；提高較保守',
    'v2_reference_dispersion': '參考顏色分散容許值；降低較保守',
    'v2_min_separation': '組織色與光源色最低可分辨程度；提高較保守',
    'v2_noise_floor': '線性 RGB 的絕對噪聲尺度；提高較保守',
    'v2_noise_relative': '隨亮度增加的相對噪聲尺度；提高較保守',
    'v2_model_penalty': '增加反光成分的模型複雜度懲罰；提高較保守',
    'v2_min_specular_fraction': '反光成分比例達此值才得到完整比例信心',
    'v2_seed_score': '高信心核心門檻 0..1；提高遮罩減少（非機率）',
    'v2_grow_score': '核心向外延伸時的最低證據分數；不可大於核心門檻',
    'v2_uncertain_score': '顯示為不確定區域的最低證據分數',
    'v2_grow_distance_px': '沿合格像素向外延伸最多幾步；0 只取核心',
    'v2_clip_level': '原始影像通道接近飽和的門檻；此類列為不確定',
    'v2_clip_channels': '至少幾個通道接近飽和才列為資訊受損 1..3',
    'v2_show_uncertain': 'Show Spatial 時橘色顯示不確定區；不加入 SIFT 遮罩',
}


def format_v2_value(value):
    return ', '.join(map(str, value)) if isinstance(value, tuple) else str(value)


def parse_v2_values(current, entries):
    annotations = get_type_hints(SpecularModelV2Config)
    changes = {}
    for name, raw in entries.items():
        if name not in annotations:
            raise ValueError(f'未知 V2 參數: {name}')
        text = str(raw).strip()
        try:
            if name == 'v2_light_rgb':
                value = tuple(float(x.strip()) for x in text.strip('()[]').split(','))
            elif name == 'v2_reference_radii_px':
                value = tuple(int(x.strip()) for x in text.strip('()[]').split(','))
            elif annotations[name] is bool:
                if text.lower() not in ('true', 'false'):
                    raise ValueError('請填 True 或 False')
                value = text.lower() == 'true'
            else:
                value = annotations[name](text)
            changes[name] = value
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name}: {exc}') from exc
    return validate_specular_model_v2_config(replace(current, **changes))


def show_specular_model_v2_settings(parent, current, on_apply):
    import tkinter as tk
    from tkinter import ttk, messagebox

    window = tk.Toplevel(parent)
    window.title('新版反光 V2 參數（本次執行）')
    window.geometry('1100x700')
    window.minsize(880, 450)
    window.transient(parent)
    ttk.Label(window, padding=10, wraplength=1000,
              text='套用會啟用新版 V2 並重算左右圖。所有 v2_ 參數與舊版獨立；重新啟動恢復程式設定。'
                   '藍色為排除區，橘色為不確定且不排除。證據分數並非校準機率。').pack(anchor='w')
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
    window.bind('<MouseWheel>', lambda event: canvas.yview_scroll(-int(event.delta/120), 'units'))
    variables = {}
    form.columnconfigure(1, weight=1)
    for row, field in enumerate(fields(current)):
        name = field.name
        value = getattr(current, name)
        variable = tk.StringVar(window, value=format_v2_value(value))
        variables[name] = variable
        ttk.Label(form, text=name).grid(row=row, column=0, sticky='w', padx=5, pady=4)
        editor = (ttk.Combobox(form, textvariable=variable, values=('True', 'False'), state='readonly')
                  if isinstance(value, bool) else ttk.Entry(form, textvariable=variable))
        editor.grid(row=row, column=1, sticky='ew', padx=5)
        ttk.Label(form, text=V2_LABELS[name], wraplength=530).grid(row=row, column=2, sticky='w', padx=5)

    def apply():
        try:
            updated = parse_v2_values(current, {name: var.get() for name, var in variables.items()})
            on_apply(updated)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror('新版反光參數未套用', str(exc), parent=window)
            return
        window.destroy()

    def restore():
        for name, var in variables.items():
            var.set(format_v2_value(getattr(current, name)))

    footer = ttk.Frame(window, padding=10)
    footer.pack(fill='x')
    ttk.Button(footer, text='還原開啟時數值', command=restore).pack(side='left')
    ttk.Button(footer, text='取消', command=window.destroy).pack(side='right')
    ttk.Button(footer, text='套用並啟用新版', command=apply).pack(side='right', padx=8)
    window.bind('<Escape>', lambda event: window.destroy())
    window.grab_set()
    window.focus_set()
    return window
