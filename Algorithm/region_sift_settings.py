"""Validated, session-only Region-SIFT parameter editor (Tk imported lazily)."""
from dataclasses import fields, replace
import math
from typing import get_args, get_origin, get_type_hints, Union

from Algorithm.Region_SIFT_Matching import RegionSIFTConfig, RegionSIFTError, _validate_config
from Algorithm.region_sift_frames import _scale_specs


LABELS = {
    'grid_rows': '取點區塊列數（奇數）',
    'grid_cols': '取點區塊欄數（奇數）',
    'cell_width_px': '每格寬度 px', 'cell_height_px': '每格高度 px',
    'points_per_cell': '每格取點數（另保留中心 P）',
    'sobel_ksize': '梯度核大小 1/3/5/7',
    'min_point_distance_px': '取點最小間距 px',
    'exclude_click_from_cell_points': '避免取點與 P 重複',
    'auto_scale_orientation': '自動估計尺度與角度',
    'scale_keypoint_sizes_px': '候選尺度（逗號分隔，px）',
    'sift_n_octave_layers': '每 octave 層數', 'sift_sigma': 'SIFT 基準 sigma',
    'descriptor_max_support_radius_px': '最大 support 半徑 px（None 不限制）',
    'flat_keypoint_size_px': '平坦區回退尺度 px',
    'scale_min_confidence': '尺度可靠性門檻',
    'orientation_min_confidence': '角度可靠性門檻',
    'scale_boundary_is_reliable': '邊界尺度視為可靠',
    'scale_dog_ratio': 'DoG 尺度比',
    'scale_response_pool_sigma_factor': '尺度響應平滑係數',
    'scale_response_floor': '尺度響應下限',
    'orientation_bins': '角度直方圖 bin 數',
    'orientation_sigma_factor': '角度 Gaussian sigma 係數',
    'orientation_radius_factor': '角度取樣半徑係數',
    'orientation_hist_smooth_passes': '角度直方圖平滑次數',
    'frame_coordinate_quantization_px': '座標量化 px（0 保持精確）',
    'keypoint_size_px': '固定／回退尺度 px',
    'keypoint_angle_deg': '固定／回退角度 deg',
    'require_all_descriptors': '要求全部 descriptor 有效',
    'normalize_descriptors': 'descriptor 單位正規化（分數門檻須重調）',
    'search_length_px': '沿極線搜尋長度 px',
    'search_width_px': '垂直極線搜尋寬度 px',
    'search_along_step_px': '沿極線步長 px',
    'search_across_step_px': '垂直極線步長 px',
    'keep_best_ratio': '最佳點保留比例',
    'keep_best_count': '固定保留點數（None 使用比例）',
    'max_group_score': 'BestG 上限（None 關閉）',
    'max_objective_score_ratio': 'ObjRatio 上限（None 關閉）',
    'epipolar_penalty_weight': '極線偏移懲罰權重',
    'second_best_exclusion_radius_px': '次佳候選排除半徑 px',
    'group_balance_weight': '全區塊平均分數權重',
    'frame_consistency_weight': '尺度／角度一致性權重',
    'frame_scale_tolerance_log2': '尺度一致性容許值 log2',
    'frame_angle_tolerance_deg': '角度一致性容許值 deg',
    'frame_min_reliable_pairs': '一致性最少可靠點對',
    'reject_uninformative_group': '拒絕全零 descriptor 群組',
    'reject_flat_score_surface': '拒絕平坦分數面',
    'flat_score_relative_tolerance': '平坦分數面容許差異',
    'warp_interpolation': 'Warp 插值 0=nearest / 1=linear / 2=cubic / 4=lanczos',
    'min_valid_warp_ratio': 'Warp 有效像素比例',
    'descriptor_border_margin_px': 'descriptor 邊界留白 px',
    'descriptor_batch_size': '每批 descriptor 數量',
    'specular_check_support': 'Reject SpecPts：True 檢查完整 support；False 只檢查中心',
}


def format_value(value):
    if isinstance(value, tuple):
        return ', '.join(str(item) for item in value)
    return str(value)


def _parse_value(text, annotation):
    text = str(text).strip()
    if get_origin(annotation) is Union:
        if text.lower() in ('', 'none'):
            return None
        annotation = next(item for item in get_args(annotation) if item is not type(None))
    if annotation is bool:
        if text.lower() not in ('true', 'false'):
            raise ValueError('請填 True 或 False')
        return text.lower() == 'true'
    if get_origin(annotation) is tuple:
        value = tuple(float(item.strip()) for item in text.strip('()[]').split(','))
        if not value or not all(math.isfinite(item) for item in value):
            raise ValueError('請填有限數值並以逗號分隔')
        return value
    value = annotation(text)
    if not math.isfinite(value):
        raise ValueError('必須是有限數字')
    return value


def parse_config_values(current, entries):
    """Parse all edits before returning a new immutable config; never mutate current."""
    annotations = get_type_hints(RegionSIFTConfig)
    changes = {}
    for name, text in entries.items():
        if name not in annotations:
            raise ValueError(f'未知參數: {name}')
        try:
            changes[name] = _parse_value(text, annotations[name])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name}: {exc}') from exc
    config = replace(current, **changes)
    try:
        _validate_config(config)
    except RegionSIFTError as exc:
        raise ValueError(str(exc)) from exc
    if config.min_point_distance_px < 0:
        raise ValueError('min_point_distance_px 不可小於 0')
    for name in ('scale_min_confidence', 'orientation_min_confidence'):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f'{name} 必須介於 0 和 1')
    total = config.grid_rows * config.grid_cols * config.points_per_cell + 1
    if config.keep_best_count is not None and not 1 <= config.keep_best_count <= total:
        raise ValueError(f'keep_best_count 必須介於 1 和 {total}，或填 None')
    capacity = config.cell_width_px * config.cell_height_px
    if config.points_per_cell > capacity - int(config.exclude_click_from_cell_points):
        raise ValueError('每格取點數超過可用像素數')
    if config.warp_interpolation not in (0, 1, 2, 4):
        raise ValueError('warp_interpolation 請使用 0、1、2 或 4')
    _scale_specs(config)  # Reject an automatic scale list entirely beyond support.
    return config


def show_region_sift_settings(parent, current, on_apply):
    """Modal child of the existing TkAgg window; apply is an atomic callback."""
    import tkinter as tk
    from tkinter import ttk, messagebox

    window = tk.Toplevel(parent)
    window.title('Region-SIFT 參數（本次執行）')
    window.geometry('1080x700')
    window.minsize(820, 450)
    window.transient(parent)
    ttk.Label(window, text='修改後按「套用」：下一次量測生效。未按套用不會變更；重新啟動後使用程式設定。',
              padding=10).pack(anchor='w')
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
            editor = ttk.Combobox(form, textvariable=variable, values=('True', 'False'), state='readonly')
        else:
            editor = ttk.Entry(form, textvariable=variable)
        editor.grid(row=row, column=1, sticky='ew', padx=6, pady=4)
        ttk.Label(form, text=LABELS.get(name, '')).grid(row=row, column=2, sticky='w', padx=6)

    def apply():
        try:
            updated = parse_config_values(current, {name: var.get() for name, var in variables.items()})
            on_apply(updated)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror('參數未套用', str(exc), parent=window)
            return
        window.destroy()

    def restore():
        for name, variable in variables.items():
            variable.set(format_value(getattr(current, name)))

    footer = ttk.Frame(window, padding=10)
    footer.pack(fill='x')
    ttk.Button(footer, text='還原開啟時的數值', command=restore).pack(side='left')
    ttk.Button(footer, text='取消', command=window.destroy).pack(side='right', padx=4)
    ttk.Button(footer, text='套用', command=apply).pack(side='right', padx=4)
    window.bind('<Escape>', lambda event: window.destroy())
    window.grab_set()
    window.focus_set()
    return window
