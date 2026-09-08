#!/usr/bin/env python3
"""Standalone OpenCV function benchmark for the Zebra depth/RT pipeline.

Default behavior:
  * deterministic synthetic inputs (seeded RNG)
  * 1920x1080 image-wide inputs where that reflects the real call site
  * 10 warm-up calls, then 100 measured calls per benchmark
  * GUI benchmarks disabled unless --include-gui is supplied
  * OpenCL disabled unless --enable-opencl is supplied
  * CSV + JSON output with environment metadata and input notes

Use --input-image PATH to make SIFT, ORB, corner/optical-flow and ArUco
benchmarks use the same real reference image on every platform. The default
--input-image-mode native preserves its pixels; benchmark-size resizes it to
--width x --height.

This is a wall-clock microbenchmark, not an end-to-end application benchmark.
Run it on otherwise idle machines and use identical command-line options when
comparing platforms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class BenchmarkCase:
    category: str
    name: str
    func: Callable[[Any], Any]
    input_description: str
    variability_note: str = ""
    prepare: Callable[[], Any] | None = None
    cleanup_iteration: Callable[[Any, Any], None] | None = None
    enabled: bool = True
    skip_reason: str = ""


@dataclass
class Result:
    category: str
    function_name: str
    status: str
    iterations: int
    warmup: int
    mean_ms: float | None
    median_ms: float | None
    p95_ms: float | None
    min_ms: float | None
    max_ms: float | None
    stdev_ms: float | None
    input_description: str
    variability_note: str
    error: str


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def timed_call(case: BenchmarkCase) -> float:
    context = case.prepare() if case.prepare else None
    result = None
    start = time.perf_counter_ns()
    try:
        result = case.func(context)
    finally:
        end = time.perf_counter_ns()
        if case.cleanup_iteration:
            case.cleanup_iteration(context, result)
    return (end - start) / 1_000_000.0


def run_case(case: BenchmarkCase, iterations: int, warmup: int) -> Result:
    if not case.enabled:
        return Result(case.category, case.name, "SKIPPED", 0, 0, None, None,
                      None, None, None, None, case.input_description,
                      case.variability_note, case.skip_reason)
    try:
        for _ in range(warmup):
            timed_call(case)
        samples = [timed_call(case) for _ in range(iterations)]
        return Result(
            case.category, case.name, "OK", iterations, warmup,
            statistics.fmean(samples), statistics.median(samples),
            percentile(samples, 0.95), min(samples), max(samples),
            statistics.pstdev(samples) if len(samples) > 1 else 0.0,
            case.input_description, case.variability_note, "")
    except Exception as exc:  # Keep the remaining suite running.
        return Result(case.category, case.name, "ERROR", 0, warmup, None,
                      None, None, None, None, None, case.input_description,
                      case.variability_note,
                      f"{type(exc).__name__}: {exc}")


def make_scene(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Create a repeatable textured 1080p-style scene without external files."""
    y = np.linspace(20, 210, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 45, width, dtype=np.float32)[None, :]
    base = np.clip(y + x, 0, 255).astype(np.uint8)
    image = np.dstack((base, np.roll(base, width // 11, axis=1),
                       np.flipud(base))).copy()
    noise = rng.normal(0, 8, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(140):
        x0 = int(rng.integers(0, max(1, width - 80)))
        y0 = int(rng.integers(0, max(1, height - 80)))
        x1 = min(width - 1, x0 + int(rng.integers(15, 180)))
        y1 = min(height - 1, y0 + int(rng.integers(15, 180)))
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.rectangle(image, (x0, y0), (x1, y1), color,
                      int(rng.integers(1, 4)))
    for i in range(30):
        org = (int(rng.integers(0, max(1, width - 160))),
               int(rng.integers(35, max(36, height))))
        cv2.putText(image, f"CV{i:02d}", org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def load_reference_image(image_path: Path, mode: str, width: int,
                         height: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a user image robustly, including Windows paths with non-ASCII text."""
    resolved = image_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input image does not exist: {resolved}")
    raw = resolved.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode input image: {resolved}")
    original_height, original_width = image.shape[:2]
    if mode == "benchmark-size" and (original_width != width or original_height != height):
        interpolation = (cv2.INTER_AREA
                         if original_width > width or original_height > height
                         else cv2.INTER_LINEAR)
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    effective_height, effective_width = image.shape[:2]
    info = {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "original_width": original_width,
        "original_height": original_height,
        "effective_width": effective_width,
        "effective_height": effective_height,
        "mode": mode,
    }
    return image, info


def make_aruco_scene(width: int, height: int) -> tuple[np.ndarray, Any, Any]:
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    canvas = np.full((height, width), 255, np.uint8)
    side = max(80, min(width, height) // 7)
    positions = [
        (width // 8, height // 8),
        (width - width // 8 - side, height // 8),
        (width // 8, height - height // 8 - side),
        (width - width // 8 - side, height - height // 8 - side),
    ]
    for marker_id, (x, y) in enumerate(positions):
        if hasattr(aruco, "generateImageMarker"):
            marker = aruco.generateImageMarker(dictionary, marker_id, side)
        else:
            marker = aruco.drawMarker(dictionary, marker_id, side)
        canvas[y:y + side, x:x + side] = marker
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    parameters = aruco.DetectorParameters() if hasattr(
        aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    return bgr, dictionary, parameters


def camera_geometry(rng: np.random.Generator, width: int, height: int) -> dict[str, Any]:
    fx = 0.9 * width
    k = np.array([[fx, 0.0, width / 2], [0.0, fx, height / 2],
                  [0.0, 0.0, 1.0]], np.float64)
    dist = np.array([-0.08, 0.02, 0.0005, -0.0003, 0.0], np.float64)
    rvec = np.array([0.025, -0.04, 0.015], np.float64)
    rotation = cv2.Rodrigues(rvec)[0]
    translation = np.array([[120.0], [4.0], [2.0]], np.float64)
    object_points = rng.uniform([-220, -150, 900], [220, 150, 1800],
                                size=(500, 3)).astype(np.float64)
    left, _ = cv2.projectPoints(object_points, np.zeros(3), np.zeros(3), k,
                                np.zeros(5))
    right, _ = cv2.projectPoints(object_points, rvec, translation, k,
                                 np.zeros(5))
    left = left.reshape(-1, 2)
    right = right.reshape(-1, 2)
    left += rng.normal(0, 0.25, left.shape)
    right += rng.normal(0, 0.25, right.shape)
    skew_t = np.array([[0.0, -translation[2, 0], translation[1, 0]],
                       [translation[2, 0], 0.0, -translation[0, 0]],
                       [-translation[1, 0], translation[0, 0], 0.0]])
    fundamental = np.linalg.inv(k).T @ skew_t @ rotation @ np.linalg.inv(k)
    p0 = k @ np.hstack((np.eye(3), np.zeros((3, 1))))
    p1 = k @ np.hstack((rotation, translation))
    return {
        "K": k, "dist": dist, "rvec": rvec, "R": rotation, "t": translation,
        "obj": object_points, "left": left, "right": right, "F": fundamental,
        "P0": p0, "P1": p1,
    }


def make_temp_video(temp_dir: Path, frame: np.ndarray, frame_count: int = 24) -> Path:
    path = temp_dir / "benchmark_input.avi"
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create temporary MJPG video")
    for i in range(frame_count):
        shifted = np.roll(frame, i * 2, axis=1)
        writer.write(shifted)
    writer.release()
    return path


def build_cases(args: argparse.Namespace, temp_dir: Path) -> tuple[list[BenchmarkCase], list[Callable[[], None]]]:
    rng = np.random.default_rng(args.seed)
    width, height = args.width, args.height
    scene = make_scene(rng, width, height)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    shifted_gray = np.roll(gray, 3, axis=1)
    if args.input_image is not None:
        content_scene, reference_info = load_reference_image(
            args.input_image, args.input_image_mode, width, height)
        args.reference_image_info = reference_info
        content_source = f"user image: {reference_info['path']}"
    else:
        content_scene = scene
        args.reference_image_info = None
        content_source = "deterministic synthetic scene"
    content_gray = cv2.cvtColor(content_scene, cv2.COLOR_BGR2GRAY)
    content_height, content_width = content_gray.shape
    content_shifted_gray = np.roll(content_gray, 3, axis=1)
    content_size_text = f"{content_width}x{content_height}"
    mask = np.zeros((height, width), np.uint8)
    for _ in range(80):
        center = tuple(int(v) for v in (rng.integers(0, width), rng.integers(0, height)))
        cv2.circle(mask, center, int(rng.integers(10, 90)), 255, -1)
    kernel = np.ones((5, 5), np.uint8)
    geom = camera_geometry(rng, width, height)
    synthetic_aruco_scene, aruco_dict, aruco_params = make_aruco_scene(width, height)
    if args.input_image is not None:
        aruco_gray = content_gray
        aruco_input_text = f"{content_size_text} {content_source}"
    else:
        aruco_gray = cv2.cvtColor(synthetic_aruco_scene, cv2.COLOR_BGR2GRAY)
        aruco_input_text = f"{width}x{height} synthetic scene with 4 clean markers"
    temp_video = make_temp_video(temp_dir, scene)
    cleanups: list[Callable[[], None]] = []
    cases: list[BenchmarkCase] = []

    def add(category: str, name: str, func: Callable[[Any], Any],
            input_description: str, variability_note: str = "",
            prepare: Callable[[], Any] | None = None,
            cleanup_iteration: Callable[[Any, Any], None] | None = None,
            enabled: bool = True, skip_reason: str = "") -> None:
        cases.append(BenchmarkCase(category, name, func, input_description,
                                   variability_note, prepare,
                                   cleanup_iteration, enabled, skip_reason))

    # 01 - Video and image I/O.
    io_note = "高度依賴作業系統、儲存裝置、影片 codec/backend 與快取；不是純 CPU 演算法耗時。"
    add("01 影像與影片輸入輸出", "cv2.VideoCapture()",
        lambda _: cv2.VideoCapture(str(temp_video)),
        f"Open {width}x{height} MJPG AVI",
        io_note, cleanup_iteration=lambda _c, cap: cap.release())
    shared_cap = cv2.VideoCapture(str(temp_video))
    cleanups.append(shared_cap.release)
    add("01 影像與影片輸入輸出", "cv2.VideoCapture.isOpened()",
        lambda _: shared_cap.isOpened(), "Already-open MJPG capture", io_note)

    def prepare_read() -> None:
        if shared_cap.get(cv2.CAP_PROP_POS_FRAMES) >= 23:
            shared_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return None

    add("01 影像與影片輸入輸出", "cv2.VideoCapture.read()",
        lambda _: shared_cap.read(), f"Decode one {width}x{height} MJPG frame",
        io_note, prepare=prepare_read)
    add("01 影像與影片輸入輸出", "cv2.VideoCapture.grab()",
        lambda _: shared_cap.grab(), f"Grab one {width}x{height} MJPG frame",
        io_note, prepare=prepare_read)

    def prepare_retrieve() -> None:
        prepare_read()
        shared_cap.grab()
        return None

    add("01 影像與影片輸入輸出", "cv2.VideoCapture.retrieve()",
        lambda _: shared_cap.retrieve(), f"Retrieve one {width}x{height} frame",
        io_note, prepare=prepare_retrieve)
    add("01 影像與影片輸入輸出", "cv2.VideoCapture.get()",
        lambda _: shared_cap.get(cv2.CAP_PROP_POS_FRAMES),
        "CAP_PROP_POS_FRAMES on open capture", io_note)
    pos_toggle = {"v": 0}
    add("01 影像與影片輸入輸出", "cv2.VideoCapture.set()",
        lambda _: shared_cap.set(cv2.CAP_PROP_POS_FRAMES,
                                 pos_toggle.__setitem__("v", 1 - pos_toggle["v"]) or pos_toggle["v"]),
        "Seek CAP_PROP_POS_FRAMES between frame 0/1", io_note)
    add("01 影像與影片輸入輸出", "cv2.VideoCapture.release()",
        lambda cap: cap.release(), "Release newly opened MJPG capture", io_note,
        prepare=lambda: cv2.VideoCapture(str(temp_video)))
    add("01 影像與影片輸入輸出", "cv2.VideoWriter_fourcc()",
        lambda _: cv2.VideoWriter_fourcc(*"MJPG"), "FOURCC='MJPG'",
        "幾乎只是代碼組合；跨平台差異通常很小。")

    writer_counter = {"n": 0}
    def new_writer() -> cv2.VideoWriter:
        writer_counter["n"] += 1
        p = temp_dir / f"ctor_{writer_counter['n']}.avi"
        return cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"),
                               30.0, (width, height))
    add("01 影像與影片輸入輸出", "cv2.VideoWriter()", lambda _: new_writer(),
        f"Open {width}x{height} MJPG writer", io_note,
        cleanup_iteration=lambda _c, writer: writer.release())
    write_path = temp_dir / "write_benchmark.avi"
    shared_writer = cv2.VideoWriter(str(write_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                    30.0, (width, height))
    cleanups.append(shared_writer.release)
    add("01 影像與影片輸入輸出", "cv2.VideoWriter.write()",
        lambda _: shared_writer.write(scene), f"Encode/write one {width}x{height} MJPG frame", io_note)
    add("01 影像與影片輸入輸出", "cv2.VideoWriter.release()",
        lambda writer: writer.release(), f"Finalize empty {width}x{height} MJPG AVI", io_note,
        prepare=new_writer)
    png_path = temp_dir / "imwrite_benchmark.png"
    add("01 影像與影片輸入輸出", "cv2.imwrite()",
        lambda _: cv2.imwrite(str(png_path), scene, [cv2.IMWRITE_PNG_COMPRESSION, 3]),
        f"{width}x{height} BGR uint8 -> PNG compression=3", io_note)

    # 02 - Calibration and preprocessing.
    add("02 相機校正與影像前處理", "cv2.cvtColor()",
        lambda _: cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY),
        f"{width}x{height} BGR uint8 -> gray")
    add("02 相機校正與影像前處理", "cv2.resize()",
        lambda _: cv2.resize(scene, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR),
        f"{width}x{height} -> {width//2}x{height//2} INTER_LINEAR")
    add("02 相機校正與影像前處理", "cv2.createCLAHE()",
        lambda _: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)),
        "clipLimit=2.0, tileGridSize=8x8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    add("02 相機校正與影像前處理", "cv2.CLAHE.apply()",
        lambda _: clahe.apply(gray), f"{width}x{height} gray uint8",
        "耗時會隨亮度分布與 tile 設定改變。")
    add("02 相機校正與影像前處理", "cv2.getOptimalNewCameraMatrix()",
        lambda _: cv2.getOptimalNewCameraMatrix(geom["K"], geom["dist"],
                                                 (width, height), 1.0, (width, height)),
        f"{width}x{height}, 5 distortion coefficients")
    new_k = cv2.getOptimalNewCameraMatrix(geom["K"], geom["dist"],
                                          (width, height), 1.0, (width, height))[0]
    add("02 相機校正與影像前處理", "cv2.initUndistortRectifyMap()",
        lambda _: cv2.initUndistortRectifyMap(geom["K"], geom["dist"], None,
                                              new_k, (width, height), cv2.CV_16SC2),
        f"{width}x{height}, CV_16SC2 maps")
    map1, map2 = cv2.initUndistortRectifyMap(geom["K"], geom["dist"], None,
                                            new_k, (width, height), cv2.CV_16SC2)
    add("02 相機校正與影像前處理", "cv2.remap()",
        lambda _: cv2.remap(scene, map1, map2, cv2.INTER_LINEAR),
        f"{width}x{height} BGR, CV_16SC2 map, INTER_LINEAR")
    add("02 相機校正與影像前處理", "cv2.undistortPoints()",
        lambda _: cv2.undistortPoints(geom["left"].reshape(-1, 1, 2),
                                      geom["K"], geom["dist"], P=geom["K"]),
        "500 2D points, 5 distortion coefficients")
    add("02 相機校正與影像前處理", "cv2.GaussianBlur()",
        lambda _: cv2.GaussianBlur(gray, (0, 0), 9),
        f"{width}x{height} gray, sigma=9")
    add("02 相機校正與影像前處理", "cv2.split()",
        lambda _: cv2.split(scene), f"{width}x{height} BGR uint8")
    add("02 相機校正與影像前處理", "cv2.morphologyEx()",
        lambda _: cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel),
        f"{width}x{height} binary mask, 5x5 CLOSE")
    add("02 相機校正與影像前處理", "cv2.dilate()",
        lambda _: cv2.dilate(mask, kernel, iterations=1),
        f"{width}x{height} binary mask, 5x5, 1 iteration")
    add("02 相機校正與影像前處理", "cv2.bitwise_or()",
        lambda _: cv2.bitwise_or(mask, np.roll(mask, 5, axis=1)),
        f"Two {width}x{height} uint8 masks")
    overlay = np.full_like(scene, (255, 80, 80))
    add("02 相機校正與影像前處理", "cv2.addWeighted()",
        lambda _: cv2.addWeighted(scene, 0.65, overlay, 0.35, 0),
        f"Two {width}x{height} BGR uint8 images")
    add("02 相機校正與影像前處理", "cv2.absdiff()",
        lambda _: cv2.absdiff(gray, shifted_gray),
        f"Two {width}x{height} gray uint8 images")

    # 03 - ArUco.
    aruco_note = "高度依賴標記數量、尺寸、清晰度、遮擋、背景紋理與 DetectorParameters。"
    add("03 ArUco 偵測與角點精修", "cv2.aruco.getPredefinedDictionary()",
        lambda _: cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
        "DICT_4X4_100")
    add("03 ArUco 偵測與角點精修", "cv2.aruco.DetectorParameters()",
        lambda _: cv2.aruco.DetectorParameters(), "Default parameters",
        enabled=hasattr(cv2.aruco, "DetectorParameters"),
        skip_reason="DetectorParameters unavailable in this OpenCV build")
    add("03 ArUco 偵測與角點精修", "cv2.aruco.DetectorParameters_create()",
        lambda _: cv2.aruco.DetectorParameters_create(), "Legacy default parameters",
        enabled=hasattr(cv2.aruco, "DetectorParameters_create"),
        skip_reason="Legacy DetectorParameters_create unavailable in this OpenCV build")
    add("03 ArUco 偵測與角點精修", "cv2.aruco.ArucoDetector()",
        lambda _: cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters()),
        "DICT_4X4_100, default parameters",
        enabled=hasattr(cv2.aruco, "ArucoDetector"),
        skip_reason="ArucoDetector unavailable in this OpenCV build")
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) if hasattr(
        cv2.aruco, "ArucoDetector") else None
    if aruco_detector is not None:
        detected_corners, detected_ids, _ = aruco_detector.detectMarkers(aruco_gray)
    elif hasattr(cv2.aruco, "detectMarkers"):
        detected_corners, detected_ids, _ = cv2.aruco.detectMarkers(
            aruco_gray, aruco_dict, parameters=aruco_params)
    else:
        detected_corners, detected_ids = [], None
    detected_marker_count = 0 if detected_ids is None else int(len(detected_ids))
    aruco_input_text += f", precheck detected markers={detected_marker_count}"
    add("03 ArUco 偵測與角點精修", "cv2.aruco.ArucoDetector.detectMarkers()",
        lambda _: aruco_detector.detectMarkers(aruco_gray),
        aruco_input_text, aruco_note,
        enabled=aruco_detector is not None,
        skip_reason="ArucoDetector unavailable in this OpenCV build")
    add("03 ArUco 偵測與角點精修", "cv2.aruco.detectMarkers()",
        lambda _: cv2.aruco.detectMarkers(aruco_gray, aruco_dict, parameters=aruco_params),
        aruco_input_text, aruco_note,
        enabled=hasattr(cv2.aruco, "detectMarkers"),
        skip_reason="Legacy detectMarkers unavailable in this OpenCV build")
    if detected_corners:
        corners = np.asarray(detected_corners[0], np.float32)
        corner_source = "first detected ArUco marker"
    else:
        cx, cy = content_width / 2, content_height / 2
        radius = max(3.0, min(content_width, content_height) * 0.08)
        corners = np.array([[[cx - radius, cy - radius], [cx + radius, cy - radius],
                             [cx + radius, cy + radius], [cx - radius, cy + radius]]],
                           np.float32)
        corner_source = "fallback center square (no ArUco marker detected)"
    add("03 ArUco 偵測與角點精修", "cv2.cornerSubPix()",
        lambda _: cv2.cornerSubPix(aruco_gray, corners.copy(), (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                    80, 1e-4)),
        f"{aruco_input_text}, 4 corners from {corner_source}, 5x5 window", aruco_note)

    # 04 - Features and matching. Local-patch operations intentionally use
    # realistic patch/ROI sizes instead of full HD; the dimensions are recorded.
    feature_note = "高度依賴影像紋理、特徵數、描述子數、搜尋區大小與 RANSAC/收斂狀況。"
    add("04 特徵擷取與立體匹配", "cv2.SIFT_create()",
        lambda _: cv2.SIFT_create(contrastThreshold=0.005),
        "contrastThreshold=0.005")
    add("04 特徵擷取與立體匹配", "cv2.ORB_create()",
        lambda _: cv2.ORB_create(nfeatures=1000), "nfeatures=1000")
    add("04 特徵擷取與立體匹配", "cv2.KeyPoint()",
        lambda _: cv2.KeyPoint(500.0, 400.0, 31.0), "One KeyPoint(size=31)")
    sift_detect = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.01)
    orb_detect = cv2.ORB_create(nfeatures=1500)
    sift_precheck_kp = sift_detect.detect(content_gray, None)
    orb_precheck_kp = orb_detect.detect(content_gray, None)
    add("04 特徵擷取與立體匹配", "Feature2D.detectAndCompute()[SIFT]",
        lambda _: sift_detect.detectAndCompute(content_gray, None),
        f"SIFT on {content_size_text} {content_source}, max 1500 features; "
        f"precheck keypoints={len(sift_precheck_kp)}", feature_note)
    add("04 特徵擷取與立體匹配", "Feature2D.detectAndCompute()[ORB]",
        lambda _: orb_detect.detectAndCompute(content_gray, None),
        f"ORB on {content_size_text} {content_source}, max 1500 features; "
        f"precheck keypoints={len(orb_precheck_kp)}", feature_note)
    low = np.array([5.0, 5.0])
    high = np.array([max(6.0, content_width - 5.0),
                     max(6.0, content_height - 5.0)])
    keypoints = [cv2.KeyPoint(float(x), float(y), 31.0)
                 for x, y in rng.uniform(low, high, (500, 2))]
    sift_compute = cv2.SIFT_create()
    orb_compute = cv2.ORB_create(nfeatures=1000)
    add("04 特徵擷取與立體匹配", "Feature2D.compute()[SIFT]",
        lambda _: sift_compute.compute(content_gray, keypoints),
        f"SIFT descriptors for 500 injected keypoints on {content_size_text} {content_source}",
        feature_note)
    add("04 特徵擷取與立體匹配", "Feature2D.compute()[ORB]",
        lambda _: orb_compute.compute(content_gray, keypoints),
        f"ORB descriptors for 500 injected keypoints on {content_size_text} {content_source}",
        feature_note)
    add("04 特徵擷取與立體匹配", "cv2.BFMatcher()",
        lambda _: cv2.BFMatcher(cv2.NORM_L2), "NORM_L2")
    desc_a = rng.normal(size=(1000, 128)).astype(np.float32)
    desc_b = (desc_a + rng.normal(0, 0.08, desc_a.shape)).astype(np.float32)
    bf_l2 = cv2.BFMatcher(cv2.NORM_L2)
    add("04 特徵擷取與立體匹配", "DescriptorMatcher.knnMatch()",
        lambda _: bf_l2.knnMatch(desc_a, desc_b, k=2),
        "1000x128 float32 vs 1000x128 float32, k=2", feature_note)
    add("04 特徵擷取與立體匹配", "DescriptorMatcher.match()",
        lambda _: bf_l2.match(desc_a, desc_b),
        "1000x128 float32 vs 1000x128 float32", feature_note)
    add("04 特徵擷取與立體匹配", "cv2.DMatch()",
        lambda _: cv2.DMatch(_queryIdx=10, _trainIdx=12, _imgIdx=0, _distance=0.25),
        "One DMatch record")
    binary_a = rng.integers(0, 256, (32,), np.uint8)
    binary_b = rng.integers(0, 256, (32,), np.uint8)
    add("04 特徵擷取與立體匹配", "cv2.norm()",
        lambda _: cv2.norm(binary_a, binary_b, cv2.NORM_HAMMING),
        "Two 32-byte ORB-style descriptors")
    patch = gray[height // 2 - 128:height // 2 + 128,
                 width // 2 - 128:width // 2 + 128].copy()
    add("04 特徵擷取與立體匹配", "cv2.Sobel()",
        lambda _: cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3),
        "256x256 gray patch, dx=1, dy=0, ksize=3", feature_note)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    add("04 特徵擷取與立體匹配", "cv2.magnitude()",
        lambda _: cv2.magnitude(gx, gy), "Two 256x256 float32 gradient arrays")
    add("04 特徵擷取與立體匹配", "cv2.sqrt()",
        lambda _: cv2.sqrt(gx * gx + gy * gy), "One 256x256 non-negative float32 array")
    add("04 特徵擷取與立體匹配", "cv2.cornerMinEigenVal()",
        lambda _: cv2.cornerMinEigenVal(patch, blockSize=3, ksize=3),
        "256x256 gray patch, blockSize=3, ksize=3", feature_note)
    add("04 特徵擷取與立體匹配", "cv2.goodFeaturesToTrack()",
        lambda _: cv2.goodFeaturesToTrack(content_gray, 1000, 0.01, 7),
        f"{content_size_text} {content_source}, maxCorners=1000", feature_note)
    flow_pts = cv2.goodFeaturesToTrack(content_gray, 1000, 0.01, 7)
    if flow_pts is None or len(flow_pts) == 0:
        xs = np.linspace(8, max(8, content_width - 9), 20)
        ys = np.linspace(8, max(8, content_height - 9), 15)
        flow_pts = np.array([[[x, y]] for y in ys for x in xs], np.float32)
        flow_point_source = "fallback 20x15 grid (no corners detected)"
    else:
        flow_point_source = f"goodFeaturesToTrack points={len(flow_pts)}"
    add("04 特徵擷取與立體匹配", "cv2.calcOpticalFlowPyrLK()",
        lambda _: cv2.calcOpticalFlowPyrLK(
            content_gray, content_shifted_gray, flow_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)),
        f"{content_size_text} {content_source}, 3-pixel horizontal shift, "
        f"{flow_point_source}, 4 pyramid levels", feature_note)
    roi = gray[height // 2 - 256:height // 2 + 256,
               width // 2 - 256:width // 2 + 256]
    template = roi[210:274, 220:284].copy()
    add("04 特徵擷取與立體匹配", "cv2.matchTemplate()",
        lambda _: cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED),
        "512x512 search ROI, 64x64 template, TM_CCOEFF_NORMED", feature_note)
    match_map = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    add("04 特徵擷取與立體匹配", "cv2.minMaxLoc()",
        lambda _: cv2.minMaxLoc(match_map), f"{match_map.shape[1]}x{match_map.shape[0]} float32 score map")
    add("04 特徵擷取與立體匹配", "cv2.pyrDown()",
        lambda _: cv2.pyrDown(roi), "512x512 gray -> 256x256")
    ecc_template = cv2.resize(template, (128, 128))
    ecc_input = np.roll(ecc_template, 2, axis=1)
    add("04 特徵擷取與立體匹配", "cv2.findTransformECC()",
        lambda _: cv2.findTransformECC(
            ecc_template, ecc_input, np.eye(2, 3, dtype=np.float32),
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)),
        "128x128 gray pair, translation model, max 50 iterations", feature_note)
    hsv_patch_a = hsv[100:356, 100:356]
    hsv_patch_b = hsv[110:366, 110:366]
    add("04 特徵擷取與立體匹配", "cv2.calcHist()",
        lambda _: cv2.calcHist([hsv_patch_a], [0, 1], None, [18, 16], [0, 180, 0, 256]),
        "256x256 HSV patch, H/S histogram 18x16", feature_note)
    hist_a = cv2.calcHist([hsv_patch_a], [0, 1], None, [18, 16], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_patch_b], [0, 1], None, [18, 16], [0, 180, 0, 256])
    add("04 特徵擷取與立體匹配", "cv2.normalize()",
        lambda _: cv2.normalize(hist_a, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX),
        "18x16 float32 histogram")
    hist_an = cv2.normalize(hist_a, None, 0, 1, cv2.NORM_MINMAX)
    hist_bn = cv2.normalize(hist_b, None, 0, 1, cv2.NORM_MINMAX)
    add("04 特徵擷取與立體匹配", "cv2.compareHist()",
        lambda _: cv2.compareHist(hist_an, hist_bn, cv2.HISTCMP_BHATTACHARYYA),
        "Two normalized 18x16 histograms", feature_note)

    # 05 - RT and geometry.
    rt_note = "耗時依點數、雜訊、離群值比例、初始值、RANSAC 迭代與收斂難度而變。"
    obj100 = geom["obj"][:100].astype(np.float32)
    img100 = geom["left"][:100].astype(np.float32)
    add("05 RT、幾何估計與三角化", "cv2.solvePnP()",
        lambda _: cv2.solvePnP(obj100, img100, geom["K"], np.zeros(5),
                               flags=cv2.SOLVEPNP_ITERATIVE),
        "100 non-planar 3D-2D points, SOLVEPNP_ITERATIVE", rt_note)
    square = np.array([[-50, 50, 0], [50, 50, 0], [50, -50, 0],
                       [-50, -50, 0]], np.float32)
    square_img, _ = cv2.projectPoints(square, np.array([0.1, -0.05, 0.02]),
                                      np.array([0, 0, 700.0]), geom["K"], np.zeros(5))
    add("05 RT、幾何估計與三角化", "cv2.solvePnPGeneric()",
        lambda _: cv2.solvePnPGeneric(square, square_img, geom["K"], np.zeros(5),
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE),
        "4 square planar points, SOLVEPNP_IPPE_SQUARE", rt_note)
    img_ransac = img100.copy()
    img_ransac[:12] += rng.normal(0, 35, (12, 2)).astype(np.float32)
    add("05 RT、幾何估計與三角化", "cv2.solvePnPRansac()",
        lambda _: cv2.solvePnPRansac(obj100, img_ransac, geom["K"], np.zeros(5),
                                     iterationsCount=150, reprojectionError=2.0,
                                     confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE),
        "100 non-planar points, 12 synthetic outliers, max 150 iterations", rt_note)
    add("05 RT、幾何估計與三角化", "cv2.Rodrigues()",
        lambda _: cv2.Rodrigues(geom["rvec"]), "One float64 3x1 rotation vector")
    add("05 RT、幾何估計與三角化", "cv2.projectPoints()",
        lambda _: cv2.projectPoints(geom["obj"], geom["rvec"], geom["t"],
                                    geom["K"], np.zeros(5)),
        "500 3D points, no distortion", rt_note)
    left500 = geom["left"].astype(np.float64)
    right500 = geom["right"].astype(np.float64)
    add("05 RT、幾何估計與三角化", "cv2.findEssentialMat()",
        lambda _: cv2.findEssentialMat(left500, right500, geom["K"],
                                       method=cv2.RANSAC, prob=0.999, threshold=1.5),
        "500 noisy point pairs, RANSAC threshold=1.5", rt_note)
    essential, essential_mask = cv2.findEssentialMat(left500, right500, geom["K"],
                                                      method=cv2.RANSAC, prob=0.999,
                                                      threshold=1.5)
    add("05 RT、幾何估計與三角化", "cv2.recoverPose()",
        lambda _: cv2.recoverPose(essential, left500, right500, geom["K"],
                                  mask=essential_mask.copy()),
        "500 noisy point pairs and precomputed essential matrix", rt_note)
    h_true = np.array([[1.01, 0.015, 7.0], [-0.01, 0.995, 5.0],
                       [1e-5, -1e-5, 1.0]], np.float64)
    h_src = rng.uniform([0, 0], [width, height], (500, 2)).astype(np.float32)
    h_dst = cv2.perspectiveTransform(h_src.reshape(-1, 1, 2), h_true).reshape(-1, 2)
    h_dst += rng.normal(0, 0.35, h_dst.shape).astype(np.float32)
    add("05 RT、幾何估計與三角化", "cv2.findHomography()",
        lambda _: cv2.findHomography(h_src, h_dst, cv2.RANSAC, 2.0),
        "500 point pairs, RANSAC threshold=2.0", rt_note)
    add("05 RT、幾何估計與三角化", "cv2.estimateAffinePartial2D()",
        lambda _: cv2.estimateAffinePartial2D(h_src, h_dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=2.0),
        "500 point pairs, RANSAC threshold=2.0", rt_note)
    add("05 RT、幾何估計與三角化", "cv2.correctMatches()",
        lambda _: cv2.correctMatches(geom["F"], left500.reshape(1, -1, 2),
                                     right500.reshape(1, -1, 2)),
        "500 point pairs and precomputed fundamental matrix", rt_note)
    add("05 RT、幾何估計與三角化", "cv2.triangulatePoints()",
        lambda _: cv2.triangulatePoints(geom["P0"], geom["P1"],
                                        left500.T, right500.T),
        "500 point pairs, two 3x4 projection matrices", rt_note)
    warp_h = np.array([[1, 0.01, 5], [-0.01, 1, 3], [1e-6, 1e-6, 1]], np.float64)
    add("05 RT、幾何估計與三角化", "cv2.warpPerspective()",
        lambda _: cv2.warpPerspective(gray, warp_h, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT),
        f"{width}x{height} gray, INTER_LINEAR")

    # 06 - Regions and quality.
    contour_note = "耗時依輪廓數量、遮罩複雜度、連通區數與點數改變。"
    add("06 輪廓、區域與品質評估", "cv2.findContours()",
        lambda _: cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE),
        f"{width}x{height} binary mask with ~80 blobs", contour_note)
    contour = np.array([[100, 100], [800, 120], [900, 700], [400, 900],
                        [120, 600]], np.float32)
    add("06 輪廓、區域與品質評估", "cv2.contourArea()",
        lambda _: cv2.contourArea(contour), "5-point float32 contour")
    add("06 輪廓、區域與品質評估", "cv2.minAreaRect()",
        lambda _: cv2.minAreaRect(contour), "5-point float32 contour")
    rect = cv2.minAreaRect(contour)
    add("06 輪廓、區域與品質評估", "cv2.boxPoints()",
        lambda _: cv2.boxPoints(rect), "One rotated rectangle")
    hull_points = rng.uniform([0, 0], [width, height], (1000, 2)).astype(np.float32)
    add("06 輪廓、區域與品質評估", "cv2.convexHull()",
        lambda _: cv2.convexHull(hull_points), "1000 random 2D points", contour_note)
    polygon = np.array([[100, 100], [width - 100, 180],
                        [width - 200, height - 100], [180, height - 160]], np.int32)
    fill_target = np.zeros((height, width), np.uint8)
    add("06 輪廓、區域與品質評估", "cv2.fillConvexPoly()",
        lambda _: cv2.fillConvexPoly(fill_target, polygon, 255),
        f"{width}x{height} uint8 target, 4-point polygon")
    add("06 輪廓、區域與品質評估", "cv2.connectedComponentsWithStats()",
        lambda _: cv2.connectedComponentsWithStats(mask),
        f"{width}x{height} binary mask with ~80 blobs", contour_note)
    add("06 輪廓、區域與品質評估", "cv2.Laplacian()",
        lambda _: cv2.Laplacian(gray, cv2.CV_64F),
        f"{width}x{height} gray -> CV_64F")

    # 07 - Drawing/UI. Drawing calls are safe headlessly; window calls are opt-in.
    ui_note = "繪圖成本依線長、粗細、反鋸齒、文字長度與圖元數量改變；不代表 RT/深度計算。"
    draw_target = scene.copy()
    add("07 UI 顯示與診斷繪圖", "cv2.rectangle()",
        lambda _: cv2.rectangle(draw_target, (100, 100), (800, 600),
                                (0, 255, 0), 3, cv2.LINE_AA),
        f"{width}x{height} BGR, 700x500 outline, thickness=3", ui_note)
    add("07 UI 顯示與診斷繪圖", "cv2.putText()",
        lambda _: cv2.putText(draw_target, "OpenCV benchmark", (100, 200),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                              2, cv2.LINE_AA),
        f"{width}x{height} BGR, 16 characters, scale=1, thickness=2", ui_note)
    add("07 UI 顯示與診斷繪圖", "cv2.getTextSize()",
        lambda _: cv2.getTextSize("OpenCV benchmark", cv2.FONT_HERSHEY_SIMPLEX,
                                  1.0, 2),
        "16 characters, scale=1, thickness=2", ui_note)
    add("07 UI 顯示與診斷繪圖", "cv2.circle()",
        lambda _: cv2.circle(draw_target, (width // 2, height // 2), 30,
                             (0, 0, 255), 3, cv2.LINE_AA),
        f"{width}x{height} BGR, radius=30, thickness=3", ui_note)
    add("07 UI 顯示與診斷繪圖", "cv2.line()",
        lambda _: cv2.line(draw_target, (100, 100), (width - 100, height - 100),
                           (255, 0, 0), 4, cv2.LINE_AA),
        f"{width}x{height} BGR, long diagonal, thickness=4", ui_note)
    polyline = polygon.reshape(-1, 1, 2)
    add("07 UI 顯示與診斷繪圖", "cv2.polylines()",
        lambda _: cv2.polylines(draw_target, [polyline], True, (255, 255, 0),
                                3, cv2.LINE_AA),
        f"{width}x{height} BGR, closed 4-point polygon", ui_note)
    add("07 UI 顯示與診斷繪圖", "cv2.drawContours()",
        lambda _: cv2.drawContours(draw_target, [polyline], -1, (0, 255, 255),
                                   3, cv2.LINE_AA),
        f"{width}x{height} BGR, one 4-point contour", ui_note)

    gui_enabled = bool(args.include_gui)
    add("07 UI 顯示與診斷繪圖", "cv2.imshow()",
        lambda _: cv2.imshow("OpenCV benchmark", scene),
        f"{width}x{height} BGR window", "包含 GUI/backend 與視窗系統影響；imshow 本身可能非同步。",
        prepare=lambda: cv2.namedWindow("OpenCV benchmark", cv2.WINDOW_NORMAL),
        enabled=gui_enabled, skip_reason="Use --include-gui to benchmark window functions")
    add("07 UI 顯示與診斷繪圖", "cv2.waitKey()",
        lambda _: cv2.waitKey(1), "waitKey(1) with an OpenCV window",
        "至少含約 1 ms 等待與 OS 事件處理，不是純函式運算時間。",
        prepare=lambda: cv2.namedWindow("OpenCV benchmark", cv2.WINDOW_NORMAL),
        enabled=gui_enabled, skip_reason="Use --include-gui to benchmark window functions")
    add("07 UI 顯示與診斷繪圖", "cv2.destroyAllWindows()",
        lambda _: cv2.destroyAllWindows(), "Destroy one prepared OpenCV window",
        "主要反映 GUI/backend 與作業系統視窗管理。",
        prepare=lambda: cv2.namedWindow("OpenCV benchmark", cv2.WINDOW_NORMAL),
        enabled=gui_enabled, skip_reason="Use --include-gui to benchmark window functions")
    if gui_enabled:
        cleanups.append(cv2.destroyAllWindows)

    return cases, cleanups


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    build = cv2.getBuildInformation()
    interesting = {}
    for key in ("Parallel framework", "CPU/HW features", "GUI", "Video I/O",
                "Intel IPP", "OpenCL"):
        match = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", build, re.MULTILINE)
        if match:
            interesting[key] = match.group(1).strip()
    return {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "command": " ".join(sys.argv),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "opencv_num_threads": cv2.getNumThreads(),
        "opencv_use_optimized": cv2.useOptimized(),
        "opencv_opencl_available": bool(cv2.ocl.haveOpenCL()),
        "opencv_opencl_enabled": bool(cv2.ocl.useOpenCL()),
        "width": args.width,
        "height": args.height,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "seed": args.seed,
        "filter": args.filter,
        "include_gui": args.include_gui,
        "reference_image": getattr(args, "reference_image_info", None),
        "build_summary": interesting,
    }


def write_outputs(output_dir: Path, metadata: dict[str, Any], results: list[Result]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "opencv_function_timings.csv"
    json_path = output_dir / "opencv_function_timings.json"
    fields = list(Result.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata,
                   "results": [asdict(result) for result in results]},
                  f, ensure_ascii=False, indent=2)
    print(f"\nCSV : {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenCV functions used by the Zebra stereo/RT pipeline.")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Measured calls per function (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Unmeasured warm-up calls per function (default: 10)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--input-image", type=Path, default=None,
        help=("Reference image for SIFT, ORB, goodFeaturesToTrack, optical flow, "
              "ArUco detection and cornerSubPix; synthetic image is used when omitted"))
    parser.add_argument(
        "--input-image-mode", choices=("native", "benchmark-size"), default="native",
        help=("Use reference image at native resolution (default), or resize it to "
              "--width x --height before content-sensitive benchmarks"))
    parser.add_argument("--filter", default="",
                        help="Regex applied to category + function name")
    parser.add_argument("--include-gui", action="store_true",
                        help="Include imshow/waitKey/destroyAllWindows")
    parser.add_argument("--enable-opencl", action="store_true",
                        help="Enable OpenCL if this OpenCV build supports it")
    parser.add_argument("--opencv-threads", type=int, default=None,
                        help="Set cv2 thread count; omit to keep OpenCV default")
    parser.add_argument("--list", action="store_true",
                        help="List benchmark names without running them")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be >= 1 and --warmup must be >= 0")
    if args.width < 64 or args.height < 64:
        parser.error("--width and --height must both be >= 64")
    return args


def main() -> int:
    args = parse_args()
    cv2.setUseOptimized(True)
    cv2.ocl.setUseOpenCL(bool(args.enable_opencl))
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    output_dir = args.output_dir or Path(
        f"opencv_benchmark_results_{datetime.now():%Y%m%d_%H%M%S}")

    print("Preparing deterministic synthetic inputs and temporary 1080p video...")
    with tempfile.TemporaryDirectory(prefix="opencv_bench_") as temp_name:
        cases, cleanups = build_cases(args, Path(temp_name))
        try:
            if args.filter:
                pattern = re.compile(args.filter, re.IGNORECASE)
                cases = [case for case in cases
                         if pattern.search(f"{case.category} {case.name}")]
            if args.list:
                for case in cases:
                    status = "enabled" if case.enabled else f"skipped: {case.skip_reason}"
                    print(f"{case.category}\t{case.name}\t{status}")
                return 0
            metadata = environment_metadata(args)
            results: list[Result] = []
            total = len(cases)
            for index, case in enumerate(cases, 1):
                print(f"[{index:02d}/{total:02d}] {case.name} ... ", end="", flush=True)
                result = run_case(case, args.iterations, args.warmup)
                results.append(result)
                if result.status == "OK":
                    print(f"mean={result.mean_ms:.4f} ms, p95={result.p95_ms:.4f} ms")
                else:
                    print(f"{result.status}: {result.error}")
            write_outputs(output_dir, metadata, results)
            ok = sum(r.status == "OK" for r in results)
            skipped = sum(r.status == "SKIPPED" for r in results)
            errors = sum(r.status == "ERROR" for r in results)
            print(f"Completed: OK={ok}, SKIPPED={skipped}, ERROR={errors}")
            return 1 if errors else 0
        finally:
            for cleanup in reversed(cleanups):
                try:
                    cleanup()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
