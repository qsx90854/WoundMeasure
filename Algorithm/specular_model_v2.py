"""Conservative local dichromatic-model detector, independent of legacy thresholds.

I ~= a*d + s*e in approximate linear RGB. d is estimated from compatible
neighbours, e is the configured illuminant colour. Scores are evidence scores,
NOT calibrated probabilities. Near-neutral, clipped and unsupported pixels can
remain uncertain. Only ``mask`` is suitable for the binary masked-SIFT API.
"""
from dataclasses import dataclass, fields
import math
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class SpecularModelV2Config:
    v2_input_gamma: float = 2.2
    v2_light_rgb: tuple[float, ...] = (1.0, 1.0, 1.0)
    v2_reference_radii_px: tuple[int, ...] = (4, 8, 16)
    v2_reference_hue_sigma: float = 0.25
    v2_reference_chroma_power: float = 2.0
    v2_min_reference_affinity: float = 0.35
    v2_min_reference_samples: float = 6.0
    v2_reference_dispersion: float = 0.12
    v2_min_separation: float = 0.12
    v2_noise_floor: float = 0.008
    v2_noise_relative: float = 0.025
    v2_model_penalty: float = 3.0
    v2_min_specular_fraction: float = 0.12
    v2_seed_score: float = 0.65
    v2_grow_score: float = 0.45
    v2_uncertain_score: float = 0.20
    v2_grow_distance_px: int = 3
    v2_clip_level: int = 250
    v2_clip_channels: int = 2
    v2_show_uncertain: bool = True


def validate_specular_model_v2_config(config):
    if not isinstance(config, SpecularModelV2Config):
        raise ValueError('Expected SpecularModelV2Config')
    for field in fields(config):
        value = getattr(config, field.name)
        for number in value if isinstance(value, tuple) else (value,):
            if not isinstance(number, (int, float, np.number, bool)) or not math.isfinite(number):
                raise ValueError(f'{field.name}: must contain finite numbers')
    if len(config.v2_light_rgb) != 3 or min(config.v2_light_rgb) <= 0:
        raise ValueError('v2_light_rgb: three positive RGB values required')
    radii = config.v2_reference_radii_px
    if (not radii or any(type(r) is not int or not 1 <= r <= 128 for r in radii)
            or tuple(sorted(set(radii))) != radii or len(radii) > 8):
        raise ValueError('v2_reference_radii_px: 1..8 increasing distinct integer radii, each 1..128')
    if not 0.1 <= config.v2_input_gamma <= 4:
        raise ValueError('v2_input_gamma: expected 0.1..4 (1 for linear input)')
    for name in ('v2_reference_hue_sigma', 'v2_reference_chroma_power',
                 'v2_min_reference_samples', 'v2_reference_dispersion', 'v2_noise_floor'):
        if getattr(config, name) <= 0:
            raise ValueError(f'{name}: must be positive')
    if config.v2_min_reference_samples > 8 * len(radii):
        raise ValueError('v2_min_reference_samples: exceeds available reference samples')
    if config.v2_noise_relative < 0 or config.v2_model_penalty < 0:
        raise ValueError('v2_noise_relative / v2_model_penalty: must be nonnegative')
    for name in ('v2_min_separation', 'v2_min_specular_fraction', 'v2_min_reference_affinity'):
        if not 0 < getattr(config, name) < 1:
            raise ValueError(f'{name}: must be between 0 and 1')
    if not 0 <= config.v2_uncertain_score <= config.v2_grow_score <= config.v2_seed_score <= 1:
        raise ValueError('scores: 0 <= uncertain <= grow <= seed <= 1')
    if config.v2_seed_score <= 0 or config.v2_grow_score <= 0:
        raise ValueError('seed / grow scores must be positive')
    for name, lo, hi in (('v2_grow_distance_px', 0, 32),
                         ('v2_clip_level', 1, 255), ('v2_clip_channels', 1, 3)):
        value = getattr(config, name)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'{name}: integer {lo}..{hi} required')
    if type(config.v2_show_uncertain) is not bool:
        raise ValueError('v2_show_uncertain: True or False required')
    return config


def _unit(v):
    norm = np.linalg.norm(v, axis=-1)
    return v / np.maximum(norm[..., None], 1e-8), norm


def fit_dichromatic_models(rgb, diffuse, illuminant):
    """Exact 2-column nonnegative least-squares, including both boundary cases."""
    ie = np.sum(rgb * illuminant, axis=-1)
    id_ = np.sum(rgb * diffuse, axis=-1)
    de = np.clip(np.sum(diffuse * illuminant, axis=-1), -1, 1)
    determinant = np.maximum(1.0 - de * de, 0)
    a = (id_ - de * ie) / np.maximum(determinant, 1e-8)
    s = (ie - de * id_) / np.maximum(determinant, 1e-8)
    residual_a = rgb - np.maximum(id_, 0)[..., None] * diffuse
    error_a = np.sum(residual_a ** 2, axis=-1)
    error_e = np.sum((rgb - np.maximum(ie, 0)[..., None] * illuminant) ** 2, axis=-1)
    interior_error = np.sum((rgb - a[..., None] * diffuse - s[..., None] * illuminant) ** 2, axis=-1)
    interior_ok = (a >= 0) & (s >= 0) & (determinant > 1e-8)
    error_b = np.minimum(error_a, error_e)
    e_wins = error_e < error_a
    best_a = np.where(e_wins, 0, np.maximum(id_, 0))
    best_s = np.where(e_wins, np.maximum(ie, 0), 0)
    interior_wins = interior_ok & (interior_error < error_b)
    error_b = np.where(interior_wins, interior_error, error_b)
    best_a = np.where(interior_wins, a, best_a)
    best_s = np.where(interior_wins, s, best_s)
    return error_a, error_b, best_a, best_s, np.sqrt(determinant)


def detect_specular_model_v2(bgr, config=SpecularModelV2Config(), *, return_debug=False):
    """Return a uint8 exclusion mask; optionally return evidence/uncertainty maps.

    References use 8 directions at each radius, weighted by illuminant-orthogonal
    hue agreement and chroma (less contaminated neighbours have more weight).
    No percentile quotas, legacy thresholds, closing or unconditional white gate.
    Computation is full resolution with streamed references, O(H*W*radii).
    """
    validate_specular_model_v2_config(config)
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
        raise ValueError('V2 expects a nonempty uint8 BGR image')
    started = time.perf_counter()
    rgb = (bgr[..., ::-1].astype(np.float32) / 255.0) ** config.v2_input_gamma
    light = np.asarray(config.v2_light_rgb, np.float32)
    light /= np.linalg.norm(light)
    colour, intensity = _unit(rgb)
    hue, chroma = _unit(colour - np.sum(colour * light, axis=-1)[..., None] * light)
    clipped = (np.count_nonzero(bgr >= config.v2_clip_level, axis=-1) >= config.v2_clip_channels)
    h, w = intensity.shape
    sum_w = np.zeros((h, w), np.float32)
    sum_w2 = np.zeros_like(sum_w)
    sum_base_w = np.zeros_like(sum_w)
    sum_colour = np.zeros_like(rgb)
    sum_norm2 = np.zeros_like(sum_w)
    # Sample only real pixels: no replicated/wrapped border references.
    for radius in config.v2_reference_radii_px:
        diagonal = max(1, int(round(radius / np.sqrt(2))))
        for dy, dx in ((0, radius), (0, -radius), (radius, 0), (-radius, 0),
                       (diagonal, diagonal), (diagonal, -diagonal),
                       (-diagonal, diagonal), (-diagonal, -diagonal)):
            if abs(dy) >= h or abs(dx) >= w:
                continue
            y0, y1 = max(0, -dy), min(h, h-dy)
            x0, x1 = max(0, -dx), min(w, w-dx)
            dst = np.s_[y0:y1, x0:x1]
            src = np.s_[y0+dy:y1+dy, x0+dx:x1+dx]
            hue_error = np.sum((hue[dst] - hue[src]) ** 2, axis=-1)
            # Hue becomes unobservable near the illuminant colour.
            hue_reliability = np.minimum(1, chroma[dst] / config.v2_min_separation)
            weight = np.exp(-hue_error * hue_reliability ** 2 /
                            (2 * config.v2_reference_hue_sigma ** 2))
            base_weight = chroma[src] ** config.v2_reference_chroma_power
            base_weight *= (~clipped[src]) & (intensity[src] > config.v2_noise_floor)
            weight *= base_weight
            sum_base_w[dst] += base_weight
            sum_w[dst] += weight
            sum_w2[dst] += weight ** 2
            sum_colour[dst] += weight[..., None] * colour[src]
            sum_norm2[dst] += weight * np.sum(colour[src] ** 2, axis=-1)
    mean_colour = sum_colour / np.maximum(sum_w[..., None], 1e-8)
    diffuse, mean_norm = _unit(mean_colour)
    effective_samples = sum_w ** 2 / np.maximum(sum_w2, 1e-16)
    variance = np.maximum(sum_norm2 / np.maximum(sum_w, 1e-8) - mean_norm ** 2, 0)
    reference_quality = np.minimum(1, effective_samples / config.v2_min_reference_samples)
    reference_quality *= np.exp(-variance / config.v2_reference_dispersion ** 2)
    affinity = sum_w / np.maximum(sum_base_w, 1e-8)
    reference_quality *= affinity
    supported = ((sum_w > 1e-7) & (effective_samples >= config.v2_min_reference_samples)
                 & (affinity >= config.v2_min_reference_affinity))
    error_a, error_b, a, s, separation = fit_dichromatic_models(rgb, diffuse, light)
    noise2 = (config.v2_noise_floor + config.v2_noise_relative * intensity) ** 2
    gain = np.maximum((error_a - error_b) / noise2 - config.v2_model_penalty, 0)
    fraction = s / np.maximum(a+s, 1e-8)
    score = (gain / (gain + 1)) * np.minimum(1, fraction / config.v2_min_specular_fraction)
    raw_evidence = score * np.exp(-error_b / (3 * noise2))
    score = raw_evidence * reference_quality
    score *= np.minimum(1, separation / (2 * config.v2_min_separation))
    score[~supported] = 0
    score = np.clip(score, 0, 1).astype(np.float32)
    identifiable = supported & (separation >= config.v2_min_separation) & ~clipped
    seeds = (score >= config.v2_seed_score) & identifiable
    allowed = (score >= config.v2_grow_score) & identifiable
    accepted = seeds.copy()
    # Geodesic growth may only visit pixels with model evidence, for a bounded
    # number of steps. It cannot close/fill arbitrary gaps in normal tissue.
    for _ in range(config.v2_grow_distance_px):
        expanded = (cv2.dilate(accepted.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & allowed
        if np.array_equal(expanded, accepted):
            break
        accepted = expanded
    unresolved = ((~supported | (separation < config.v2_min_separation))
                  & (intensity > config.v2_noise_floor))
    uncertain = ((raw_evidence >= config.v2_uncertain_score) | clipped | unresolved) & ~accepted
    mask = accepted.astype(np.uint8) * 255
    if not return_debug:
        return mask
    return mask, dict(
        score=score, uncertain_mask=uncertain.astype(np.uint8)*255,
        clipped_mask=clipped.astype(np.uint8)*255, seed_mask=seeds.astype(np.uint8)*255,
        reference_quality=reference_quality, reference_affinity=affinity, effective_samples=effective_samples,
        separation=separation, specular_fraction=fraction,
        diffuse_error=np.sqrt(error_a), specular_error=np.sqrt(error_b),
        model_gain=gain, elapsed_ms=(time.perf_counter()-started)*1000,
        mask_fraction=float(accepted.mean()), uncertain_fraction=float(uncertain.mean()))


def overlay_specular_v2_uncertain(rgb, uncertain_mask, alpha=0.4):
    """Orange uncertainty is display-only and never feeds the SIFT exclusion mask."""
    if uncertain_mask is None:
        return rgb
    out = rgb.copy()
    selected = np.asarray(uncertain_mask) > 0
    out[selected] = np.rint((1-alpha)*out[selected] + alpha*np.array([255, 175, 40])).astype(np.uint8)
    return out
