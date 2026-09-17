"""Explicit ArUco allowlist: None allows all; an empty list allows none."""
from numbers import Integral
import numpy as np


def normalize_allowed_ids(allowed_ids):
    if allowed_ids is None:
        return None
    result = set()
    for value in allowed_ids:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError('Allowed ArUco IDs must be nonnegative integers')
        result.add(int(value))
    return frozenset(result)


def filter_marker_detections(corners, ids, allowed_ids):
    allowed = normalize_allowed_ids(allowed_ids)
    if allowed is None or ids is None:
        return corners, ids
    values = np.asarray(ids).reshape(-1)
    indices = [i for i, mid in enumerate(values) if int(mid) in allowed]
    if not indices:
        return [], None
    return [corners[i] for i in indices], values[indices].reshape(-1, 1)


def filter_marker_mapping(mapping, allowed_ids):
    allowed = normalize_allowed_ids(allowed_ids)
    return {int(mid): value for mid, value in mapping.items()
            if allowed is None or int(mid) in allowed}
