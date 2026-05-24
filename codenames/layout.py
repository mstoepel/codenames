"""Geometric helpers for finding and ordering the 5x5 layout in an image."""

from __future__ import annotations

import cv2
import numpy as np


def find_grid_bbox(image_bgr, value_threshold=150, kernel_size=60):
    """Locate the bounding box of the 5x5 card layout.

    Aggressively closes adjacent cards into one blob, then takes the largest
    contour. Returns (x, y, w, h) on the input image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, value_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Open with same kernel to drop wood-grain scratches / other small bright noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no card grid found in image")
    contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(contour), contour


def grid_skew_angle(contour):
    """Angle (deg, normalized to [-45, 45]) of the grid's min-area rectangle."""
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    w, h = rect[1]
    if w < h:
        angle = angle + 90
    while angle > 45:
        angle -= 90
    while angle <= -45:
        angle += 90
    return float(angle)


def deskew(image_bgr, angle_deg):
    """Rotate image by -angle_deg around its center, expanding the canvas to fit."""
    if abs(angle_deg) < 0.1:
        return image_bgr.copy()
    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        image_bgr, M, (new_w, new_h), borderValue=(0, 0, 0)
    )


def slice_bbox(bbox, rows=5, cols=5):
    """Divide bbox (x, y, w, h) into rows*cols uniform cells.

    Returns a `rows x cols` list of (x, y, w, h) tuples.
    """
    x, y, w, h = bbox
    cell_w = w / cols
    cell_h = h / rows
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cx = int(x + c * cell_w)
            cy = int(y + r * cell_h)
            row.append((cx, cy, int(cell_w), int(cell_h)))
        out.append(row)
    return out


def assign_to_grid(centroids, rows=5, cols=5):
    """Assign N=rows*cols centroids to grid cells.

    Sorts by y, chunks into `rows` row-groups, sorts each row by x.
    Returns a `rows x cols` array of indices into the original centroids list.
    """
    n = rows * cols
    if len(centroids) != n:
        raise ValueError(f"expected {n} centroids, got {len(centroids)}")
    indexed = sorted(enumerate(centroids), key=lambda t: t[1][1])
    grid = np.empty((rows, cols), dtype=int)
    for r in range(rows):
        row = indexed[r * cols : (r + 1) * cols]
        row.sort(key=lambda t: t[1][0])
        for c, (orig_idx, _) in enumerate(row):
            grid[r, c] = orig_idx
    return grid


def order_quad_corners(pts):
    """Return 4 points as (TL, TR, BR, BL) given any order.

    `pts` is a (4, 2) array-like.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # x - y... wait: diff is pts[:,1]-pts[:,0]
    # Use sum (min=TL, max=BR) and difference y-x (min=TR, max=BL)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = pts[:, 1] - pts[:, 0]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def warp_to_square(image_bgr, quad, size=500):
    """Perspective-warp `quad` (TL, TR, BR, BL) to a `size x size` square."""
    src = order_quad_corners(quad)
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, M, (size, size))


