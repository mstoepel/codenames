"""Extract the 5x5 grid of team colors from a photo of the spymaster key card."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from . import layout

Team = Literal["red", "blue", "neutral", "assassin"]

# HSV thresholds tuned for Images/image2.JPG (OpenCV HSV: H 0-179, S 0-255, V 0-255).
# Implement as ordered checks so wedge boundaries don't matter.
ASSASSIN_DARK_V = 100        # value below this counts as "dark"
ASSASSIN_DARK_FRAC = 0.20    # fraction of dark pixels needed to call a cell assassin
NEUTRAL_S_MAX = 50           # saturation below this -> beige/neutral
RED_H_LO = 15                # H <= this -> red (or H >= 165)
RED_H_HI = 165

WARP_SIZE = 500
SAMPLE_FRAC = 0.55           # patch size as fraction of cell width


@dataclass
class KeyCardResult:
    grid: list[list[Team]]
    warped: np.ndarray  # WARP_SIZE x WARP_SIZE
    cell_size: int


def _cells_mask(image_hsv: np.ndarray) -> np.ndarray:
    """Mask covering only the colored cells (red + blue), excluding the brown frame.

    Beige (neutral) cells and the brown frame are both low-saturation, so we
    can't separate them by HSV — instead we locate red+blue cells and let the
    bounding rect span the full grid because the corner cells include colored
    ones.
    """
    h, s, v = cv2.split(image_hsv)
    red = ((h <= 15) | (h >= 165)) & (s > 100) & (v > 80)
    blue = (h >= 90) & (h <= 130) & (s > 100) & (v > 80)
    mask = (red | blue).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _grid_quad(mask: np.ndarray) -> np.ndarray:
    """Find the 4 corners of the 5x5 grid via filtered cell contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no colored cells found on key card")
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    if not areas:
        raise ValueError("no colored cells found on key card")
    # Filter to cell-sized contours: drop the LED reflections at the frame edges
    # (much smaller than cells). Use 60% of the median cell area as the cutoff.
    median = areas[len(areas) // 2]
    cells = [c for c in contours if cv2.contourArea(c) >= median * 0.6]
    if len(cells) < 10:
        raise ValueError(f"only found {len(cells)} cell contours; expected >= 10")
    all_pts = np.vstack(cells).reshape(-1, 2)
    rect = cv2.minAreaRect(all_pts)
    box = cv2.boxPoints(rect)
    # Expand slightly outward so we capture the full cell area, not just
    # the inner colored circles/diamonds.
    cx, cy = rect[0]
    expansion = 1.05
    box = np.array([
        [cx + (p[0] - cx) * expansion, cy + (p[1] - cy) * expansion] for p in box
    ], dtype=np.float32)
    return box


def _classify_patch(patch_hsv: np.ndarray) -> Team:
    # The assassin cell is black with a white X painted on top. Detect it by
    # the fraction of dark pixels — non-assassin cells have ~0% dark pixels.
    flat = patch_hsv.reshape(-1, 3)
    dark_frac = float((flat[:, 2] < ASSASSIN_DARK_V).mean())
    if dark_frac > ASSASSIN_DARK_FRAC:
        return "assassin"
    h, s, _v = np.median(flat, axis=0)
    if s < NEUTRAL_S_MAX:
        return "neutral"
    if h <= RED_H_LO or h >= RED_H_HI:
        return "red"
    return "blue"


def extract_colors(image_bgr: np.ndarray) -> KeyCardResult:
    """Locate the inner 5x5 grid, warp it square, classify each cell.

    Returns a KeyCardResult.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = _cells_mask(hsv)
    quad = _grid_quad(mask)
    warped = layout.warp_to_square(image_bgr, quad, size=WARP_SIZE)
    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    cell = WARP_SIZE // 5
    patch_half = max(2, int(cell * SAMPLE_FRAC / 2))
    grid: list[list[Team]] = [["neutral"] * 5 for _ in range(5)]
    for r in range(5):
        for c in range(5):
            cx = c * cell + cell // 2
            cy = r * cell + cell // 2
            patch = warped_hsv[
                cy - patch_half : cy + patch_half,
                cx - patch_half : cx + patch_half,
            ]
            grid[r][c] = _classify_patch(patch)
    return KeyCardResult(grid=grid, warped=warped, cell_size=cell)


TEAM_BGR: dict[Team, tuple[int, int, int]] = {
    "red": (0, 0, 220),
    "blue": (220, 80, 0),
    "neutral": (200, 220, 240),
    "assassin": (30, 30, 30),
}


def annotate(result: KeyCardResult) -> np.ndarray:
    """Return the warped image with each cell tinted by its classified team."""
    img = result.warped.copy()
    cell = result.cell_size
    overlay = img.copy()
    for r in range(5):
        for c in range(5):
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell, y0 + cell
            cv2.rectangle(overlay, (x0, y0), (x1, y1), TEAM_BGR[result.grid[r][c]], -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    for r in range(5):
        for c in range(5):
            x0, y0 = c * cell, r * cell
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (255, 255, 255), 2)
    return img
