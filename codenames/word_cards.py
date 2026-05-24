"""Extract the 5x5 grid of words from a photo of the Codenames word cards."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from . import layout
from .fuzzy import closest_word


@dataclass
class WordCardsResult:
    grid: list[list[str | None]]
    raw_grid: list[list[str]]
    deskewed: np.ndarray
    cell_boxes: list[list[tuple[int, int, int, int]]]


def _ocr_card(card_bgr: np.ndarray) -> str:
    """OCR a single card crop and return raw uppercase text."""
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    target = 400
    if max(h, w) < target:
        scale = target / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Pad with white border so tesseract doesn't think the edge is part of a glyph
    binar = cv2.copyMakeBorder(binar, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    text = pytesseract.image_to_string(binar, config=config)
    return "".join(ch for ch in text.upper() if ch.isalpha() or ch == "-")


def extract_words(image_bgr: np.ndarray) -> WordCardsResult:
    """Detect 5x5 grid of word cards, OCR each, fuzzy-match against the word list.

    Strategy: find the bounding box of the card region (one blob via aggressive
    morphological close), deskew based on its min-area rect, slice into 25
    uniform cells, OCR the bottom half of each cell.
    """
    bbox, contour = layout.find_grid_bbox(image_bgr)
    angle = layout.grid_skew_angle(contour)
    deskewed = layout.deskew(image_bgr, angle)
    # Re-find the bbox on the deskewed image (now axis-aligned).
    bbox2, _ = layout.find_grid_bbox(deskewed)
    cells = layout.slice_bbox(bbox2)

    raw_grid: list[list[str]] = [["" for _ in range(5)] for _ in range(5)]
    out_grid: list[list[str | None]] = [[None for _ in range(5)] for _ in range(5)]
    cell_boxes: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(5)] for _ in range(5)
    ]
    for r in range(5):
        for c in range(5):
            x, y, w, h = cells[r][c]
            cell_boxes[r][c] = (x, y, w, h)
            # Bottom half: the right-side-up word label.
            # Pad inward so we miss the card border / inter-card gap.
            pad_x = int(w * 0.10)
            pad_top = int(h * 0.05)
            pad_bot = int(h * 0.18)  # bottom border + the dark gap below the card
            y0 = y + h // 2 + pad_top
            y1 = y + h - pad_bot
            x0 = x + pad_x
            x1 = x + w - pad_x
            if y0 >= y1 or x0 >= x1:
                continue
            crop = deskewed[y0:y1, x0:x1]
            raw = _ocr_card(crop)
            raw_grid[r][c] = raw
            out_grid[r][c] = closest_word(raw)

    return WordCardsResult(
        grid=out_grid, raw_grid=raw_grid, deskewed=deskewed, cell_boxes=cell_boxes
    )


def annotate(result: WordCardsResult) -> np.ndarray:
    """Return a copy of the deskewed image with detected cells outlined and indexed."""
    img = result.deskewed.copy()
    for r in range(5):
        for c in range(5):
            x, y, w, h = result.cell_boxes[r][c]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"{r},{c}"
            cv2.putText(
                img, label, (x + 8, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
    return img
