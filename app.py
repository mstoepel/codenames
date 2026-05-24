"""Streamlit demo for the Codenames CV/OCR pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from codenames import analyze
from codenames.key_card import TEAM_BGR, annotate as annotate_key
from codenames.word_cards import annotate as annotate_words

ROOT = Path(__file__).resolve().parent
DEFAULT_WORDS_IMG = ROOT / "Images" / "image1.JPG"
DEFAULT_KEY_IMG = ROOT / "Images" / "image2.JPG"


TEAM_HEX = {
    "red": "#dc2626",
    "blue": "#2563eb",
    "neutral": "#d1c7a8",
    "assassin": "#111111",
}
TEAM_TEXT = {
    "red": "white",
    "blue": "white",
    "neutral": "#111111",
    "assassin": "white",
}


def _read_image(uploaded, fallback_path: Path) -> np.ndarray:
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode upload: {uploaded.name}")
        return img
    img = cv2.imread(str(fallback_path))
    if img is None:
        raise FileNotFoundError(f"missing default image: {fallback_path}")
    return img


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _grid_html(view) -> str:
    rows = []
    for r in range(5):
        cells = []
        for c in range(5):
            word, team = view.grid[r][c]
            bg = TEAM_HEX[team]
            fg = TEAM_TEXT[team]
            label = word if word else "?"
            cells.append(
                f'<td style="background:{bg};color:{fg};padding:14px 8px;'
                f"text-align:center;font-weight:600;font-family:sans-serif;"
                f'border:2px solid #fff;width:18%;">{label}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table style="border-collapse:collapse;width:100%;table-layout:fixed;">'
        + "".join(rows)
        + "</table>"
    )


def main():
    st.set_page_config(page_title="Codenames Spymaster CV", layout="wide")
    st.title("Codenames spymaster CV")
    st.caption(
        "Upload a photo of the word cards (5x5) and the key card, or hit Run "
        "to use the bundled defaults."
    )

    col1, col2 = st.columns(2)
    with col1:
        words_upload = st.file_uploader(
            "Word cards photo", type=["jpg", "jpeg", "png"], key="words"
        )
    with col2:
        key_upload = st.file_uploader(
            "Key card photo", type=["jpg", "jpeg", "png"], key="key"
        )

    if st.button("Run", type="primary"):
        with st.spinner("Detecting cards, OCR'ing words, classifying colors..."):
            words_img = _read_image(words_upload, DEFAULT_WORDS_IMG)
            key_img = _read_image(key_upload, DEFAULT_KEY_IMG)
            view = analyze(words_img, key_img)

        st.subheader("Spymaster view")
        st.markdown(_grid_html(view), unsafe_allow_html=True)

        cols = st.columns(4)
        for col, label, words, bg in zip(
            cols,
            ["Red", "Blue", "Neutral", "Assassin"],
            [view.red, view.blue, view.neutral, view.assassin],
            ["#dc2626", "#2563eb", "#d1c7a8", "#111111"],
        ):
            with col:
                st.markdown(
                    f"<div style='background:{bg};color:white;padding:6px 12px;"
                    f"font-weight:600;border-radius:4px;'>{label} ({len(words)})</div>",
                    unsafe_allow_html=True,
                )
                for w in words:
                    st.write(w)

        st.subheader("Debug overlays")
        d1, d2 = st.columns(2)
        with d1:
            st.image(
                _bgr_to_rgb(annotate_words(view.words_result)),
                caption="Word cards: detected cells",
                use_container_width=True,
            )
        with d2:
            st.image(
                _bgr_to_rgb(annotate_key(view.key_result)),
                caption="Key card: classified cells",
                use_container_width=True,
            )

        with st.expander("Raw OCR output (pre-fuzzy)"):
            raw = view.words_result.raw_grid
            for r in range(5):
                st.text(" | ".join(f"{raw[r][c]:<12}" for c in range(5)))


if __name__ == "__main__":
    main()
