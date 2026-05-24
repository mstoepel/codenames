# Codenames CV/OCR

A computer-vision + OCR pipeline that turns two photos — a 5×5 layout of
**word cards** (`Images/image1.JPG`) and a **spymaster key card**
(`Images/image2.JPG`) — into a "which words belong to which team" answer.

## Architecture

```
codenames/
├── layout.py      # geometric helpers: grid bbox, deskew, slice, perspective warp
├── fuzzy.py       # difflib-based closest-word lookup vs full_word_list.txt
├── word_cards.py  # extract_words(image) -> 5x5 of strings (OCR + fuzzy correct)
├── key_card.py    # extract_colors(image) -> 5x5 of red/blue/neutral/assassin
└── pipeline.py    # analyze(words_img, key_img) -> SpymasterView
```

Both images get sliced into a 5×5 grid, but along different paths:

- **Word cards**: threshold for bright pixels, aggressively close to fuse the
  25 cards into one blob, take the largest contour, deskew by its min-area
  rect angle, slice the bounding box into 5×5 uniform cells, crop the bottom
  half of each cell (top half is upside-down mirror text), OCR with
  `pytesseract` PSM 6, fuzzy-correct against `full_word_list.txt`.
- **Key card**: HSV-mask the red and blue cells (beige cells are
  indistinguishable from the brown frame), find the bounding rotated rectangle
  of those cells (which spans the full 5×5 grid because at least one corner
  cell is colored), perspective-warp to a canonical square, sample each cell's
  median HSV. Classify: any cell with >20% dark pixels is the assassin;
  otherwise low saturation → neutral, hue in red wedge → red, else blue.

## Running

```bash
brew install tesseract        # or: apt-get install tesseract-ocr
pip install -r requirements.txt
pytest -v                     # 15 tests, all should pass
streamlit run app.py          # open browser, hit Run with defaults
```

## Notes for editors

- `full_word_list.txt` is the Codenames base-game word list (~400 entries),
  uppercase, one per line. If you swap it, ensure all 25 words visible in
  `Images/image1.JPG` are still in the list, or `test_word_cards.py` will
  fail.
- HSV thresholds for the key card are module-level constants in
  `codenames/key_card.py` — tune there if a new key-card photo trips them.
- Tests gated on tesseract (`tests/conftest.py` `needs_tesseract`) skip when
  the binary isn't installed; pure-Python tests run unconditionally.
