from .conftest import needs_tesseract

CANONICAL_GRID = [
    ["CLIFF",   "BANK",    "HAM",   "EYE",     "CAP"],
    ["BATTERY", "DICE",    "LEMON", "LION",    "BOOT"],
    ["DROP",    "POST",    "BOARD", "KID",     "VET"],
    ["PRESS",   "CALF",    "DAY",   "MAMMOTH", "HEAD"],
    ["STREAM",  "CENTAUR", "WAVE",  "ENGINE",  "MICROSCOPE"],
]


@needs_tesseract
def test_extract_words_finds_most_words_in_correct_cells(image1_bgr):
    from codenames.word_cards import extract_words

    result = extract_words(image1_bgr)
    assert len(result.grid) == 5
    assert all(len(row) == 5 for row in result.grid)

    correct = 0
    for r in range(5):
        for c in range(5):
            if result.grid[r][c] == CANONICAL_GRID[r][c]:
                correct += 1
    assert correct >= 20, (
        f"only {correct}/25 words OCR'd correctly; got grid:\n"
        + "\n".join(
            " ".join(f"{(result.grid[r][c] or '?'):<10}" for c in range(5))
            for r in range(5)
        )
    )


@needs_tesseract
def test_extract_words_corner_alignment(image1_bgr):
    from codenames.word_cards import extract_words

    result = extract_words(image1_bgr)
    assert result.grid[0][0] == "CLIFF"
    assert result.grid[0][4] == "CAP"
    assert result.grid[4][0] == "STREAM"
    assert result.grid[4][4] == "MICROSCOPE"
