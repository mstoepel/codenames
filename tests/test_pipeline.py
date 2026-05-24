from .conftest import needs_tesseract


@needs_tesseract
def test_analyze_returns_full_spymaster_view(image1_bgr, image2_bgr):
    from codenames import analyze

    view = analyze(image1_bgr, image2_bgr)
    pairs = [pair for row in view.grid for pair in row]
    assert len(pairs) == 25
    assert (
        len(view.red) + len(view.blue) + len(view.neutral) + len(view.assassin)
        <= 25
    )
    # Most words should have been OCR'd successfully.
    ocr_ok = sum(1 for (w, _) in pairs if w is not None)
    assert ocr_ok >= 20
