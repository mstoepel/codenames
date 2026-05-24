from codenames.fuzzy import closest_word, load_word_list


def test_corrects_single_letter_swap():
    assert closest_word("STREAN") == "STREAM"


def test_corrects_trailing_letter_error():
    assert closest_word("MICROSCOPF") == "MICROSCOPE"


def test_returns_none_below_cutoff():
    assert closest_word("XQZ123") is None


def test_returns_none_for_empty():
    assert closest_word("") is None


def test_word_list_is_uppercase():
    words = load_word_list()
    assert words
    assert all(w == w.upper() for w in words)
    assert "MICROSCOPE" in words
    assert "STREAM" in words
