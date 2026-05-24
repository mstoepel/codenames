from collections import Counter

# Ground truth from Images/image2.JPG, sampled from the cell centers.
# Standard 9 blue / 8 red / 7 neutral / 1 assassin distribution.
KEY_GRID = [
    ["blue",    "blue",    "neutral", "red",     "neutral"],
    ["red",     "blue",    "blue",    "neutral", "blue"],
    ["red",     "blue",    "red",     "blue",    "neutral"],
    ["red",     "red",     "blue",    "red",     "neutral"],
    ["blue",    "neutral", "assassin","neutral", "red"],
]


def test_extract_colors_counts_match_distribution(image2_bgr):
    from codenames.key_card import extract_colors

    result = extract_colors(image2_bgr)
    flat = [t for row in result.grid for t in row]
    counts = Counter(flat)
    assert sum(counts.values()) == 25
    assert counts["assassin"] == 1
    assert counts["red"] >= 5
    assert counts["blue"] >= 5
    assert counts["neutral"] >= 5


def test_extract_colors_assassin_at_expected_cell(image2_bgr):
    from codenames.key_card import extract_colors

    result = extract_colors(image2_bgr)
    assassins = [
        (r, c) for r in range(5) for c in range(5) if result.grid[r][c] == "assassin"
    ]
    assert assassins == [(4, 2)]


def test_extract_colors_full_grid(image2_bgr):
    from codenames.key_card import extract_colors

    result = extract_colors(image2_bgr)
    mismatches = []
    for r in range(5):
        for c in range(5):
            if result.grid[r][c] != KEY_GRID[r][c]:
                mismatches.append((r, c, result.grid[r][c], KEY_GRID[r][c]))
    assert not mismatches, f"{len(mismatches)} mismatches: {mismatches}"
