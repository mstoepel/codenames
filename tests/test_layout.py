import numpy as np

from codenames import layout


def test_assign_to_grid_jittered_centroids():
    # Build 25 points on a regular grid with small jitter; verify permutation.
    rng = np.random.default_rng(0)
    pts = []
    expected = np.empty((5, 5), dtype=int)
    idx = 0
    # Build in a non-row-major order to make sure the function sorts them.
    raw = []
    for r in range(5):
        for c in range(5):
            x = 100 + c * 50 + rng.uniform(-5, 5)
            y = 100 + r * 50 + rng.uniform(-5, 5)
            raw.append(((x, y), (r, c)))
    rng.shuffle(raw)
    pts = [p for p, _ in raw]
    truth = {i: rc for i, (_, rc) in enumerate(raw)}
    grid = layout.assign_to_grid(pts)
    assert grid.shape == (5, 5)
    assert sorted(grid.ravel().tolist()) == list(range(25))
    for r in range(5):
        for c in range(5):
            orig = int(grid[r, c])
            assert truth[orig] == (r, c)


def test_order_quad_corners_canonical_order():
    pts = np.array([[10, 200], [200, 10], [10, 10], [200, 200]], dtype=np.float32)
    ordered = layout.order_quad_corners(pts)
    assert tuple(ordered[0]) == (10.0, 10.0)   # TL
    assert tuple(ordered[1]) == (200.0, 10.0)  # TR
    assert tuple(ordered[2]) == (200.0, 200.0) # BR
    assert tuple(ordered[3]) == (10.0, 200.0)  # BL


def test_deskew_no_rotation_is_near_identity():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    out = layout.deskew(img, 0.0)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_assign_to_grid_wrong_count_raises():
    import pytest
    with pytest.raises(ValueError):
        layout.assign_to_grid([(0, 0)] * 24)
