"""BRM-style mammogram preprocessing (ported from the BRM repo, stage0_preprocess).

Self-contained port adapted for this project's JPEG multi-view inputs. Pipeline
per view image (grayscale float32, breast bright):

    ensure tissue bright  ->  Otsu breast mask (largest CC + morphology)
    ->  crop to breast bbox (+margin)  ->  zero out non-breast background
    ->  [MLO only] remove pectoral muscle triangle

The BRM repo keeps crop_to_breast OFF because it broke L/R correspondence for
their *registration* task. Here the task is breast-density classification, where
removing the black background and the pectoral muscle is beneficial, so it is ON.

References baked into the ported logic: Otsu + largest connected component for the
breast mask; Canny + Hough line for the pectoral edge (Karssemeijer 1998 / Kwok 2004).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# M0.1 — intensity orientation
# --------------------------------------------------------------------------- #
def ensure_tissue_bright(arr: np.ndarray) -> np.ndarray:
    """Make sure the air background (4 corners) is DARK and breast tissue BRIGHT;
    invert if the corners are bright. Robust to MONOCHROME1/2 inconsistencies."""
    arr = np.squeeze(arr).astype(np.float32)
    h, w = arr.shape[-2], arr.shape[-1]
    ch, cw = max(1, h // 20), max(1, w // 20)
    corners = np.concatenate([
        arr[:ch, :cw].ravel(), arr[:ch, -cw:].ravel(),
        arr[-ch:, :cw].ravel(), arr[-ch:, -cw:].ravel(),
    ])
    air = float(np.median(corners))
    lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    mid = 0.5 * (lo + hi)
    if air > mid:                      # corners (air) are bright -> inverted -> flip
        arr = float(arr.max()) - arr
    amin = float(arr.min())
    if amin < 0:
        arr = arr - amin
    return arr.astype(np.float32)


# --------------------------------------------------------------------------- #
# M0.2 — breast mask (Otsu + largest connected component + morphology)
# --------------------------------------------------------------------------- #
def _air_level(img: np.ndarray) -> float:
    """Estimate the air-background intensity from the 4 image corners (median is
    robust: a bright corner artifact doesn't move it as long as most corners are air)."""
    h, w = img.shape[-2], img.shape[-1]
    ch, cw = max(1, h // 20), max(1, w // 20)
    corners = np.concatenate([
        img[:ch, :cw].ravel(), img[:ch, -cw:].ravel(),
        img[-ch:, :cw].ravel(), img[-ch:, -cw:].ravel(),
    ])
    return float(np.median(corners))


def breast_mask(img: np.ndarray, closing_radius: int = 5, bg_frac: float = 0.06) -> np.ndarray:
    """Segment breast from background. Returns a bool mask (True = breast).

    Uses an inclusive threshold = min(Otsu, air + bg_frac*(p99 - air)). Plain Otsu
    lands on a mid-histogram valley for over-exposed or high-contrast breasts and
    then drops faint peripheral fibroglandular tissue; the background-relative
    floor keeps all tissue clearly above the air level.
    """
    from scipy.ndimage import binary_fill_holes
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.morphology import binary_closing, disk

    otsu = float(threshold_otsu(img))
    air = _air_level(img)
    p99 = float(np.percentile(img, 99))
    low = air + bg_frac * (p99 - air)
    thr = min(otsu, low)
    fg = img > thr

    lbl = label(fg)
    if lbl.max() == 0:
        return np.zeros(img.shape, dtype=bool)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0                       # drop background label
    mask = lbl == int(counts.argmax())

    mask = binary_closing(mask, disk(closing_radius))
    mask = binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)


def _resize(a: np.ndarray, h: int, w: int, order: int = 1) -> np.ndarray:
    from skimage.transform import resize
    return resize(a, (h, w), order=order, preserve_range=True,
                  anti_aliasing=(order > 0)).astype(np.float32)


# --------------------------------------------------------------------------- #
# M0.6 — crop to breast bounding box
# --------------------------------------------------------------------------- #
def breast_bbox(img: np.ndarray, closing_radius: int = 5, det_size: int = 512,
                margin_frac: float = 0.03) -> tuple[int, int, int, int]:
    """Breast bbox (x0,y0,x1,y1) in original coords; estimated on a low-res mask."""
    H, W = img.shape
    small = _resize(img, det_size, det_size)
    m = breast_mask(small, closing_radius=closing_radius)
    ys, xs = np.where(m)
    if xs.size == 0:
        return 0, 0, W, H
    sx, sy = W / det_size, H / det_size
    x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
    y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
    mw, mh = (x1 - x0) * margin_frac, (y1 - y0) * margin_frac
    x0, y0 = max(0.0, x0 - mw), max(0.0, y0 - mh)
    x1, y1 = min(float(W), x1 + mw), min(float(H), y1 + mh)
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, W, H
    return x0, y0, x1, y1


# --------------------------------------------------------------------------- #
# M0.4 — pectoral muscle mask (MLO only)
# --------------------------------------------------------------------------- #
def _signed(shape, theta: float, rho: float) -> np.ndarray:
    H, W = shape
    return (np.arange(W)[None, :] * np.cos(theta)
            + np.arange(H)[:, None] * np.sin(theta) - rho)


def _downsample(im, bm, max_dim: int = 800):
    from skimage.transform import rescale
    H, W = im.shape
    s = max_dim / max(H, W) if max(H, W) > max_dim else 1.0
    if s == 1.0:
        return im.astype(np.float32), bm, 1.0
    sim = rescale(im, s, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    sbm = rescale(bm.astype(float), s, order=0, preserve_range=True) > 0.5
    return sim, sbm, s


def _score_line(im, bm, theta, rho, band):
    s = _signed(im.shape, theta, rho)
    pec = (s <= 0) & bm
    area = int(pec.sum())
    if area == 0:
        return None
    ps = (s > -band) & (s <= 0) & bm
    bs = (s > 0) & (s < band) & bm
    if ps.sum() < 20 or bs.sum() < 20:
        return None
    return area, float(im[ps].mean() - im[bs].mean())


def _detect_pectoral_line(im, bm, top_frac, min_frac, max_frac,
                          angle_lo=8.0, angle_hi=82.0):
    from scipy.ndimage import binary_erosion
    from skimage.feature import canny
    from skimage.transform import hough_line, hough_line_peaks

    H, W = im.shape
    er_it = max(1, int(0.012 * max(H, W)))
    inside = binary_erosion(bm, iterations=er_it)
    roi = np.zeros((H, W), dtype=bool)
    roi[: int(H * top_frac), : int(W * 0.60)] = True
    search = inside & roi
    if search.sum() < 50:
        return None

    edges = canny(im, sigma=2.0, mask=search, use_quantiles=True,
                  low_threshold=0.80, high_threshold=0.92)
    if edges.sum() < 15:
        return None

    thetas = np.deg2rad(np.arange(angle_lo, angle_hi, 0.5))
    acc, angs, dists = hough_line(edges, theta=thetas)
    peaks = hough_line_peaks(acc, angs, dists, num_peaks=30,
                             threshold=0.15 * float(acc.max()))
    band = 0.03 * max(H, W)
    br = int(bm.sum())
    best, best_score = None, -1.0
    for votes, theta, rho in zip(*peaks):
        if rho <= 0 or np.cos(theta) <= 0 or np.sin(theta) <= 0:
            continue
        c0 = rho / np.cos(theta)
        r0 = rho / np.sin(theta)
        if not (0.02 * W < c0 < 0.98 * W):
            continue
        if not (0.05 * H < r0 < 1.6 * H):
            continue
        sc = _score_line(im, bm, theta, rho, band)
        if sc is None:
            continue
        area, contrast = sc
        frac = area / br
        if not (min_frac <= frac <= max_frac):
            continue
        if contrast <= 0:
            continue
        score = contrast * float(np.sqrt(area))
        if score > best_score:
            best_score, best = score, (float(theta), float(rho))
    return best


def _pectoral_plausible(pec: np.ndarray, bm: np.ndarray,
                        max_col_frac: float = 0.55, max_row_frac: float = 0.85,
                        max_area_frac: float = 0.30) -> bool:
    """Reject implausible pectoral masks that eat into the breast: a real MLO
    pectoral is a compact triangle pinned to the TOP-LEFT corner, so it must not
    reach far right, stretch to the bottom, or cover too much of the breast."""
    area = int(pec.sum())
    if area == 0:
        return False
    br = int(bm.sum())
    if br == 0 or area / br > max_area_frac:
        return False
    ys, xs = np.where(pec)
    H, W = pec.shape
    if xs.max() > max_col_frac * W:          # spills right past the corner
        return False
    if ys.max() > max_row_frac * H:          # reaches the bottom
        return False
    return True


def pectoral_mask_mlo(img: np.ndarray, bmask: np.ndarray, side: str,
                      top_frac: float = 0.55, min_area_frac: float = 0.01,
                      max_area_frac_of_breast: float = 0.30) -> np.ndarray:
    """Pectoral muscle mask for an MLO view. `side` in {'L','R'} = chest-wall side.

    Conservative: only returns a mask that looks like a real corner triangle,
    otherwise returns empty (leaving the muscle in is safer for density than
    carving out fibroglandular tissue by mistake)."""
    # Bring the muscle to the TOP-LEFT corner: mirror if chest wall is on the right.
    flip = side == "R"
    im = img[:, ::-1] if flip else img
    bm = bmask[:, ::-1] if flip else bmask

    sim, sbm, s = _downsample(im, bm)
    line = _detect_pectoral_line(sim, sbm, top_frac, min_area_frac,
                                 max_area_frac_of_breast, angle_lo=15.0, angle_hi=75.0)
    if line is None:
        pec = np.zeros(im.shape, dtype=bool)
    else:
        theta, rho = line
        pec = (_signed(im.shape, theta, rho / s) <= 0) & bm
        if not _pectoral_plausible(pec, bm, max_area_frac=max_area_frac_of_breast):
            pec = np.zeros(im.shape, dtype=bool)

    if flip:
        pec = pec[:, ::-1]
    return np.ascontiguousarray(pec, dtype=bool)


# --------------------------------------------------------------------------- #
# Top-level: preprocess one view
# --------------------------------------------------------------------------- #
def preprocess_view(gray: np.ndarray, view: str, side: str,
                    closing_radius: int = 5, margin_frac: float = 0.03,
                    zero_background: bool = True,
                    remove_pectoral: bool = False,
                    normalize: bool = True,
                    norm_low: float = 2.0, norm_high: float = 98.0) -> np.ndarray:
    """Full BRM-style preprocessing of ONE grayscale view.

    Steps: ensure tissue-bright -> Otsu breast mask -> crop to breast bbox ->
    [optional] pectoral removal (MLO) -> in-mask robust intensity normalization
    (map breast [p_low, p_high] to [0,255]) -> zero background. NOT resized.

    Args:
        gray: 2D grayscale array (any scale). Breast may be dark or bright.
        view: 'CC' or 'MLO'.
        side: 'L' or 'R' (breast laterality / chest-wall side).
        remove_pectoral: OFF by default. The Hough-based detector is unreliable
            on this data and over-removes into fibroglandular tissue, which would
            corrupt the density label; leaving the muscle in is the safer choice.
        normalize: per-image robust contrast normalization *within the breast
            mask*. Standard for mammography (removes exposure variation) and
            density-preserving (a linear map keeps tissue proportions/texture).
    Returns:
        float32 grayscale in [0, 255], cropped to the breast, background zeroed.
    """
    g = ensure_tissue_bright(gray)

    x0, y0, x1, y1 = breast_bbox(g, closing_radius=closing_radius, margin_frac=margin_frac)
    crop = g[y0:y1, x0:x1]

    bm = breast_mask(crop, closing_radius=closing_radius)

    if remove_pectoral and view.upper() == "MLO" and bm.any():
        pec = pectoral_mask_mlo(crop, bm, side)
        bm = bm & ~pec

    out = crop.astype(np.float32, copy=True)

    if normalize and bm.any():
        vals = out[bm]
        lo = float(np.percentile(vals, norm_low))
        hi = float(np.percentile(vals, norm_high))
        if hi > lo:
            out = np.clip((out - lo) / (hi - lo), 0.0, 1.0) * 255.0

    if zero_background and bm.any():
        out = np.where(bm, out, 0.0)

    return out.astype(np.float32)
