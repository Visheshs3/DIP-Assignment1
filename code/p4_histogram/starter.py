"""P4 starter. Implement every function marked TODO.

Dataset map (relative to the bundle root):
  4.1  images/p4/equalization/input.png
  4.2  images/p4/matching/{source,reference}.png
       images/p4/specification/source.png
       images/p4/colour_source.png
  4.3  images/p4/local_regions.png
  4.4  images/p4/exposure/ev0.png ... ev4.png
"""
import numpy as np
from pathlib import Path


def hist(img, bins=256):
    return np.bincount(img.ravel(), minlength=bins).astype(np.float64)


def cdf(h):
    """TODO 4.1: normalised cumulative histogram."""
    c = np.cumsum(h, dtype=np.float64)
    return c / c[-1]


def equalise(img):
    """TODO 4.1: CDF-based global equalisation."""
    h = hist(img)
    F = cdf(h)
    lut = np.round(255.0 * F).astype(np.uint8)
    return lut[img]


def specify(img, target_hist):
    """TODO 4.2: match img's histogram to target_hist."""
    h_src = hist(img)
    F_src = cdf(h_src)
    F_tgt = cdf(target_hist)
    lut = np.zeros(256, dtype=np.uint8)

    for v in range(256):
        diff = np.abs(F_tgt - F_src[v])
        best_z = np.argmin(diff)
        lut[v] = best_z

    return lut[img]


def wasserstein1(h1, h2):
    F1 = np.cumsum(h1, dtype=np.float64)
    F1 /= F1[-1]

    F2 = np.cumsum(h2, dtype=np.float64)
    F2 /= F2[-1]

    return float(np.sum(np.abs(F1 - F2)))


def ahe(img, tiles=8):
    """TODO 4.3: plain tiled AHE. No clipping, no interpolation. This one is
    SUPPOSED to look bad -- that is the point."""
    H, W = img.shape
    out = np.zeros_like(img)

    # Determine tile boundary intervals
    y_edges = np.linspace(0, H, tiles + 1, dtype=int)
    x_edges = np.linspace(0, W, tiles + 1, dtype=int)

    for i in range(tiles):
        for j in range(tiles):
            r0, r1 = y_edges[i], y_edges[i + 1]
            c0, c1 = x_edges[j], x_edges[j + 1]

            tile = img[r0:r1, c0:c1]
            

            out[r0:r1, c0:c1] = equalise(tile)

    return out


def clahe(img, tiles=8, clip=3.0, bins=256):
    """TODO 4.3: clip at `clip` x mean bin height, redistribute the excess,
    and bilinearly interpolate between the four surrounding tile LUTs.

    Watch the tile-centre offset. That is where the marks go."""

    H, W = img.shape

    y_edges = np.linspace(0, H, tiles + 1, dtype=int)
    x_edges = np.linspace(0, W, tiles + 1, dtype=int)

    # Tile centers in continuous pixel coordinates
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:] - 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:] - 1)

    # Compute clipped and redistributed LUT for each tile
    luts = np.zeros((tiles, tiles, bins), dtype=np.float64)

    for i in range(tiles):
        for j in range(tiles):
            r0, r1 = y_edges[i], y_edges[i + 1]
            c0, c1 = x_edges[j], x_edges[j + 1]
            tile = img[r0:r1, c0:c1]

            h = hist(tile, bins=bins)

            if clip < np.inf:
                mean_height = tile.size / float(bins)
                limit = clip * mean_height
                excess = np.sum(np.maximum(h - limit, 0.0))
                h_clipped = np.minimum(h, limit)
                h_clipped += excess / float(bins)
            else:
                h_clipped = h

            F = np.cumsum(h_clipped) / np.sum(h_clipped)
            luts[i, j] = np.round((bins - 1) * F)

    dy = ( y_centers[1] - y_centers[0] if tiles > 1 else 1.0 )
    dx = ( x_centers[1] - x_centers[0] if tiles > 1 else 1.0 )

    y_coords = np.arange(H)
    x_coords = np.arange(W)

    v = (y_coords - y_centers[0]) / dy
    u = (x_coords - x_centers[0]) / dx

    v_clamped = np.clip(v, 0.0, tiles - 1)
    u_clamped = np.clip(u, 0.0, tiles - 1)

    i0 = np.clip(np.floor(v_clamped).astype(int), 0, max(0, tiles - 2))
    i1 = np.minimum(i0 + 1, tiles - 1)
    j0 = np.clip(np.floor(u_clamped).astype(int), 0, max(0, tiles - 2))
    j1 = np.minimum(j0 + 1, tiles - 1)

    beta = (v_clamped - i0)[:, None]  # Shape (H, 1)
    alpha = (u_clamped - j0)[None, :]  # Shape (1, W)

    w00 = (1.0 - beta) * (1.0 - alpha)
    w01 = (1.0 - beta) * alpha
    w10 = beta * (1.0 - alpha)
    w11 = beta * alpha

    v00 = luts[i0[:, None], j0[None, :], img]
    v01 = luts[i0[:, None], j1[None, :], img]
    v10 = luts[i1[:, None], j0[None, :], img]
    v11 = luts[i1[:, None], j1[None, :], img]

    out = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    return np.clip(np.round(out), 0, bins - 1).astype(np.uint8)

def apply_on_luma(rgb, fn):
    """TODO 4.2: run a greyscale operator on luma only, preserving hue."""
    rgb_f = rgb.astype(np.float64)
    Y = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    Y_u8 = np.clip(np.round(Y), 0, 255).astype(np.uint8)

    Y_proc = fn(Y_u8).astype(np.float64)
    ratio = np.zeros_like(Y)
    nonzero = Y > 1e-6
    ratio[nonzero] = Y_proc[nonzero] / Y[nonzero]
    rgb_out = rgb_f * ratio[..., None]
    return np.clip(np.round(rgb_out), 0, 255).astype(np.uint8)

def auto_correct(rgb):
    """TODO 4.4: no parameters. Estimate what you need from the image."""
    rgb_f = rgb.astype(np.float64)
    Y = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    v_low = np.percentile(Y, 1.0)
    v_high = np.percentile(Y, 99.0)
    denom = max(v_high - v_low, 1e-4)
    u = np.clip((Y - v_low) / denom, 0.0, 1.0)

    unclipped = Y < 255
    # print(np.sum(Y==255))
    if np.sum(unclipped) > 0:
        m = np.median(u[unclipped])
    else:
        m = np.median(u)
    # m = np.median(u)

    # print(f"median = {m}")
    m_clamped = np.clip(m, 1e-3, 1.0 - 1e-3)
    gamma = np.log(0.5) / np.log(m_clamped)
    # print(gamma)
    gamma = np.clip(gamma, -np.inf, 5)
    Y_new = np.clip(255.0 * (u**gamma), 0.0, 255.0)
    ratio = np.zeros_like(Y)
    nonzero = Y > 1e-4
    ratio[nonzero] = Y_new[nonzero] / Y[nonzero]

    rgb_out = np.clip(np.round(rgb_f * ratio[..., None]), 0, 255).astype(np.uint8)
    return rgb_out

if __name__ == "__main__":
  from visualization import run_4_1, run_4_2, run_4_3, run_4_4
  root = Path(__file__).resolve().parents[2]
  run_4_1(root / "images/p4", root / "output/p4")
  run_4_2(root / "images/p4", root / "output/p4")
  run_4_3(root / "images/p4", root / "output/p4")
  run_4_4(root / "images/p4", root / "output/p4")