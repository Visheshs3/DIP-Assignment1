"""P5 starter. Implement every function marked TODO."""
import numpy as np


def to_ycbcr(rgb):
    """TODO 5.1: BT.601 conversion."""
    rgb_f = rgb.astype(np.float64)

    M = np.array([
        [0.299000, 0.587000, 0.114000],
        [-0.168736, -0.331264, 0.500000],
        [0.500000, -0.418688, -0.081312],
    ])
    offset = np.array([0.0, 128.0, 128.0])

    ycbcr = np.dot(rgb_f, M.T) + offset
    return np.clip(np.round(ycbcr), 0, 255).astype(np.uint8)


def key_naive_rgb(rgb, thresh=60):
    """TODO 5.1: green dominance in raw RGB. Expected to fail on the unevenly
    lit backing -- show where."""
    rgb_f = rgb.astype(np.float64)
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]

    # Green excess over the maximum of red and blue
    excess = g - np.maximum(r, b)

    alpha = np.where(excess > thresh, 0.0, 1.0)
    return alpha.astype(np.float64)


def estimate_key_colour(rgb, border=24):
    """TODO 5.2: estimate the backing chroma from the frame. Do not hard-code
    a green value; the graded footage varies."""
    ycbcr = to_ycbcr(rgb).astype(np.float64)

    left_cb = ycbcr[:, :border, 1].ravel()
    left_cr = ycbcr[:, :border, 2].ravel()
    right_cb = ycbcr[:, -border:, 1].ravel()
    right_cr = ycbcr[:, -border:, 2].ravel()
    top_cb = ycbcr[:border, :, 1].ravel()
    top_cr = ycbcr[:border, :, 2].ravel()

    cb_samples = np.concatenate([left_cb, right_cb, top_cb])
    cr_samples = np.concatenate([left_cr, right_cr, top_cr])

    key_cb = float(np.median(cb_samples))
    key_cr = float(np.median(cr_samples))
    return np.array([key_cb, key_cr], dtype=np.float64)

def key_soft(rgb, key_cbcr=None, t_in=None, t_out=None):
    """TODO 5.2: fractional alpha from chroma distance. Choose t_in and t_out
    from your own data -- look at the distance distribution for pure backing
    versus pure foreground before you pick numbers."""

    if key_cbcr is None:
        key_cbcr = estimate_key_colour(rgb)

    if t_in is None:
        t_in = 20.0
    if t_out is None:
        t_out = 50.0

    ycbcr = to_ycbcr(rgb).astype(np.float64)
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]

    # Euclidean chroma distance to backing color
    dist = np.hypot(cb - key_cbcr[0], cr - key_cbcr[1])

    # Smooth continuous alpha ramp
    alpha = np.clip((dist - t_in) / (t_out - t_in), 0.0, 1.0)
    return alpha.astype(np.float64)

def suppress_spill(rgb, alpha, strength=1.0):
    """TODO 5.2: remove green bounce. Should scale with (1 - alpha). Why?"""
    rgb_f = rgb.astype(np.float64)
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]

    # Maximum allowed green without color tinting
    g_limit = 0.5 * (r + b)
    green_excess = np.maximum(0.0, g - g_limit)

    # Spill reduction weighted by backing transparency (1 - alpha)
    spill_factor = strength * (1.0 - alpha)
    g_clean = g - spill_factor * green_excess

    rgb_clean = np.stack([r, g_clean, b], axis=-1)
    return np.clip(np.round(rgb_clean), 0, 255).astype(np.uint8)


def composite(fg, bg, alpha, spill=True):
    """TODO 5.2: I = a*F + (1-a)*B."""
    H, W = fg.shape[:2]

    # Ensure background plate matches foreground dimensions
    bg_crop = bg[:H, :W].astype(np.float64)

    if spill:
        fg_proc = suppress_spill(fg, alpha).astype(np.float64)
    else:
        fg_proc = fg.astype(np.float64)

    a = alpha[..., None]
    comp = a * fg_proc + (1.0 - a) * bg_crop
    return np.clip(np.round(comp), 0, 255).astype(np.uint8)


class ChromaLUT:
    """TODO 5.3: quantise (Cb,Cr) so keying is one table lookup per pixel."""

    def __init__(self, key_cbcr, t_in=None, t_out=None, bins=128):
        self.bins = int(bins)
        self.key_cbcr = np.asarray(key_cbcr, dtype=np.float64)
        self.t_in = 20.0 if t_in is None else float(t_in)
        self.t_out = 50.0 if t_out is None else float(t_out)

        bin_centers = (np.arange(self.bins, dtype=np.float64) + 0.5) * ( 256.0 / self.bins)

        cb_grid, cr_grid = np.meshgrid(bin_centers, bin_centers, indexing="ij")

        dist_grid = np.hypot(cb_grid - self.key_cbcr[0], cr_grid - self.key_cbcr[1])

        self.table = np.clip((dist_grid - self.t_in) / (self.t_out - self.t_in), 0.0, 1.0).astype(np.float64)

    def alpha(self, rgb):
        ycbcr = to_ycbcr(rgb)
        cb = ycbcr[..., 1]
        cr = ycbcr[..., 2]

        if self.bins == 256:
            cb_idx = cb
            cr_idx = cr
        else:
            scale = self.bins / 256.0
            cb_idx = np.clip(
                (cb.astype(np.float64) * scale).astype(np.intp), 0, self.bins - 1
            )
            cr_idx = np.clip(
                (cr.astype(np.float64) * scale).astype(np.intp), 0, self.bins - 1
            )

        return self.table[cb_idx, cr_idx]


# ---- provided metrics: do not modify
def sad(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).sum() / 1000.0)


def mse_alpha(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))


def grad_error(a, b):
    from scipy.ndimage import gaussian_filter
    ga = np.hypot(*np.gradient(gaussian_filter(a.astype(np.float64), 1.0)))
    gb = np.hypot(*np.gradient(gaussian_filter(b.astype(np.float64), 1.0)))
    return float(np.abs(ga - gb).sum() / 1000.0)


def temporal_flicker(alphas):
    if len(alphas) < 2:
        return 0.0
    total = 0.0
    count = 0
    prev = np.asarray(alphas[0], dtype=np.float32)
    for alpha in alphas[1:]:
        cur = np.asarray(alpha, dtype=np.float32)
        total += float(np.abs(cur - prev).sum(dtype=np.float64))
        count += cur.size
        prev = cur
    return total / count

if __name__ == "__main__":
  from visualization import run_5_1, run_5_2, run_5_3
  run_5_1()
  run_5_2()
  run_5_3()
