"""P1 starter. Implement every function marked TODO.

Rules: NumPy array arithmetic only. scipy/cv2/skimage may be used to CHECK
your answers, never to produce them. numpy.fft is allowed in conv2d_fft.
"""
import numpy as np

from scipy.signal import convolve2d, correlate2d
from pathlib import Path
from PIL import Image


def kernel_bank(k=15):
    """Provided. Do not modify -- your rank table must match these kernels."""
    ax = np.arange(k) - (k - 1) / 2
    box = np.ones((k, k)) / (k * k)
    s = k / 6.0
    g1 = np.exp(-(ax**2) / (2 * s * s)); g1 /= g1.sum()
    gauss = np.outer(g1, g1)
    sobel = np.outer([1, 2, 1], [-1, 0, 1]).astype(float)
    xx, yy = np.meshgrid(ax, ax); r2 = xx**2 + yy**2
    log = (r2 - 2*s*s) / (s**4) * np.exp(-r2 / (2*s*s))
    motion0 = np.zeros((k, k)); motion0[k // 2, :] = 1.0 / k
    disk = (r2 <= (k/2.0)**2).astype(float); disk /= disk.sum()
    rand = np.random.default_rng(0).normal(size=(k, k)); rand /= np.abs(rand).sum()
    return {"box": box, "gaussian": gauss, "sobel3": sobel, "log": log,
            "log_dc_removed": log - log.mean(), "motion_0deg": motion0,
            "motion_45deg": np.eye(k) / k, "disk": disk, "random": rand}


def numeric_rank(K, tol=1e-10):
    s = np.linalg.svd(K, compute_uv=False)

    return int(np.sum(s > tol)) # sum of boolean value (0, 1)


def conv2d_loops(img, K):
    """TODO 1.1: four nested Python loops. Zero-padded, 'same', TRUE convolution.
    Only run this on small crops -- see the handout."""
    H, W = img.shape
    kh, kw = K.shape

    pad_h = kh//2
    pad_w = kw//2


    img_pad = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    k_rotated = np.flip(K)

    out = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            for u in range(kh):
                for v in range(kw):
                    out[i, j] += img_pad[i+u, j+v] * k_rotated[u, v]

    return out


def conv2d_taps(img, K):
    H, W = img.shape
    kh, kw = K.shape

    pad_h = kh//2
    pad_w = kw//2


    img_pad = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    k_rotated = np.flip(K)

    out = np.zeros((H, W), dtype=np.float64)

    for u in range(kh):
        for v in range(kw):
            slice = img_pad[u:u+H, v:v+W]

            out += slice * k_rotated[u,v]

    return out


def conv2d_im2col(img, K):
    """TODO 1.1: build the (H*W, kh*kw) patch matrix, then one matmul.
    numpy.lib.stride_tricks.sliding_window_view is allowed. Report the peak
    memory and be ready to explain which step actually costs it."""
    H, W = img.shape
    kh, kw = K.shape

    pad_h = kh//2
    pad_w = kw//2


    img_pad = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    k_rotated = np.flip(K)

    view = np.lib.stride_tricks.sliding_window_view(img_pad, (kh, kw))

    view = view.reshape(H * W, kh * kw)
    
    return (view @ k_rotated.ravel()).reshape(H, W)


def conv2d_fft(img, K):
    H, W = img.shape
    kh, kw = K.shape

    buff_x = H + kh - 1
    buff_y = W + kw - 1

    tr_image = np.fft.fft2(img, (buff_x, buff_y))
    tr_k = np.fft.fft2(K, (buff_x, buff_y))

    out = np.fft.ifft2(tr_image*tr_k).real

    out = out[kh//2 : kh//2 + H, kw//2 : kw//2 + W]
    return out


def conv2d_separable(img, K, tol=1e-10):
    if(numeric_rank(K, tol) > 1):
        raise Exception("Only 1 rank allowed!")

    u, sigma, vt = np.linalg.svd(K)
    u_vec = u[:, 0] * sigma[0]
    v_vec = vt[0, :]

    u_f = np.flip(u_vec)
    v_f = np.flip(v_vec)

    H, W = img.shape
    k = len(u_vec)
    pad = k // 2

    img_pad_v = np.pad(img, ((pad, pad), (0, 0)), mode='constant', constant_values=0)
    inter = np.zeros((H, W), dtype=np.float64)
    for i in range(k):
        inter += u_f[i] * img_pad_v[i : i + H, :]

    inter_pad_h = np.pad(inter, ((0, 0), (pad, pad)), mode='constant', constant_values=0)
    out = np.zeros((H, W), dtype=np.float64)
    for j in range(k):
        out += v_f[j] * inter_pad_h[:, j : j + W]

    return out

def conv2d_lowrank(img, K, r):

    u, sigma, vt = np.linalg.svd(K)
    out = np.zeros_like(img, dtype=np.float64)

    for i in range(r):

        check = sigma[i] * np.outer(u[ : ,i], vt[i])
        out += conv2d_separable(img, check)

    return out

def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)




if __name__ == "__main__":
    from visualization import run_1_1, run_1_2, run_1_3, run_1_4
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)

    run_1_1(img)
    run_1_2("./output/p1/")
    run_1_3(img, "./output/p1/")
    run_1_4(img, "./output/p1/" )
