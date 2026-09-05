"""P2 starter. Implement every function marked TODO."""
import numpy as np
from pathlib import Path


def bit_planes(img):
    """TODO 2.1: (H,W) uint8 -> (8,H,W) uint8 in {0,1}, index 0 = LSB."""
    H, W = img.shape
    out = np.empty((8, H, W), dtype=np.uint8)
    for i in range(8):
        out[i] = (img >> i) & 1
    return out


def reconstruct(planes, keep):
    pl, H, W = planes.shape

    out = np.zeros((H,W), dtype=np.uint8)

    for i in range(1,keep+1):
        out += planes[-i]*(1 << (8-i))

    return out

def gray_encode(img):
    return img^(img >> 1)


def embed_lsb(cover, bits, plane=0):
    out = cover.copy()
    flat = out.ravel()
    n = len(bits)
    
    mask = np.uint8(~(1 << plane))
    
    flat[:n] = (flat[:n] & mask) | ((bits.astype(np.uint8) & 1) << plane)
    return out


def extract_lsb(stego, n, plane=0):
    flat = stego.ravel()
    return ((flat[:n] >> plane)&1).astype(np.uint8)


def embed_robust(cover, bits, **kw):
    """TODO 2.3: your design. Must carry 128 bits at PSNR(cover,stego) >= 40 dB
    and survive the channels below. Document your parameters."""
    raise NotImplementedError


def extract_robust(stego, cover, n, **kw):
    """TODO 2.3: matching extractor."""
    raise NotImplementedError


# ---- provided channels: do not modify, these are what you are graded against
def degrade_gaussian(img, sigma, rng=None):
    rng = rng or np.random.default_rng(0)
    return np.clip(np.round(img.astype(np.float64) + rng.normal(0, sigma, img.shape)),
                   0, 255).astype(np.uint8)


def degrade_jpeg(img, quality):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


def ber(a, b):
    return float(np.mean(a != b))


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


if __name__ == '__main__':
  from visualization import run_2_1
  root = Path(__file__).resolve().parents[2]
  cover_path = root / 'images/p2/cover_textured.png'
  out_dir = root / 'output/p2'
  run_2_1(cover_path, out_dir)