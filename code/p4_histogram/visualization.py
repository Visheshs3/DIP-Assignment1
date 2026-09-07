from pathlib import Path
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from starter import cdf, equalise, hist, wasserstein1, specify, apply_on_luma, ahe, clahe, auto_correct


def run_4_1(p4_dir, out_dir):
  p4_dir = Path(p4_dir)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  in_path = p4_dir / "equalization/input.png"
  img_in = np.asarray(Image.open(in_path).convert("L")).astype(np.uint8)

  img_eq = equalise(img_in)

  h_in = hist(img_in)
  h_eq = hist(img_eq)
  F_in = cdf(h_in)
  F_eq = cdf(h_eq)

  Image.fromarray(img_eq).save(out_dir / "equalized_output.png")

  fig, axes = plt.subplots(3, 2, figsize=(10, 11))

  axes[0, 0].imshow(img_in, cmap="gray", vmin=0, vmax=255)
  axes[0, 0].set_title("Input Image", fontweight="bold")
  axes[0, 0].axis("off")

  axes[0, 1].imshow(img_eq, cmap="gray", vmin=0, vmax=255)
  axes[0, 1].set_title("Equalised Image", fontweight="bold")
  axes[0, 1].axis("off")

  v = np.arange(256)
  axes[1, 0].bar(v, h_in, width=1.0, color="steelblue")
  axes[1, 0].set_title("Input Histogram ", fontweight="bold")
  axes[1, 0].set_xlim(0, 255)
  axes[1, 0].grid(True, linestyle=":", alpha=0.5)

  axes[1, 1].bar(v, h_eq, width=1.0, color="coral")
  axes[1, 1].set_title(
      "Equalised Histogram ", fontweight="bold"
  )
  axes[1, 1].set_xlim(0, 255)
  axes[1, 1].grid(True, linestyle=":", alpha=0.5)

  axes[2, 0].plot(v, F_in, color="steelblue", lw=2)
  axes[2, 0].set_title("Input CDF ", fontweight="bold")
  axes[2, 0].set_xlim(0, 255)
  axes[2, 0].set_ylim(0, 1.05)
  axes[2, 0].grid(True, linestyle=":", alpha=0.5)

  axes[2, 1].plot(v, F_eq, color="coral", lw=2)
  axes[2, 1].plot([0, 255], [0, 1], "k--", alpha=0.4, label="Ideal Uniform CDF")
  axes[2, 1].set_title("Equalised CDF ", fontweight="bold")
  axes[2, 1].set_xlim(0, 255)
  axes[2, 1].set_ylim(0, 1.05)
  axes[2, 1].legend(loc="upper left")
  axes[2, 1].grid(True, linestyle=":", alpha=0.5)

  plt.tight_layout()
  plot_path = out_dir / "equalization_comparison.png"
  plt.savefig(plot_path, dpi=300, bbox_inches="tight")
  plt.close()



def circular_hue_distance(h1, h2):
  """Computes shortest angular distance between two hue angles in [0, 1].

  Output is in degrees [0, 180].
  """
  diff = np.abs(h1 - h2)
  # Wrap around the circle
  circ_diff = np.minimum(diff, 1.0 - diff)
  return circ_diff * 360.0


def run_4_2(p4_dir, out_dir):
  p4_dir = Path(p4_dir)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)


  print("--- Problem 4.2(a): Image Matching ---")

  src_path = p4_dir / "matching/source.png"
  ref_path = p4_dir / "matching/reference.png"
  img_src = np.asarray(Image.open(src_path).convert("L")).astype(np.uint8)
  img_ref = np.asarray(Image.open(ref_path).convert("L")).astype(np.uint8)

  h_src = hist(img_src)
  h_ref = hist(img_ref)
  img_matched = specify(img_src, h_ref)
  h_matched = hist(img_matched)

  w1_before = wasserstein1(h_src, h_ref)
  w1_after = wasserstein1(h_matched, h_ref)

  print(f"Wasserstein-1 Distance to Reference:")
  print(f"  Before Matching: {w1_before:.4f} levels")
  print(f"  After Matching : {w1_after:.4f} levels")

  fig, axes = plt.subplots(3, 3, figsize=(12, 10))
  items = [
      ("Source", img_src, h_src, "steelblue"),
      ("Reference", img_ref, h_ref, "forestgreen"),
      ("Matched Result", img_matched, h_matched, "crimson"),
  ]
  v = np.arange(256)
  for col, (name, im, h, colr) in enumerate(items):
    axes[0, col].imshow(im, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(name, fontweight="bold")
    axes[0, col].axis("off")

    axes[1, col].bar(v, h, width=1.0, color=colr)
    axes[1, col].set_title(f"{name} Histogram", fontsize=10)
    axes[1, col].set_xlim(0, 255)
    axes[1, col].grid(True, linestyle=":", alpha=0.5)

    axes[2, col].plot(v, cdf(h), color=colr, lw=2)
    axes[2, col].set_title(f"{name} CDF", fontsize=10)
    axes[2, col].set_xlim(0, 255)
    axes[2, col].set_ylim(0, 1.05)
    axes[2, col].grid(True, linestyle=":", alpha=0.5)

  plt.tight_layout()
  plt.savefig(out_dir / "matching_4_2a.png", dpi=300, bbox_inches="tight")
  plt.close()

  print("Problem 4.2(b): Histogram Specification ")

  spec_src_path = p4_dir / "specification/source.png"
  img_spec_src = np.asarray(Image.open(spec_src_path).convert("L")).astype(np.uint8)
  h_spec_src = hist(img_spec_src)

  # Target 1: Uniform
  h_unif = np.ones(256, dtype=np.float64) / 256.0
  img_unif = specify(img_spec_src, h_unif)
  h_unif_res = hist(img_unif)

  # Target 2: Gaussian (mu = 128, sigma = 35)
  mu, sigma = 128.0, 35.0
  gauss = np.exp(-0.5 * ((v - mu) / sigma) ** 2)
  h_gauss = gauss / np.sum(gauss)
  img_gauss = specify(img_spec_src, h_gauss)
  h_gauss_res = hist(img_gauss)

  # Measure W1
  w1_unif_before = wasserstein1(h_spec_src, h_unif)
  w1_unif_after = wasserstein1(h_unif_res, h_unif)
  w1_gauss_before = wasserstein1(h_spec_src, h_gauss)
  w1_gauss_after = wasserstein1(h_gauss_res, h_gauss)

  print(f"{'Target':12s} | {'W1 Before (levels)':20s} | {'W1 After (levels)':20s}")
  print(f"{'Uniform':12s} | {w1_unif_before:<20.4f} | {w1_unif_after:<20.4f}")
  print(f"{'Gaussian':12s} | {w1_gauss_before:<20.4f} | {w1_gauss_after:<20.4f}")

  # Deliverable 4.2(b) Plot
  fig, axes = plt.subplots(3, 3, figsize=(12, 10))
  spec_items = [
      ("Original Source", img_spec_src, h_spec_src, "steelblue"),
      ("Uniform Specified", img_unif, h_unif_res, "coral"),
      ("Gaussian Specified", img_gauss, h_gauss_res, "mediumpurple"),
  ]
  for col, (name, im, h, colr) in enumerate(spec_items):
    axes[0, col].imshow(im, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(name, fontweight="bold")
    axes[0, col].axis("off")

    axes[1, col].bar(v, h, width=1.0, color=colr)
    axes[1, col].set_title(f"{name} Histogram", fontsize=10)
    axes[1, col].set_xlim(0, 255)
    axes[1, col].grid(True, linestyle=":", alpha=0.5)

    axes[2, col].plot(v, cdf(h), color=colr, lw=2, label="Achieved CDF")
    if col == 1:
      axes[2, col].plot(v, cdf(h_unif), "k--", alpha=0.5, label="Target CDF")
      axes[2, col].legend()
    elif col == 2:
      axes[2, col].plot(v, cdf(h_gauss), "k--", alpha=0.5, label="Target CDF")
      axes[2, col].legend()
    axes[2, col].set_title(f"{name} CDF", fontsize=10)
    axes[2, col].set_xlim(0, 255)
    axes[2, col].set_ylim(0, 1.05)
    axes[2, col].grid(True, linestyle=":", alpha=0.5)

  plt.tight_layout()
  plt.savefig(out_dir / "specification_4_2b.png", dpi=300, bbox_inches="tight")
  plt.close()

  print("--- Problem 4.2(c): Colour Equalisation & Hue Shift ---")

  col_src_path = p4_dir / "colour_source.png"
  img_col = np.asarray(Image.open(col_src_path).convert("RGB")).astype(np.uint8)

  # Per-channel RGB equalisation
  img_rgb_eq = np.zeros_like(img_col)
  for ch in range(3):
    img_rgb_eq[..., ch] = equalise(img_col[..., ch])

  # Luminance-only equalisation
  img_luma_eq = apply_on_luma(img_col, equalise)

  # Convert all to HSV for hue analysis (Hue in [0, 1], Saturation in [0, 1])
  hsv_orig = mcolors.rgb_to_hsv(img_col / 255.0)
  hsv_rgb = mcolors.rgb_to_hsv(img_rgb_eq / 255.0)
  hsv_luma = mcolors.rgb_to_hsv(img_luma_eq / 255.0)

  # Exclude near-achromatic pixels (S < 0.10) where hue is undefined/unstable
  SAT_THRESH = 0.10
  valid_mask = hsv_orig[..., 1] >= SAT_THRESH

  # Measure Circular Hue Distances (in degrees)
  hue_diff_rgb = circular_hue_distance(hsv_orig[..., 0], hsv_rgb[..., 0])
  hue_diff_luma = circular_hue_distance(hsv_orig[..., 0], hsv_luma[..., 0])

  mean_shift_rgb = np.mean(hue_diff_rgb[valid_mask])
  mean_shift_luma = np.mean(hue_diff_luma[valid_mask])

  print(f"Hue Shift Analysis (HSV space, chromatic pixels with S >= {SAT_THRESH}):")
  print(f"  Per-Channel RGB Equalisation Mean Hue Shift : {mean_shift_rgb:.2f}°")
  print(f"  Luminance-Only Equalisation Mean Hue Shift  : {mean_shift_luma:.2f}°")

  fig, axes = plt.subplots(1, 3, figsize=(14, 5))
  axes[0].imshow(img_col)
  axes[0].set_title("Original Colour Image", fontweight="bold")
  axes[0].axis("off")

  axes[1].imshow(img_rgb_eq)
  axes[1].set_title(f"Per-Channel RGB (Shift: {mean_shift_rgb:.1f}°)", fontweight="bold")
  axes[1].axis("off")

  axes[2].imshow(img_luma_eq)
  axes[2].set_title(f"Luma-Only Equalised (Shift: {mean_shift_luma:.1f}°)", fontweight="bold")
  axes[2].axis("off")

  plt.tight_layout()
  plt.savefig(out_dir / "colour_4_2c.png", dpi=300, bbox_inches="tight")
  plt.close()


def run_4_3(p4_dir, out_dir):
  p4_dir = Path(p4_dir)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # Load image and convert to BT.601 Luminance Y = round(0.299R + 0.587G + 0.114B)
  img_rgb = np.asarray(Image.open(p4_dir / "local_regions.png").convert("RGB")).astype(np.float64)
  Y = np.clip(np.round(0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]), 0, 255).astype(np.uint8)
  H, W = Y.shape

  print("Running 4.3(a): 5 Local Patch Histograms...")

  patch_size = 64
  patch_coords = [
      (int(H * 0.15), int(W * 0.20)),  
      (int(H * 0.25), int(W * 0.70)),  
      (int(H * 0.50), int(W * 0.50)),  
      (int(H * 0.80), int(W * 0.25)),  
      (int(H * 0.80), int(W * 0.75)),  
  ]

  fig, axes = plt.subplots(2, 5, figsize=(15, 6))
  v = np.arange(256)

  for idx, (r, c) in enumerate(patch_coords):
    p = Y[r : r + patch_size, c : c + patch_size]
    h = hist(p)

    axes[0, idx].imshow(p, cmap="gray", vmin=0, vmax=255)
    axes[0, idx].set_title(f"Patch {idx+1} at ({r},{c})", fontsize=10)
    axes[0, idx].axis("off")

    axes[1, idx].bar(v, h, width=1.0, color="teal")
    axes[1, idx].set_title(f"Patch {idx+1} Hist", fontsize=10)
    axes[1, idx].set_xlim(0, 255)
    axes[1, idx].grid(True, linestyle=":", alpha=0.5)

  plt.tight_layout()
  plt.savefig(out_dir / "patches_4_3a.png", dpi=300, bbox_inches="tight")
  plt.close()

  print("Running 4.3(b) & (c): Plain AHE vs CLAHE")
  img_ahe = ahe(Y, tiles=8)
  img_clahe = clahe(Y, tiles=8, clip=3.0)

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  axes[0].imshow(Y, cmap="gray", vmin=0, vmax=255)
  axes[0].set_title("Original Luminance $Y$", fontweight="bold")
  axes[0].axis("off")

  axes[1].imshow(img_ahe, cmap="gray", vmin=0, vmax=255)
  axes[1].set_title("Plain Tiled AHE (8x8)", fontweight="bold")
  axes[1].axis("off")

  axes[2].imshow(img_clahe, cmap="gray", vmin=0, vmax=255)
  axes[2].set_title("CLAHE (8x8, Clip=3.0)", fontweight="bold")
  axes[2].axis("off")

  plt.tight_layout()
  plt.savefig(out_dir / "ahe_vs_clahe_4_3bc.png", dpi=300, bbox_inches="tight")
  plt.close()

  print("Running 4.3(d): Parameter Ablation Grid...")
  clips = [1.0, 2.0, 3.0, 10.0, np.inf]
  tiles_list = [2, 4, 8, 16, 64]

  fig, axes = plt.subplots(len(clips), len(tiles_list), figsize=(16, 16), squeeze=False)

  for row_idx, cl in enumerate(clips):
    for col_idx, t in enumerate(tiles_list):
      res = clahe(Y, tiles=t, clip=cl)
      ax = axes[row_idx, col_idx]
      ax.imshow(res, cmap="gray", vmin=0, vmax=255)
      cl_str = r"$\infty$" if cl == np.inf else f"{cl:.1f}"
      ax.set_title(f"Tiles={t}, Clip={cl_str}", fontsize=9)
      ax.axis("off")

  plt.tight_layout()
  plt.savefig(out_dir / "ablation_grid_4_3d.png", dpi=300, bbox_inches="tight")
  plt.close()

  print("Running 4.3(e): Local Histogram Specification...")
  mu, sigma = 128.0, 45.0
  gauss = np.exp(-0.5 * ((v - mu) / sigma) ** 2)
  h_gauss_target = gauss / np.sum(gauss)

  # Custom local specification using the Gaussian target on 8x8 tiles
  tiles = 8
  y_edges = np.linspace(0, H, tiles + 1, dtype=int)
  x_edges = np.linspace(0, W, tiles + 1, dtype=int)
  luts_spec = np.zeros((tiles, tiles, 256), dtype=np.float64)

  F_tgt = cdf(h_gauss_target)
  for i in range(tiles):
    for j in range(tiles):
      r0, r1 = y_edges[i], y_edges[i + 1]
      c0, c1 = x_edges[j], x_edges[j + 1]
      tile = Y[r0:r1, c0:c1]
      F_src = cdf(hist(tile))
      diff = np.abs(F_tgt[:, None] - F_src[None, :])
      luts_spec[i, j] = np.argmin(diff, axis=0)

  # Bilinear blend of local specification LUTs
  y_centers = 0.5 * (y_edges[:-1] + y_edges[1:] - 1)
  x_centers = 0.5 * (x_edges[:-1] + x_edges[1:] - 1)
  dy = y_centers[1] - y_centers[0]
  dx = x_centers[1] - x_centers[0]
  v_c = np.clip((np.arange(H) - y_centers[0]) / dy, 0.0, tiles - 1)
  u_c = np.clip((np.arange(W) - x_centers[0]) / dx, 0.0, tiles - 1)
  i0 = np.clip(np.floor(v_c).astype(int), 0, tiles - 2)
  i1 = i0 + 1
  j0 = np.clip(np.floor(u_c).astype(int), 0, tiles - 2)
  j1 = j0 + 1
  beta = (v_c - i0)[:, None]
  alpha = (u_c - j0)[None, :]

  out_spec = (
      (1 - beta) * (1 - alpha) * luts_spec[i0[:, None], j0[None, :], Y]
      + (1 - beta) * alpha * luts_spec[i0[:, None], j1[None, :], Y]
      + beta * (1 - alpha) * luts_spec[i1[:, None], j0[None, :], Y]
      + beta * alpha * luts_spec[i1[:, None], j1[None, :], Y]
  )
  img_local_spec = np.clip(np.round(out_spec), 0, 255).astype(np.uint8)

  fig, axes = plt.subplots(1, 2, figsize=(12, 6))
  axes[0].imshow(img_clahe, cmap="gray", vmin=0, vmax=255)
  axes[0].set_title(
      "Standard CLAHE (Uniform Target)\n(Can force harsh unnatural edges)",
      fontweight="bold",
  )
  axes[0].axis("off")

  axes[1].imshow(img_local_spec, cmap="gray", vmin=0, vmax=255)
  axes[1].set_title(
      "Local Gaussian Specification\n(Preserves organic tonal rolloff)",
      fontweight="bold",
  )
  axes[1].axis("off")

  plt.tight_layout()
  plt.savefig(out_dir / "local_spec_4_3e.png", dpi=300, bbox_inches="tight")
  plt.close()

def get_luma(rgb):
  rgb_f = rgb.astype(np.float64)
  return np.clip(
      np.round(
          0.299 * rgb_f[..., 0]
          + 0.587 * rgb_f[..., 1]
          + 0.114 * rgb_f[..., 2]
      ),
      0,
      255,
  ).astype(np.uint8)


def run_4_4(p4_dir, out_dir):
  p4_dir = Path(p4_dir)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  print("--- Problem 4.4: Automatic Exposure Correction ---")

  names = [f"ev{i}" for i in range(5)]
  orig_images = []
  corr_images = []
  orig_lumas = []
  corr_lumas = []

  for name in names:
    im_path = p4_dir / f"exposure/{name}.png"
    im = np.asarray(Image.open(im_path).convert("RGB")).astype(np.uint8)
    im_corr = auto_correct(im)

    orig_images.append(im)
    corr_images.append(im_corr)
    orig_lumas.append(get_luma(im))
    corr_lumas.append(get_luma(im_corr))

  pairs = list(itertools.combinations(range(5), 2))
  w1_before = []
  w1_after = []

  print(f"{'Pair':12s} | {'W1 Before (levels)':20s} | {'W1 After (levels)':20s}")

  for i, j in pairs:
    pair_name = f"ev{i} <-> ev{j}"
    d_before = wasserstein1(hist(orig_lumas[i]), hist(orig_lumas[j]))
    d_after = wasserstein1(hist(corr_lumas[i]), hist(corr_lumas[j]))
    w1_before.append(d_before)
    w1_after.append(d_after)
    print(f"{pair_name:12s} | {d_before:<20.4f} | {d_after:<20.4f}")

  print(f"Mean Pairwise W1 : {np.mean(w1_before):.4f} levels -> {np.mean(w1_after):.4f} levels")
  print(f"Max  Pairwise W1 : {np.max(w1_before):.4f} levels -> {np.max(w1_after):.4f} levels")

  print("Output Luminance Quality Metrics (Anti-Degeneracy Check):")
  print(f"{'Image':10s} | {'Std Dev (sigma)':18s} | {'1st-99th Range (levels)':25s}")

  for idx, name in enumerate(names):
    Y_c = corr_lumas[idx]
    std_val = np.std(Y_c)
    p_range = np.percentile(Y_c, 99.0) - np.percentile(Y_c, 1.0)
    print(f"{name:10s} | {std_val:<18.2f} | {p_range:<25.2f}")

  fig, axes = plt.subplots(4, 5, figsize=(18, 12))
  v = np.arange(256)

  for col in range(5):
    # Row 0: Original image
    axes[0, col].imshow(orig_images[col])
    axes[0, col].set_title(f"Original {names[col]}", fontweight="bold")
    axes[0, col].axis("off")

    # Row 1: Original luma histogram
    axes[1, col].bar(v, hist(orig_lumas[col]), width=1.0, color="steelblue")
    axes[1, col].set_xlim(0, 255)
    axes[1, col].set_title(f"{names[col]} Hist (Raw)", fontsize=9)
    axes[1, col].grid(True, linestyle=":", alpha=0.5)

    # Row 2: Corrected image
    axes[2, col].imshow(corr_images[col])
    axes[2, col].set_title(f"Corrected {names[col]}", fontweight="bold")
    axes[2, col].axis("off")

    # Row 3: Corrected luma histogram
    axes[3, col].bar(v, hist(corr_lumas[col]), width=1.0, color="coral")
    axes[3, col].set_xlim(0, 255)
    axes[3, col].set_title(f"{names[col]} Hist (Aligned)", fontsize=9)
    axes[3, col].grid(True, linestyle=":", alpha=0.5)

  plt.tight_layout()
  save_path = out_dir / "exposure_correction_4_4.png"
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()