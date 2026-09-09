from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from starter import key_naive_rgb, to_ycbcr, estimate_key_colour, composite, key_soft, suppress_spill, sad, grad_error,ChromaLUT, mse_alpha
import time
import imageio

def run_5_1():
  root = Path(__file__).resolve().parents[2]
  frame_path = root / "images/p5/frames/f000.png"
  out_dir = root / "output/p5"
  out_dir.mkdir(parents=True, exist_ok=True)

  img = np.asarray(Image.open(frame_path).convert("RGB"))
  H, W, _ = img.shape

  alpha_rgb = key_naive_rgb(img, thresh=60)

  ycbcr = to_ycbcr(img).astype(np.float64)
  
  key_cb, key_cr = estimate_key_colour(img)

  # Euclidean distance in (Cb, Cr)
  dist_chroma = np.hypot(ycbcr[..., 1] - key_cb, ycbcr[..., 2] - key_cr)

  alpha_chroma = np.where(dist_chroma < 50.0, 0.0, 1.0).astype(np.float64)

  # We look for pixels where Chroma says BACKING (0.0) but RGB says FOREGROUND (1.0)
  disagreements = (alpha_rgb == 1.0) & (alpha_chroma == 0.0)
  fail_ys, fail_xs = np.where(disagreements)

  lum_failures = ycbcr[fail_ys, fail_xs, 0]
  idx = np.argmin(lum_failures)
  py, px = fail_ys[idx], fail_xs[idx]


  r_val, g_val, b_val = img[py, px]
  excess_val = int(g_val) - max(int(r_val), int(b_val))
  y_val, cb_val, cr_val = ycbcr[py, px]

  print(f"FAILED PIXEL IDENTIFIED AT: Row (Y) = {py}, Col (X) = {px}")

  print(f"Raw RGB Values       : R={r_val}, G={g_val}, B={b_val}")

  print(f"Green Excess Metric  : G - max(R, B) = {excess_val} (Threshold is 60)")
  print(f"RGB Key Verdict      : alpha = {alpha_rgb[py, px]} (MISTAKEN FOREGROUND)")
  print(f"YCbCr Values         : Y={y_val:.1f}, Cb={cb_val:.1f}, Cr={cr_val:.1f}")
  print(f"Chroma Distance      : {dist_chroma[py, px]:.2f} (from Key Cb={key_cb:.1f}, Cr={key_cr:.1f})")
  print(f"Chroma Key Verdict   : alpha = {alpha_chroma[py, px]} (CORRECT BACKING)")


  fig, axes = plt.subplots(1, 4, figsize=(20, 5))

  axes[0].imshow(img)
  axes[0].set_title(f"Source", fontweight="bold")
  axes[0].axis("off")

  axes[1].imshow(alpha_rgb, cmap="gray", vmin=0, vmax=1)
  axes[1].set_title("Naive RGB Matte\n(Green Excess > 60)", fontweight="bold")
  axes[1].axis("off")

  axes[2].imshow(alpha_chroma, cmap="gray", vmin=0, vmax=1)
  axes[2].set_title("Chroma-Only Matte\n(YCbCr Distance)", fontweight="bold")
  axes[2].axis("off")

  axes[3].imshow(disagreements, cmap="magma")
  axes[3].set_title("RGB Failure Regions\n(False Foregrounds)", fontweight="bold")
  axes[3].axis("off")

  plt.tight_layout()
  save_path = out_dir / "p5_1_naive_failure.png"
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  

def run_5_2():
  root = Path(__file__).resolve().parents[2]
  frame_path = root / "images/p5/frames/f000.png"
  out_dir = root / "output/p5"
  out_dir.mkdir(parents=True, exist_ok=True)

  img = np.asarray(Image.open(frame_path).convert("RGB"))
  H, W, _ = img.shape

  bg_path = root / "images/p5/bg.jpg"
  
  bg_full = np.asarray(Image.open(bg_path).convert("RGB"))
  bg = bg_full[:H, :W]


  key_cbcr = estimate_key_colour(img, border=24)

  ycbcr = to_ycbcr(img).astype(np.float64)
  dist_chroma = np.hypot(ycbcr[..., 1] - key_cbcr[0], ycbcr[..., 2] - key_cbcr[1])

  all_dist = dist_chroma.ravel()

  # 3. Plot distribution of all pixels
  fig, ax = plt.subplots(figsize=(9, 4.5))
  bins = np.linspace(0, 120, 150)
  
  ax.hist(
      all_dist,
      bins=bins,
      color="steelblue",
      edgecolor="black",
      linewidth=0.5,
      label="All Pixels in Frame",
  )
  
  # Threshold indicators
  t_in = 15.0
  t_out = 80.0
  ax.axvline(
      t_in,
      color="black",
      linestyle="--",
      linewidth=1.8,
      label=f"$t_{{in}} = {t_in}$ (Backing end)",
  )
  ax.axvline(
      t_out,
      color="crimson",
      linestyle="--",
      linewidth=1.8,
      label=f"$t_{{out}} = {t_out}$ (Foreground start)",
  )
  
  # Log scale prevents the zero-distance backing peak from flattening the valley
  ax.set_yscale("log")
  ax.set_xlim(0, 120)
  ax.set_title("Chroma Distance Distribution across All Pixels", fontweight="bold")
  ax.set_xlabel("Euclidean Chroma Distance ($d$) from Key Color")
  ax.set_ylabel("Pixel Count (Log Scale)")
  ax.legend(loc="upper right")
  ax.grid(True, which="both", linestyle=":", alpha=0.5)
  
  plt.tight_layout()
  save_path = out_dir / "p5_2a_chroma_distributions.png"
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()

  print("\nContinuous Alpha Matte")

  alpha_soft = key_soft(img, key_cbcr=key_cbcr, t_in=t_in, t_out=t_out)

  # Pixels strictly between 0.05 and 0.95 represent fractional transparency
  frac_mask = (alpha_soft > 0.05) & (alpha_soft < 0.95)
  frac_percentage = 100.0 * np.mean(frac_mask)
  print(f"Fractional Alpha Pixels: {frac_percentage:.2f}% ")

  comp_no_spill = composite(img, bg, alpha_soft, spill=False)

  print("\nSpill Suppression ")

  img_despilled = suppress_spill(img, alpha_soft, strength=1.0)
  comp_spill = composite(img, bg, alpha_soft, spill=True)

  def get_green_excess(rgb_arr, mask):
    rgb_f = rgb_arr.astype(np.float64)
    g_excess = np.maximum(0.0, rgb_f[..., 1] - 0.5 * (rgb_f[..., 0] + rgb_f[..., 2]))
    return float(np.mean(g_excess[mask]))

  excess_before = get_green_excess(img, frac_mask)
  excess_after = get_green_excess(img_despilled, frac_mask)
  print(f"Residual Green Excess on Fractional Pixels (Before): {excess_before:.2f}")
  print(f"Residual Green Excess on Fractional Pixels (After) : {excess_after:.2f}")
  print(f"Spill Suppression Reduction: {100.0 * (excess_before - excess_after) / max(excess_before, 1e-4):.1f}%")


  fig, axes = plt.subplots(2, 1, figsize=(12, 10))
  axes[0].imshow(comp_no_spill)
  axes[0].set_title("Composite WITHOUT De-spill", fontweight="bold")
  axes[0].axis("off")

  axes[1].imshow(comp_spill)
  axes[1].set_title(
      "Composite WITH Spill Suppression", fontweight="bold"
  )
  axes[1].axis("off")



  plt.tight_layout()
  plt.savefig(out_dir / "p5_2c_spill_comparison.png", dpi=300, bbox_inches="tight")
  plt.close()

  print("\n Matte Difference Metrics ")
  alpha_naive = key_naive_rgb(img, thresh=60)

  mad_val = float(np.mean(np.abs(alpha_soft - alpha_naive)))
  sad_val = sad(alpha_soft, alpha_naive)

  g_err = grad_error(alpha_soft, alpha_naive)

  print(f"Mean Absolute Difference (MAD) : {mad_val:.4f}")
  print(f"Starter Scaled SAD             : {sad_val:.2f}")
  print(f"Gradient Field Error           : {g_err:.2f}")

  fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  axes[0].imshow(alpha_naive, cmap="gray", vmin=0, vmax=1)
  axes[0].set_title("Naive Binary Matte ", fontweight="bold")
  axes[0].axis("off")

  axes[1].imshow(alpha_soft, cmap="gray", vmin=0, vmax=1)
  axes[1].set_title("Continuous Soft Matte ", fontweight="bold")
  axes[1].axis("off")

  axes[2].imshow(np.abs(alpha_soft - alpha_naive), cmap="inferno")
  axes[2].set_title(
      f"Absolute Difference (|Soft - Naive|)\nMAD={mad_val:.3f}, GradError={g_err:.1f}",
      fontweight="bold",
  )
  axes[2].axis("off")

  plt.tight_layout()
  plt.savefig(
      out_dir / "p5_2d_matte_comparison.png", dpi=300, bbox_inches="tight"
  )
  plt.close()


def run_5_3():
  root = Path(__file__).resolve().parents[2]
  frames_dir = root / "images/p5/frames"
  out_dir = root / "output/p5"
  out_dir.mkdir(parents=True, exist_ok=True)

  frame_files = sorted(frames_dir.glob("f*.png"))[:48]
  num_frames = len(frame_files)
  if num_frames < 48:
    print(f"Warning: Found {num_frames} frames (expected 48).")

  f0 = np.asarray(Image.open(frame_files[0]).convert("RGB"))
  H, W, _ = f0.shape

  bg_path = root / "images/p5/bg.jpg"
  bg_full = np.asarray(Image.open(bg_path).convert("RGB"))
  H_bg, W_bg, _ = bg_full.shape

  if H_bg < H:
    bg_full = np.asarray(
        Image.fromarray(bg_full).resize(
            (int(W_bg * H / H_bg), H), Image.BILINEAR
        )
    )
    H_bg, W_bg, _ = bg_full.shape

  if W_bg < int(2.5 * W):
    # Resize width to 2.5x if original asset is not wide enough
    target_w = int(2.5 * W)
    bg_full = np.asarray(
        Image.fromarray(bg_full).resize((target_w, H), Image.BILINEAR)
    )
    W_bg = target_w

  max_shift_x = W_bg - W
  print(
      f"Background Plate: {W_bg}x{H_bg} (Span: {W_bg / W:.2f}x frame width,"
      f" Panning Shift: {max_shift_x}px)"
  )

  key_cbcr = estimate_key_colour(f0, border=24)
  t_in, t_out = 15.0, 80.0
  lut = ChromaLUT(key_cbcr=key_cbcr, t_in=t_in, t_out=t_out, bins=128)

  t_io = 0.0
  t_key = 0.0
  t_spill = 0.0
  t_comp = 0.0

  raw_frames, raw_alphas, bg_crops = [], [], []

  t_start_all = time.perf_counter()

  for i, f_path in enumerate(frame_files):
    # Time Disk I/O
    t0 = time.perf_counter()
    fg = np.asarray(Image.open(f_path).convert("RGB"))
    t_io += time.perf_counter() - t0

    # Panning window calculation across the background
    shift_x = int((i / max(1, num_frames - 1)) * max_shift_x)
    bg_crop = bg_full[:H, shift_x : shift_x + W]

    # Time Vectorized Keying (LUT)
    t0 = time.perf_counter()
    alpha = lut.alpha(fg)
    t_key += time.perf_counter() - t0

    raw_frames.append(fg)
    raw_alphas.append(alpha)
    bg_crops.append(bg_crop)

  roi_static = (slice(440, 540), slice(590, 690))

  flicker_before = np.mean([
      np.mean(
          np.abs(raw_alphas[t][roi_static] - raw_alphas[t - 1][roi_static])
      )
      for t in range(1, num_frames)
  ])

  beta = 0.70
  filtered_alphas = [raw_alphas[0].copy()]
  for t in range(1, num_frames):
    a_filtered = beta * raw_alphas[t] + (1.0 - beta) * filtered_alphas[-1]
    filtered_alphas.append(a_filtered)

  flicker_after = np.mean([
      np.mean(
          np.abs(
              filtered_alphas[t][roi_static]
              - filtered_alphas[t - 1][roi_static]
          )
      )
      for t in range(1, num_frames)
  ])

  composited_frames = []
  for t in range(num_frames):
    fg = raw_frames[t]
    a = filtered_alphas[t]
    bg = bg_crops[t]

    # Time Spill Suppression
    t0 = time.perf_counter()
    fg_clean = suppress_spill(fg, a, strength=1.0)
    t_spill += time.perf_counter() - t0

    # Time Alpha Compositing
    t0 = time.perf_counter()
    comp = composite(fg_clean, bg, a, spill=False)
    t_comp += time.perf_counter() - t0

    composited_frames.append(comp)

  mp4_path = out_dir / "p5_composite.mp4"
  t0 = time.perf_counter()
  imageio.mimwrite(
      mp4_path, composited_frames, fps=24, codec="libx264", quality=8
  )
  t_encode = time.perf_counter() - t0

  total_elapsed = time.perf_counter() - t_start_all
  fps = num_frames / total_elapsed

  fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

  # Panel 1: Flicker across time in static region
  fl_per_frame_raw = [
      np.mean(
          np.abs(raw_alphas[t][roi_static] - raw_alphas[t - 1][roi_static])
      )
      for t in range(1, num_frames)
  ]
  fl_per_frame_filt = [
      np.mean(
          np.abs(
              filtered_alphas[t][roi_static]
              - filtered_alphas[t - 1][roi_static]
          )
      )
      for t in range(1, num_frames)
  ]

  axes[0].plot(
      range(1, num_frames),
      fl_per_frame_raw,
      label=f"Raw Alpha (mean={flicker_before:.4f})",
      color="crimson",
      alpha=0.7,
  )
  axes[0].plot(
      range(1, num_frames),
      fl_per_frame_filt,
      label=f"Filtered Alpha (mean={flicker_after:.4f})",
      color="forestgreen",
      linewidth=1.8,
  )
  axes[0].set_title(
      "Frame-to-Frame Flicker in Static ROI", fontweight="bold"
  )
  axes[0].set_xlabel("Frame Index")
  axes[0].set_ylabel("Mean ")
  axes[0].grid(True, linestyle=":", alpha=0.6)
  axes[0].legend()

  # Horizontal line profile across a moving edge (e.g. hair/ear, row 300)
  test_row = 300
  axes[1].plot(
      raw_alphas[20][test_row, 650:800],
      label="Frame 20: Raw Alpha",
      color="black",
      linestyle="--",
  )
  axes[1].plot(
      filtered_alphas[20][test_row, 650:800],
      label="Frame 20: Filtered Alpha",
      color="blue",
      alpha=0.7,
  )
  axes[1].plot(
      filtered_alphas[21][test_row, 650:800],
      label="Frame 21: Filtered Alpha (Shifted)",
      color="orange",
      alpha=0.7,
  )
  axes[1].set_title(
      "Edge Profile Across Moving Boundary (No Smear/Lag)", fontweight="bold"
  )
  axes[1].set_xlabel("Pixel Column (X)")
  axes[1].set_ylabel("Alpha Opacity")
  axes[1].grid(True, linestyle=":", alpha=0.6)
  axes[1].legend()

  plt.tight_layout()
  plot_path = out_dir / "p5_3_flicker_and_lag.png"
  plt.savefig(plot_path, dpi=300, bbox_inches="tight")
  plt.close()

  stages = {
      "Disk I/O (Read PNGs)": t_io,
      "Vectorized LUT Keying": t_key,
      "Spill Suppression": t_spill,
      "Alpha Blending Comp": t_comp,
      "Video Encoding (x264)": t_encode,
  }
  bottleneck = max(stages, key=stages.get)
  compute_total = sum(stages.values())

  print(f"Frames Processed       : {num_frames} frames")
  print(f"Overall Pipeline Speed : {fps:.2f} FPS ({total_elapsed:.2f}s total)")
  print("\nStage Breakdown (per frame average):")
  for name, s_time in stages.items():
    pct = 100.0 * s_time / compute_total
    print(
        f"  * {name:<22} : {(s_time/num_frames)*1000:6.2f} ms ({pct:5.1f}%)"
    )
  print(
      f"\nNamed Bottleneck       : {bottleneck} ({100.0*stages[bottleneck]/compute_total:.1f}% of total pipeline time)"
  )
  print("\nTemporal Flicker (Static ROI):")
  print(f"  * Before Filtering   : {flicker_before:.6f}")
  print(f"  * After Filtering    : {flicker_after:.6f}")
  print(
      "  * Flicker Reduction  :"
      f" {100.0 * (flicker_before - flicker_after) / flicker_before:.1f}%"
  )
  print(f"\nDeliverable MP4 Saved  : {mp4_path.resolve()}")
  print(f"Deliverable Plot Saved : {plot_path.resolve()}")