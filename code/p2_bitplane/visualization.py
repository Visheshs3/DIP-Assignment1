import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from starter import (
    bit_planes,
    gray_encode, 
    psnr, 
    reconstruct, 
    extract_lsb, 
    embed_lsb,
    embed_robust,
    extract_robust,
    degrade_gaussian,
    degrade_jpeg,
    ber
    )


def run_2_1(cover_path, out_dir):
  out_dir = Path(out_dir)
  out_dir.mkdir(exist_ok=True)

  img = np.asarray(Image.open(cover_path)).astype(np.uint8)

  planes = bit_planes(img)  # Shape: (8, 1024, 1024)

  fig, axes = plt.subplots(2, 4, figsize=(14, 7))
  for b in range(8):
    ax = axes[b // 4, b % 4] 

    ax.imshow(planes[b] * 255, cmap='gray', vmin=0, vmax=255)

    ax.set_title(f'Bit Plane {b}')
    ax.axis('off')
  plt.suptitle('Bit-Plane Slicing of Textured Cover', fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.savefig(out_dir / 'textured_bit_planes.png', dpi=300, bbox_inches='tight')
  plt.close()

  psnr_vals = []
  print('\nPSNR Reconstruction Table ')
  print(f"{'n':16s} | {'PSNR (dB)':12s} | {'Gain / Step (dB)':16s}")

  prev_p = None
  for n in range(1, 9):
    rec = reconstruct(planes, n)
    p = psnr(img, rec)
    psnr_vals.append(p)
    gain_str = (f'{p - prev_p:+.2f} dB' if (prev_p is not None and not np.isinf(p)) else '-')
    p_str = f'{p:.2f} dB' if not np.isinf(p) else 'inf'
    print(f'{n:<16d} | {p_str:<12s} | {gain_str:<16s}')
    prev_p = p

  # Measure slope across finite values (n = 1 to 7)
  n_range = np.arange(1, 8)
  finite_psnrs = np.array(psnr_vals[:7])
  slope, intercept = np.polyfit(n_range, finite_psnrs, 1)
  print(f'\nMeasured Average Gain per Plane: {slope:.2f} dB/plane')

  plt.figure(figsize=(7, 4.5))
  plt.plot(n_range, finite_psnrs, 'o-', linewidth=2, label='Measured PSNR')
  plt.plot(
      n_range,
      slope * n_range + intercept,
      '--',
      color='red',
      label=f'Linear Fit (slope = {slope:.2f} dB/plane)',
  )
  plt.title(
      'Reconstruction PSNR vs. Top n Bit Planes', fontsize=12, fontweight='bold'
  )
  plt.xlabel('Number of Top Bit Planes Kept (n)', fontsize=11)
  plt.ylabel('PSNR (dB)', fontsize=11)
  plt.xticks(range(1, 8))
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_dir / 'psnr_vs_n.png', dpi=300, bbox_inches='tight')
  plt.close()


  ramp = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
  gray_ramp = gray_encode(ramp)

  ramp_planes = bit_planes(ramp)
  gray_planes = bit_planes(gray_ramp)

  fig, axes = plt.subplots(4, 4, figsize=(14, 10))
  for b in range(8):
    # Standard Binary Ramp
    row_bin = (7 - b) // 2
    col_bin = ((7 - b) % 2) * 2
    axes[row_bin, col_bin].imshow(
        ramp_planes[b] * 255, cmap='gray', vmin=0, vmax=255
    )
    axes[row_bin, col_bin].set_title(f'Standard Binary: Plane {b}')
    axes[row_bin, col_bin].axis('off')

    # Gray-Coded Ramp
    axes[row_bin, col_bin + 1].imshow(
        gray_planes[b] * 255, cmap='gray', vmin=0, vmax=255
    )
    axes[row_bin, col_bin + 1].set_title(f'Gray-Coded: Plane {b}')
    axes[row_bin, col_bin + 1].axis('off')

  plt.suptitle(
      'Linear Ramp: Standard Binary vs. Gray Code Bit Planes',
      fontsize=14,
      fontweight='bold',
  )
  plt.tight_layout()
  plt.savefig(
      out_dir / 'ramp_vs_gray_decomposition.png', dpi=300, bbox_inches='tight'
  )
  plt.close()


def _bits_to_uint8(b):
    val = 0
    for bit in b:
        val = (val << 1) | int(bit)
    return val

def run_2_2(textured_path, smooth_path, stego_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cover_textured = np.asarray(Image.open(textured_path)).astype(np.uint8)
    cover_smooth = np.asarray(Image.open(smooth_path)).astype(np.uint8)
    stego_img = np.asarray(Image.open(stego_path)).astype(np.uint8)

    header = extract_lsb(stego_img, 24, plane=0)
    sync = _bits_to_uint8(header[0:8])
    h = _bits_to_uint8(header[8:16])
    w = _bits_to_uint8(header[16:24])

    # In 8-bit unsigned, a size of 256 overflows to 0
    if h == 0:
        h = 256
    if w == 0:
        w = 256

    assert sync == 0b10101010, f'Sync mismatch! Got {sync:08b}, expected 10101010'

    print(f'Payload Dimensions     : {h} x {w} pixels ({h * w} bits)')

    total_bits = 24 + (h * w)
    all_bits = extract_lsb(stego_img, total_bits, plane=0)
    payload_bits = all_bits[24:]  

    payload_img = (payload_bits.reshape((h, w)) * 255).astype(np.uint8)
    Image.fromarray(payload_img).save(out_dir / 'recovered_payload.png')

    p_textured = psnr(cover_textured, stego_img)
    print(f'PSNR(cover_textured, stego): {p_textured:.2f} dB')

    diff_map = (np.abs(stego_img.astype(np.int16) - cover_textured.astype(np.int16)) * 255).astype(np.uint8)
    Image.fromarray(diff_map).save(out_dir / "difference_map.png")

    # Embed payload of at least 256 x 256 bits into plane 4 of both covers
    req_bits = 256 * 256
    repeats = int(np.ceil(req_bits / len(payload_bits)))
    test_payload = np.tile(payload_bits, repeats)[:req_bits]

    stego_tex_p4 = embed_lsb(cover_textured, test_payload, plane=3)
    stego_sm_p4 = embed_lsb(cover_smooth, test_payload, plane=3)

    p_tex_p4 = psnr(cover_textured, stego_tex_p4)
    p_sm_p4 = psnr(cover_smooth, stego_sm_p4)
    print(f"PSNR(cover_textured, plane3): {p_tex_p4:.2f} dB")
    print(f"PSNR(cover_smooth, plane3)  : {p_sm_p4:.2f} dB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(stego_tex_p4, cmap="gray")
    axes[0].set_title(
        f"Textured Cover (Plane 3 Embedded)\nPSNR = {p_tex_p4:.2f} dB (Texture"
        " masks artifacts)",
        fontsize=11,
    )
    axes[0].axis('off')

    axes[1].imshow(stego_sm_p4, cmap="gray")
    axes[1].set_title(
        f"Smooth Cover (Plane 3 Embedded)\nPSNR = {p_sm_p4:.2f} dB (Severe"
        " contouring artifacts)",
        fontsize=11,
    )
    axes[1].axis('off')

    plt.suptitle(
        "Steganalysis Evidence: Plane 3 Embedding (256x256 bits)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        out_dir / 'textured_vs_smooth_evidence.png', dpi=300, bbox_inches='tight'
    )
    plt.close()


def run_2_3(cover_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cover = np.asarray(Image.open(cover_path)).astype(np.uint8)

    print("2.3 Channel Robustness (Gaussian & JPEG)")

    # Deterministic 128 secret bits
    rng = np.random.default_rng(42)
    secret_bits = rng.integers(0, 2, size=128, dtype=np.uint8)

    # Embed using both schemes
    stego_rob = embed_robust(cover, secret_bits, delta=2)
    stego_lsb = embed_lsb(cover, secret_bits, plane=0)

    p_rob = psnr(cover, stego_rob)
    p_lsb = psnr(cover, stego_lsb)
    print(f"Robust Scheme Stego PSNR: {p_rob:.2f} dB (Constraint: >= 40 dB)")
    print(f"Naive LSB Stego PSNR    : {p_lsb:.2f} dB")

    sigmas = [0, 1, 2, 5, 10, 20]
    ber_rob_gauss = []
    ber_lsb_gauss = []

    print("\nGaussian Noise Sweep")
    print(f"{'Sigma':6s} | {'Robust BER':12s} | {'Naive LSB BER':14s}")

    for s in sigmas:
        deg_rob = degrade_gaussian(stego_rob, s)
        rec_rob = extract_robust(deg_rob, cover, 128)
        b_rob = ber(secret_bits, rec_rob)
        ber_rob_gauss.append(b_rob)

        deg_lsb = degrade_gaussian(stego_lsb, s, rng=np.random.default_rng(0))
        rec_lsb = extract_lsb(deg_lsb, 128, plane=0)
        b_lsb = ber(secret_bits, rec_lsb)
        ber_lsb_gauss.append(b_lsb)

        print(f"{s:<6d} | {b_rob:<12.4f} | {b_lsb:<14.4f}")

    deg_rob_jpg = degrade_jpeg(stego_rob, 75)
    rec_rob_jpg = extract_robust(deg_rob_jpg, cover, 128)
    ber_rob_jpg = ber(secret_bits, rec_rob_jpg)

    deg_lsb_jpg = degrade_jpeg(stego_lsb, 75)
    rec_lsb_jpg = extract_lsb(deg_lsb_jpg, 128, plane=0)
    ber_lsb_jpg = ber(secret_bits, rec_lsb_jpg)

    print("\nJPEG Quality 75 Evaluation")
    print(f"Robust Scheme JPEG BER : {ber_rob_jpg:.4f}")
    print(f"Naive LSB JPEG BER     : {ber_lsb_jpg:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(sigmas, ber_rob_gauss, "o-", linewidth=2, color="blue", label="Block-Mean Robust Scheme")
    plt.plot(sigmas, ber_lsb_gauss, "s--", linewidth=2, color="red", label="Naive LSB (Plane 0)")
    plt.title("Bit Error Rate (BER) vs. Gaussian Noise ", fontsize=12, fontweight="bold")
    plt.xlabel("Noise Standard Deviation ", fontsize=11)
    plt.ylabel("Bit Error Rate (BER)", fontsize=11)
    plt.xticks(sigmas)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "ber_vs_sigma.png", dpi=300, bbox_inches="tight")
    plt.close()