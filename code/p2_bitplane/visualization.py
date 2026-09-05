import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from starter import bit_planes, gray_encode, psnr, reconstruct


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


