import numpy as np
import time
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt
import os

from starter import (
    kernel_bank,
    conv2d_fft, conv2d_im2col, conv2d_loops, conv2d_taps,conv2d_lowrank,conv2d_separable,
    psnr
)

def run_1_1(img):
    crop128 = img[:128, :128]
    K_sym = kernel_bank(7)["gaussian"]
    ref_sym = convolve2d(crop128, K_sym, mode="same", boundary="fill")

    print("1.1 Verification against scipy.signal.convolve2d (7x7 Gaussian) ")
    for fn in (conv2d_loops, conv2d_taps, conv2d_im2col, conv2d_fft):
        res = fn(crop128, K_sym)
        max_err = np.max(np.abs(res - ref_sym))
        print(f"  {fn.__name__:16s} max |error| = {max_err:.3e}")

    print("\n1.1 Asymmetric Kernel Test (3x3 Non-Symmetric Ramp)")
    K_asym = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])
    ref_conv = convolve2d(crop128, K_asym, mode="same", boundary="fill")
    ref_corr = correlate2d(crop128, K_asym, mode="same", boundary="fill")
    my_conv = conv2d_taps(crop128, K_asym)

    err_vs_conv = np.max(np.abs(my_conv - ref_conv))
    diff_vs_corr = np.max(np.abs(my_conv - ref_corr))

    print(f"  |My Output - convolve2d|  = {err_vs_conv:.3e}  (true convolution)")
    print(f"  |My Output - correlate2d| = {diff_vs_corr:.3e}  (Differs from correlation)")

    print("\n1.1 conv2d_loops Timing & Extrapolation ")
    t0 = time.perf_counter()
    conv2d_loops(crop128, K_sym)
    t_measured = time.perf_counter() - t0

    # Theoretical scale factor: (H_new / H_old)^2 * (k_new / k_old)^2
    scale_factor = ((2048 / 128) ** 2) * ((15 / 7) ** 2)
    t_extrapolated = t_measured * scale_factor

    print(f"  Measured time (128x128, k=7):     {t_measured:.4f} s")
    print(f"  Extrapolated time (2048x2048, k=15): {t_extrapolated:.2f} s (~{t_extrapolated/60:.2f} min)")


def run_1_2(out_dir):
    print()
    os.makedirs(out_dir, exist_ok=True)
    bank = kernel_bank(k=15)
    tol = 1e-3

    print(f"1.2 Kernel Rank Table (k=15,tol={tol})")
    print(f"{'Kernel Name':18s} | {'Rank':6s} | {'Leading Singular Value':30s}")

    plt.figure(figsize=(9, 6))
    for name, K in bank.items():
        s = np.linalg.svd(K, compute_uv=False)
        r = int(np.sum(s > tol))
        s_norm = s / (s[0] + 1e-15)

        print(f"{name:18s} | {r:6d} | {s[0]:.2e}")
        plt.semilogy(range(1, len(s) + 1), s_norm, marker='o', label=f"{name} (r={r})")

    plt.axhline(tol, color='black', linestyle='--', linewidth=1.2, label=f"Threshold ({tol})")
    plt.title("Normalized Singular Value Spectra (k=15)", fontsize=13, fontweight='bold')
    plt.xlabel("Singular Value Index i", fontsize=11)
    plt.ylabel(r"$\sigma_i / \sigma_1$ (log scale)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir + "rank_spectra.png", dpi=300)
    plt.close()


def run_1_3(img_full, out_dir):
    print()
    os.makedirs(out_dir, exist_ok=True)
    k = 21

    bank = kernel_bank(k)
    crop512 = img_full[:512, :512]

    target_kernels = ["disk", "log", "motion_45deg", "random"]
    ranks = list(range(1, k + 1))

    results = {name: [] for name in target_kernels}

    print(f"1.3 Low-Rank PSNR Sweeps (k={k}, 512x512 Crop)")
    for name in target_kernels:
        K = bank[name]
        ref = conv2d_fft(crop512, K)  
        for r in ranks:
            approx = conv2d_lowrank(crop512, K, r)
            p = psnr(ref, approx)
            results[name].append(p)
        print(f"  {name:14s} | r=1 PSNR: {results[name][0]:6.2f} dB | r={k} PSNR: {results[name][-1]:6.2f} dB")

    plt.figure(figsize=(8, 5.5))
    for name in target_kernels:
        p_vals = np.array(results[name])
        # Replace infinite PSNR (exact reconstruction) with 120 dB for plotting
        p_vals_clean = np.where(np.isinf(p_vals), 120.0, p_vals)
        plt.plot(ranks, p_vals_clean, marker='s', markersize=4, label=name)

    plt.title(f"Reconstruction PSNR vs. Truncation Rank r (k={k})", fontsize=12, fontweight='bold')
    plt.xlabel("Truncation Rank r (Number of SVD Components)", fontsize=11)
    plt.ylabel("PSNR (dB) vs Exact Convolution", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir + "psnr_vs_rank.png", dpi=300)
    plt.close()



def run_1_4(img_full, out_dir):
    print()
    os.makedirs(out_dir, exist_ok=True)

    k_vals = [3, 7, 11, 15, 21, 31]
    n_vals = [128, 256, 512, 1024, 2048]

    # Memory Table (k=15, float64 = 8 bytes)
    print("1.4 Peak im2col Memory Requirements (k=15) ")
    print(f"{'N':6s} | {'Matrix Shape (Rows x Cols)':28s} | {'Memory (MB)':14s} | {'Memory (GiB)':14s}")
    for n in n_vals:
        elements = (n**2) * (15**2)
        bytes_used = elements * 8
        mb = bytes_used / (1024**2)
        gib = bytes_used / (1024**3)
        print(f"{n:<6d} | {n*n:<10d} x {15*15:<15d} | {mb:12.2f} MB | {gib:12.4f} GiB")

    # Sweep 1: vs k at N=512
    crop512 = img_full[:512, :512]
    t_taps_k, t_im2col_k, t_sep_k, t_fft_k = [], [], [], []

    for k in k_vals:
        K = kernel_bank(k)["gaussian"]

        t0 = time.perf_counter()
        conv2d_taps(crop512, K)
        t_taps_k.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        conv2d_im2col(crop512, K)
        t_im2col_k.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        conv2d_separable(crop512, K)
        t_sep_k.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        conv2d_fft(crop512, K)
        t_fft_k.append(time.perf_counter() - t0)

    log_k = np.log(k_vals)
    slope_taps_k = np.polyfit(log_k, np.log(t_taps_k), 1)[0]
    slope_im2col_k = np.polyfit(log_k, np.log(t_im2col_k), 1)[0]
    slope_sep_k = np.polyfit(log_k, np.log(t_sep_k), 1)[0]

    plt.figure(figsize=(7.5, 5))
    plt.loglog(k_vals, t_taps_k, 'o-', label=f"conv2d_taps (slope = {slope_taps_k:.2f})")
    plt.loglog(k_vals, t_im2col_k, 's-', label=f"conv2d_im2col (slope = {slope_im2col_k:.2f})")
    plt.loglog(k_vals, t_sep_k, '^-', label=f"conv2d_separable (slope = {slope_sep_k:.2f})")
    plt.loglog(k_vals, t_fft_k, 'd-', label="conv2d_fft")
    plt.title("Runtime vs. Kernel Size k (N=512)", fontsize=12, fontweight='bold')
    plt.xlabel("Kernel Size k", fontsize=11)
    plt.ylabel("Time (seconds)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir + "runtime_vs_k.png", dpi=300)
    plt.close()

    # Sweep 2: vs N at k=15
    K15 = kernel_bank(15)["gaussian"]
    t_taps_n, t_im2col_n, t_sep_n, t_fft_n = [], [], [], []

    for n in n_vals:
        crop_n = img_full[:n, :n]
        t0 = time.perf_counter(); conv2d_taps(crop_n, K15); t_taps_n.append(time.perf_counter() - t0)

        if n <= 1024:
            t0 = time.perf_counter(); conv2d_im2col(crop_n, K15); t_im2col_n.append(time.perf_counter() - t0)
        else:
            t_im2col_n.append(t_im2col_n[-1] * 4.0)

        t0 = time.perf_counter(); conv2d_separable(crop_n, K15); t_sep_n.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); conv2d_fft(crop_n, K15); t_fft_n.append(time.perf_counter() - t0)

    log_n = np.log(n_vals)
    slope_taps_n = np.polyfit(log_n, np.log(t_taps_n), 1)[0]
    slope_sep_n = np.polyfit(log_n, np.log(t_sep_n), 1)[0]

    plt.figure(figsize=(7.5, 5))
    plt.loglog(n_vals, t_taps_n, 'o-', label=f"conv2d_taps (slope = {slope_taps_n:.2f})")
    plt.loglog(n_vals, t_im2col_n, 's--', label="conv2d_im2col (N=2048 extrapolated)")
    plt.loglog(n_vals, t_sep_n, '^-', label=f"conv2d_separable (slope = {slope_sep_n:.2f})")
    plt.loglog(n_vals, t_fft_n, 'd-', label="conv2d_fft")
    plt.title("Runtime vs. Image Dimension N (k=15)", fontsize=12, fontweight='bold')
    plt.xlabel("Image Dimension N", fontsize=11)
    plt.ylabel("Time (seconds)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir + "runtime_vs_n.png", dpi=300)
    plt.close()

    # Separable Speedup
    idx_512 = n_vals.index(512)
    idx_2048 = n_vals.index(2048)
    sp_512 = t_taps_n[idx_512] / t_sep_n[idx_512]
    sp_2048 = t_taps_n[idx_2048] / t_sep_n[idx_2048]

    print(f"\nSeparable Speedup at 512x512:   {sp_512:.2f}x (Theoretical: k/2 = 7.5x)")
    print(f"Separable Speedup at 2048x2048: {sp_2048:.2f}x")