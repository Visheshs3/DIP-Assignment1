from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from starter import apply_recovered, transfer_curve, fit_piecewise, support_report


def detect_segments(T, cnt, min_pts=4, threshold_jump=2):
    """Helper to find linear segments from the empirical transfer curve T[v]."""
    valid_v = np.where(cnt > 0)[0]
    valid_T = T[valid_v]

    # First difference (slope estimate between adjacent observed points)
    dT = np.diff(valid_T) / np.diff(valid_v)

    # Breakpoints occur where the derivative changes significantly
    bp_indices = [0]
    for i in range(1, len(dT)):
    # Check for sudden slope jump
        if (abs(dT[i] - dT[i - 1]) > threshold_jump and (i - bp_indices[-1]) >= min_pts):
            bp_indices.append(i)

    bp_indices.append(len(valid_v) - 1)

    bps = [valid_v[idx] for idx in bp_indices]
    bps[0] = 0
    bps[-1] = 255

    # Fit a line y = m*x + c to each segment
    slopes = []
    intercepts = []
    for i in range(len(bps) - 1):
        mask = (valid_v >= bps[i]) & (
            valid_v <= bps[i + 1] if i == len(bps) - 2 else valid_v < bps[i + 1]
        )
        seg_x = valid_v[mask]
        seg_y = valid_T[mask]

        if len(seg_x) >= 2:
            m, c = np.polyfit(seg_x, seg_y, 1)
        else:
            m, c = 0.0, seg_y[0] if len(seg_y) > 0 else 0.0

        slopes.append(m)
        intercepts.append(c)

    return bps, slopes, intercepts


def run_3_1(p3_dir, out_dir):
    p3_dir = Path(p3_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_in = np.asarray(Image.open(p3_dir / 'field_in.png')).astype(np.uint8)
    out_A = np.asarray(Image.open(p3_dir / 'field_out_A.png')).astype(np.uint8)
    out_B = np.asarray(Image.open(p3_dir / 'field_out_B.png')).astype(np.uint8)

    cases = [('Transform A (Levels)', out_A, 'scatter_A.png'),
            ('Transform B (Slicing)', out_B, 'scatter_B.png')]

    for title, img_out, fname in cases:
        print(f'3.1: {title}')

        T, cnt = transfer_curve(img_in, img_out)
        bps, slopes, intercepts = detect_segments(T, cnt)

        print(f'Recovered {len(slopes)} Segments:')
        print(f"{'Segment':8s} | {'Interval [x_start, x_end]':26s} | {'Slope (m)':12s} | {'Intercept (c)':14s}")
        for i in range(len(slopes)):
            print(f'{i+1:<8d} | [{bps[i]:<11d}, {bps[i+1]:<11d}] | {slopes[i]:<12.4f} | {intercepts[i]:<14.4f}')

        reconstructed = apply_recovered(img_in, bps, slopes, intercepts)

        abs_err = np.abs(reconstructed.astype(float) - img_out.astype(float))
        max_err = np.max(abs_err)
        mae = np.mean(abs_err)

        print(f'\nRe-application Error:')
        print(f'  Max Absolute Error : {max_err:.1f} intensity levels')
        print(f'  Mean Absolute Error: {mae:.4f} intensity levels')

        # Plot Scatter + Overlay
        valid_idx = np.where(cnt > 0)[0]
        plt.figure(figsize=(7, 5))
        plt.scatter(
            valid_idx,
            T[valid_idx],
            color='black',
            s=12,
            alpha=0.6,
            label='Measured Data $(v, T[v])$',
        )

        # Plot fitted piecewise curve
        x_grid = np.arange(256)
        y_fit = np.zeros(256)
        for i in range(len(slopes)):
            x_s, x_e = bps[i], bps[i + 1]
            mask = (x_grid >= x_s) & (x_grid <= x_e if i == len(slopes) - 1 else x_grid < x_e)
            y_fit[mask] = slopes[i] * x_grid[mask] + intercepts[i]

        plt.plot(
            x_grid,
            y_fit,
            color='crimson',
            linewidth=2.2,
            label='Fitted Piecewise Linear Curve',
        )

        # Mark breakpoints
        for bp in bps:
            plt.axvline(bp, color='blue', linestyle=':', alpha=0.5)

        plt.title(f'{title}: Measured Scatter & Fitted Overlay', fontsize=12, fontweight='bold')
        plt.xlabel('Input Intensity Level $v$', fontsize=11)
        plt.ylabel('Output Intensity Level $T[v]$', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()


def run_3_2(p3_dir, out_dir):
    p3_dir = Path(p3_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print('Problem 3.2: Pair C Model Selection')

    img_in = np.asarray(Image.open(p3_dir / 'nebula_in.png')).astype(np.uint8)
    img_out = np.asarray(Image.open(p3_dir / 'nebula_out_C.png')).astype(np.uint8)

    T, cnt = transfer_curve(img_in, img_out)
    bps, slopes, intercepts, k = fit_piecewise(T, cnt, max_seg=6)

    print(f'Automatically Selected Segments (k): {k}')
    print(f"{'Segment':8s} | {'Interval [x_start, x_end]':26s} | {'Slope (m)':12s} | {'Intercept (c)':14s}")

    for i in range(k):
        print(f'{i+1:<8d} | [{bps[i]:<11d}, {bps[i+1]:<11d}] | {slopes[i]:<12.4f} | {intercepts[i]:<14.4f}')

    reconstructed = apply_recovered(img_in, bps, slopes, intercepts)
    abs_diff = np.abs(reconstructed.astype(float) - img_out.astype(float))
    max_err = np.max(abs_diff)
    mae = np.mean(abs_diff)

    print(f'\nRe-application Quality on Pair C:')
    print(f'  Max Absolute Error : {max_err:.1f} levels')
    print(f'  Mean Absolute Error: {mae:.4f} levels')

    valid_idx = np.where(cnt > 0)[0]
    plt.figure(figsize=(7.5, 5))
    plt.scatter(
        valid_idx,
        T[valid_idx],
        color='black',
        s=10,
        alpha=0.5,
        label='Measured $(v, T[v])$',
    )

    x_grid = np.arange(256)
    y_fit = np.zeros(256)
    for i in range(k):
        xs, xe = bps[i], bps[i + 1]
        mask = (x_grid >= xs) & (x_grid <= xe if i == k - 1 else x_grid < xe)
        y_fit[mask] = slopes[i] * x_grid[mask] + intercepts[i]

    plt.plot(
        x_grid,
        y_fit,
        color='crimson',
        linewidth=2.2,
        label=f'Fitted Curve ({k} segments)',
    )
    for bp in bps:
        plt.axvline(bp, color='blue', linestyle=':', alpha=0.5)

    plt.title(
        f'Pair C (Nebula): Fitted Transfer Curve ({k} Segments)',
        fontsize=12,
        fontweight='bold',
    )
    plt.xlabel('Input Intensity Level $v$', fontsize=11)
    plt.ylabel('Output Intensity Level $T[v]$', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / 'scatter_C.png', dpi=300, bbox_inches='tight')
    plt.close()
    print()

    
    print('--- Problem 3.2: Quantization Staircase Demonstration ---')

    # Generate a single continuous line with fractional slope m = 0.35
    x_synth = np.arange(120, dtype=np.uint8)
    y_continuous = 0.35 * x_synth + 15.0
    y_quantized = np.round(y_continuous).astype(np.uint8)  # Discrete integer steps

    T_synth = np.full(256, np.nan)
    T_synth[x_synth] = y_quantized
    cnt_synth = np.zeros(256, dtype=int)
    cnt_synth[x_synth] = 50

    bps_s, slopes_s, inter_s, k_s = fit_piecewise(T_synth, cnt_synth, max_seg=6)

    print(f'Synthetic Quantized Input (True line: y = 0.35x + 15):')
    print(f'  Selected segments (k)  : {k_s} (Expected: 1)')
    print(f'  Recovered slope        : {slopes_s[0]:.4f}')
    print(f'  Recovered intercept    : {inter_s[0]:.4f}')

    if k_s == 1:
        print('  VERIFICATION SUCCESS:')
    else:
        print(f'VERIFICATION FAILED: Overfitted into {k_s} alternating segments.')


def run_3_3(p3_dir, out_dir, min_support=25):
    print()
    p3_dir = Path(p3_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- Problem 3.3: Identifiability Analysis (Pair D) ---")

    img_in = np.asarray(Image.open(p3_dir / "field_skycrop_in.png")).astype(np.uint8)
    img_out = np.asarray(Image.open(p3_dir / "field_skycrop_out_D.png")).astype(np.uint8)

    T, cnt = transfer_curve(img_in, img_out)

    report_text, supported_ranges, unsupported_ranges = support_report(cnt, min_count=min_support)
    print(report_text)

    # Plot Histogram with Support Threshold Marked
    plt.figure(figsize=(8, 4.5))
    levels = np.arange(256)
    plt.bar(levels,cnt,width=1.0,color="steelblue",edgecolor="none",alpha=0.8,label="Input Pixel Counts")
    plt.axhline(
        min_support,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=f"Min support({min_support} pixels)",
    )
    plt.yscale("log")
    plt.title("Input Histogram & Support Threshold",fontsize=12,fontweight="bold",)
    plt.xlabel("Input Intensity Level $v$", fontsize=11)
    plt.ylabel("Pixel Count (Log Scale)", fontsize=11)
    plt.xlim(0, 255)
    plt.legend()
    plt.tight_layout()
    hist_path = out_dir / "histogram_D.png"
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()

    cnt_filtered = cnt.copy()
    cnt_filtered[cnt < min_support] = 0

    valid_v = np.where(cnt_filtered > 0)[0]
    v_min, v_max = supported_ranges[0][0], supported_ranges[0][1]
    bps, slopes, intercepts, k = fit_piecewise(T, cnt_filtered, max_seg=4)

    supported_bps = [v_min]
    for bp in bps[1:-1]:
        if v_min < bp < v_max:
            supported_bps.append(bp)
    supported_bps.append(v_max)

    final_slopes = []
    final_inter = []
    num_supp_segs = len(supported_bps) - 1

    print(f"\nRecovered {num_supp_segs} Constrained Segment(s) over [{v_min} {v_max}]:")
    print(f"{'Segment':8s} | {'Interval [start, end]':24s} | {'Slope (m)':12s} | {'Intercept (c)':14s}")
    
    for i in range(num_supp_segs):
        xs, xe = supported_bps[i], supported_bps[i + 1]
        mask = (valid_v >= xs) & (valid_v <= xe if i == num_supp_segs - 1 else valid_v < xe)
        m, c = np.polyfit(valid_v[mask], T[valid_v[mask]], 1)
        final_slopes.append(m)
        final_inter.append(c)
        print(
            f"{i+1:<8d} | [{xs:<3d}, {xe:<3d}]                 | {m:<12.4f} |"
            f" {c:<14.4f}"
        )
    plt.figure(figsize=(8, 5))
    plt.scatter(
        valid_v,
        T[valid_v],
        color="black",
        s=14,
        alpha=0.6,
        label=f"Supported Data (N >= {min_support})",
    )

    # Plot fitted segments over their domain
    for i in range(num_supp_segs):
        xs, xe = supported_bps[i], supported_bps[i + 1]
        x_seg = np.linspace(xs, xe, 100)
        y_seg = final_slopes[i] * x_seg + final_inter[i]
        lbl = f"Fitted Segment {i+1}" if i == 0 else f"Fitted Segment {i+1}"
        plt.plot(x_seg, y_seg, color="crimson", linewidth=2.5, label=lbl)

    # Shade the unconstrained region
    plt.axvspan(
        v_max,
        255,
        color="gray",
        alpha=0.2,
        hatch="//",
        label=f"Unconstrained Range [{v_max+1}, 255]",
    )
    if v_min > 0:
        plt.axvspan(
            0,
            v_min - 1,
            color="gray",
            alpha=0.15,
            hatch="//",
            label=f"Unconstrained Range [0, {v_min-1}]",
        )

    plt.title(
        f"Pair D (Sky Crop): Recovered {num_supp_segs} Supported Segment(s)",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Input Intensity Level $v$", fontsize=11)
    plt.ylabel("Output Intensity Level $T[v]$", fontsize=11)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    scatter_path = out_dir / "scatter_D.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    supp_mask = (img_in >= v_min) & (img_in <= v_max)

    lut_partial = np.zeros(256, dtype=np.float64)
    for i in range(num_supp_segs):
        xs, xe = supported_bps[i], supported_bps[i + 1]
        xr = np.arange(xs, xe + 1 if i == num_supp_segs - 1 else xe)
        lut_partial[xr] = final_slopes[i] * xr + final_inter[i]

    pred = np.clip(np.round(lut_partial[img_in]), 0, 255).astype(np.uint8)
    err = np.abs(pred[supp_mask].astype(float) - img_out[supp_mask].astype(float))
    print(f"\nRe-application Error on Supported Domain:")
    print(f"  Max Absolute Error : {np.max(err):.1f} levels")
    print(f"  Mean Absolute Error: {np.mean(err):.4f} levels")