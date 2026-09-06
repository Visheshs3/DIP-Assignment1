"""P3 starter. Implement every function marked TODO.

The transform is pointwise. That is the whole hint.
"""
import numpy as np
from pathlib import Path

def transfer_curve(img_in, img_out):
    """3.1: estimate T[v] for v in 0..255, plus the pixel count supporting
    each level. Levels with no pixels must be distinguishable from levels
    that map to zero -- you need both arrays."""
    x = img_in.ravel()
    y = img_out.ravel()

    T = np.full(256, np.nan, dtype=np.float64)
    cnt = np.bincount(x, minlength=256)

    for v in range(256):
        if cnt[v] > 0:
            T[v] = np.mean(y[x == v])

    return T, cnt




def fit_piecewise(T, cnt, max_seg=10, min_len=5):
    """3.2: fit the smallest number of segments the data justifies.

    You need a complexity penalty; a pure least-squares fit always improves
    with more segments. Read the warning in the handout about the quantisation
    floor before you tune it.

    Returns (breakpoints, slopes, intercepts, n_segments).
    """
    valid_mask = (cnt > 0) & (~np.isnan(T))
    valid_x = np.where(valid_mask)[0].astype(np.float64)
    valid_y = T[valid_mask].astype(np.float64)
    N = len(valid_x)

    if N < min_len:
        m, c = np.polyfit(valid_x, valid_y, 1) if N >= 2 else (0.0, 0.0)  # simple y=mx+c fit
        return [0, 255], [float(m)], [float(c)], 1

    # Precompute prefix sums for O(1) linear regression Sum of Squared Errors
    P_x = np.concatenate([[0.0], np.cumsum(valid_x)])  
    P_y = np.concatenate([[0.0], np.cumsum(valid_y)])
    P_xx = np.concatenate([[0.0], np.cumsum(valid_x * valid_x)])
    P_yy = np.concatenate([[0.0], np.cumsum(valid_y * valid_y)])
    P_xy = np.concatenate([[0.0], np.cumsum(valid_x * valid_y)])

    
    def segment_fit(i, j):
        """Fits line y = m*x + c to slice valid_x[i:j], valid_y[i:j]. Returns (SSE, m, c)."""
        n = j - i
        if n < 2:
            return 0.0, 0.0, valid_y[i] # constant function
        sx = P_x[j] - P_x[i]  
        sy = P_y[j] - P_y[i]
        sxx = P_xx[j] - P_xx[i]
        syy = P_yy[j] - P_yy[i]
        sxy = P_xy[j] - P_xy[i]
    
        vx = sxx - (sx * sx) / n
        vy = syy - (sy * sy) / n
        vxy = sxy - (sx * sy) / n
    
        if vx <= 1e-9:  # Degenerate vertical or single-valued slice
            return max(0.0, vy), 0.0, sy / n
    
        m = vxy / vx
        c = (sy - m * sx) / n
        sse = max(0.0, vy - m * vxy)
        return sse, m, c
    
    # cost matrix for all candidate segments of length >= min_len
    cost = np.full((N + 1, N + 1), np.inf)
    for i in range(0, N - min_len + 1):
        for j in range(i + min_len, N + 1):
            cost[i, j] = segment_fit(i, j)[0]

    # dp[k, j] = min SSE using k segments on points 0..j
    max_k = min(max_seg, N // min_len)
    dp = np.full((max_k + 1, N + 1), np.inf)
    parent = np.zeros((max_k + 1, N + 1), dtype=int)

    for j in range(min_len, N + 1):
        dp[1, j] = cost[0, j]
        parent[1, j] = 0

    # Recurrence: k segments
    for k in range(2, max_k + 1):
        for j in range(k * min_len, N + 1):
            best_cost = np.inf
            best_i = -1
            for i in range((k - 1) * min_len, j - min_len + 1):
                c_val = dp[k - 1, i] + cost[i, j]
                if c_val < best_cost:
                    best_cost = c_val
                    best_i = i
            dp[k, j] = best_cost
            parent[k, j] = best_i

    # Model Selection: BIC with the Quantization Floor (sigma_q^2 = 1/12)
    # Any residual variance below 1/12 (~0.0833) is pure integer rounding noise.
    sigma_q2 = 1.0 / 12.0
    bic_scores = []

    for k in range(1, max_k + 1):
        raw_mse = dp[k, N] / N
        effective_mse = max(raw_mse, sigma_q2)
        num_params = 3 * k - 1  # k slopes, k intercepts, (k-1) interior cut-points
        bic = N * np.log(effective_mse) + num_params * np.log(N)
        bic_scores.append(bic)

    best_k = int(np.argmin(bic_scores)) + 1

    #Backtrack split indices for the chosen best_k
    curr_j = N
    split_indices = [curr_j]
    for k in range(best_k, 1, -1):
        prev_i = parent[k, curr_j]
        split_indices.append(prev_i)
        curr_j = prev_i
    split_indices.append(0)
    split_indices.reverse()

    bps = [0]
    for idx in split_indices[1:-1]:
        bp = int(round((valid_x[idx - 1] + valid_x[idx]) / 2.0))
        bps.append(bp)
    bps.append(255)

    slopes = []
    intercepts = []
    for s in range(best_k):
        i = split_indices[s]
        j = split_indices[s + 1]
        _, m, c = segment_fit(i, j)
        slopes.append(float(m))
        intercepts.append(float(c))

    return bps, slopes, intercepts, best_k


def apply_recovered(img, bps, slopes, inter):
    """3.1: build the LUT from your fit and apply it."""
    lut = np.zeros(256, dtype=np.float64)
    n_seg = len(slopes)

    for i in range(n_seg):
        x_start = int(round(bps[i]))
        x_end = int(round(bps[i + 1]))

        if i == n_seg - 1:
            x_vals = np.arange(x_start, 256)
        else:
            x_vals = np.arange(x_start, x_end)

        lut[x_vals] = slopes[i] * x_vals + inter[i]

    lut_uint8 = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    return lut_uint8[img]


def support_report(cnt, min_count=25):
    """3.3: which intensity ranges does the input not sample well enough
    to constrain the transform? Return something human-readable."""
    supported_mask = cnt >= min_count
    supported_levels = np.where(supported_mask)[0]

    if len(supported_levels) == 0:
        return "No levels meet the support threshold.", [], [(0, 255)]

    # Find contiguous blocks of supported levels

    supported_ranges = []
    range_start = int(supported_levels[0])
    prev_level = int(supported_levels[0])
    for v in supported_levels[1:]:
        v = int(v)
        if v == prev_level + 1:
            prev_level = v
        else:
            supported_ranges.append((range_start, prev_level))
            range_start = v
            prev_level = v
    supported_ranges.append((range_start, prev_level))

    unsupported_ranges = []
    curr = 0
    for s, e in supported_ranges:
        if curr < s:
            unsupported_ranges.append((curr, s - 1))
        curr = e + 1

    if curr <= 255:
        unsupported_ranges.append((curr, 255))

    lines = [
        f"Support Threshold: >= {min_count} pixels per level",
        f"Well-Supported Range(s) (Constrained): {supported_ranges}",
        f"Unsupported Range(s) (Unconstrained): {unsupported_ranges}",
        f"Total sampled levels with support   : {len(supported_levels)} / 256",
    ]
    report_str = "\n".join(lines)
    return report_str, supported_ranges, unsupported_ranges


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    from visualization import run_3_1, run_3_2, run_3_3
    run_3_1(root / 'images/p3', root / 'output/p3')
    run_3_2(root / 'images/p3', root / 'output/p3')
    run_3_3(root / 'images/p3', root / 'output/p3')