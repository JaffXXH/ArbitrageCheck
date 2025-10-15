import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ================================
# FX Black (Garman–Kohlhagen)
# ================================
def fx_call_price_gk(S, K, T, rd, rf, sigma):
    """
    European FX call price:
    C = S * exp(-rf*T) * N(d1) - K * exp(-rd*T) * N(d2),
    d1 = [ln(S/K) + (rd - rf + 0.5*sigma^2)T]/(sigma*sqrt(T)),
    d2 = d1 - sigma*sqrt(T).
    """
    if np.any(T <= 0):
        raise ValueError("All maturities must be strictly positive.")
    from scipy.stats import norm

    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    rd = np.asarray(rd, float)
    rf = np.asarray(rf, float)
    sigma = np.asarray(sigma, float)

    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    return S * np.exp(-rf * T) * norm.cdf(d1) - K * np.exp(-rd * T) * norm.cdf(d2)


def fx_call_delta_spot(S, K, T, rd, rf, sigma):
    from scipy.stats import norm
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / vol_sqrt_t
    return np.exp(-rf * T) * norm.cdf(d1)


# ================================
# Helpers
# ================================
def forward_price(S, T, rd, rf):
    return S * np.exp((rd - rf) * T)

def log_moneyness(S, K, T, rd, rf):
    F = forward_price(S, T, rd, rf)
    return np.log(K / F)

def normalize_calls(C, K, T, S, rd, rf):
    F = forward_price(S, T, rd, rf)
    x = K / F
    Cn = C / F
    return x, Cn


# ================================
# Static arbitrage checks
# ================================
def check_vertical_spread_monotonicity(C_row, K_row, tol=1e-12):
    """
    For fixed T, C(K) must be non-increasing in K.
    Flags indices where C(K_{i+1}) > C(K_i) + tol.
    """
    idx = np.argsort(K_row)
    K = K_row[idx]
    C = C_row[idx]
    diff = np.diff(C)
    viol_positions = np.where(diff > tol)[0] + 1
    viol_mask_sorted = np.zeros_like(C, dtype=bool)
    viol_mask_sorted[viol_positions] = True
    # map back to original order
    viol_mask = np.zeros_like(C_row, dtype=bool)
    viol_mask[idx] = viol_mask_sorted
    return viol_mask, {"sorted_indices": idx.tolist(), "viol_positions_sorted": viol_positions.tolist()}


def check_butterfly_convexity_triad(C_row, K_row, tol=1e-12):
    """
    Rigorous triad convexity:
    For any adjacent triplet K1<K2<K3,
        C(K2) <= w*C(K1) + (1-w)*C(K3)
    where w = (K3 - K2) / (K3 - K1).
    Violations imply negative-cost butterfly.
    Flags the middle index (K2) when inequality fails.
    """
    idx = np.argsort(K_row)
    K = K_row[idx]
    C = C_row[idx]
    n = len(K)
    viol_mask_sorted = np.zeros(n, dtype=bool)
    bad_triplets = []

    for i in range(1, n - 1):
        K1, K2, K3 = K[i - 1], K[i], K[i + 1]
        C1, C2, C3 = C[i - 1], C[i], C[i + 1]
        denom = (K3 - K1)
        if denom <= 0:
            continue
        w = (K3 - K2) / denom
        rhs = w * C1 + (1 - w) * C3
        if C2 > rhs + tol:
            viol_mask_sorted[i] = True
            bad_triplets.append({"i_mid_sorted": i, "K_triplet": (K1, K2, K3), "C_triplet": (C1, C2, C3), "w": w, "rhs": rhs})

    viol_mask = np.zeros_like(C_row, dtype=bool)
    viol_mask[idx] = viol_mask_sorted
    return viol_mask, {"sorted_indices": idx.tolist(), "bad_triplets": bad_triplets}


def check_calendar_monotonicity(C_grid, T, tol=1e-12):
    """
    For fixed K, C(T) must be non-decreasing in T.
    Flags positions where C(T_{j+1}) < C(T_j) - tol.
    """
    nT, nK = C_grid.shape
    viol = np.zeros_like(C_grid, dtype=bool)
    details = []
    for k in range(nK):
        Ck = C_grid[:, k]
        dC = np.diff(Ck)
        bad = np.where(dC < -tol)[0]
        if bad.size > 0:
            viol[bad + 1, k] = True
            details.append({"K_index": k, "T_indices": (bad + 1).tolist()})
    return viol, details


def check_price_bounds(C_row, K_row, T, S, rd, rf, tol=1e-12):
    """
    Bounds for European call:
      Lower bound: max(0, S*e^{-rf T} - K*e^{-rd T})
      Upper bound: S*e^{-rf T}
    Flags violations beyond tolerance.
    """
    disc_f = np.exp(-rf * T)
    disc_d = np.exp(-rd * T)
    lower = np.maximum(0.0, S * disc_f - K_row * disc_d)
    upper = S * disc_f
    low_viol = C_row < lower - tol
    up_viol = C_row > upper + tol
    mask = low_viol | up_viol
    details = {"lower_bound": lower, "upper_bound": upper,
               "lower_viol_indices": np.where(low_viol)[0].tolist(),
               "upper_viol_indices": np.where(up_viol)[0].tolist()}
    return mask, details


def check_forward_normalized_limits(C_row, K_row, T, S, rd, rf, tol=1e-12):
    """
    In forward-normalized space Cn(K/F):
      As K->0: Cn ~ 1
      As K->infty: Cn -> 0
    We test near the grid edges for gross violations.
    """
    x, Cn = normalize_calls(C_row, K_row, T, S, rd, rf)
    idx = np.argsort(x)
    x_sorted = x[idx]
    Cn_sorted = Cn[idx]
    edge_mask = np.zeros_like(C_row, dtype=bool)
    details = {}

    # Leftmost should not be << 1 by more than tol (not a strict inequality, just a sanity check)
    if Cn_sorted[0] < -tol or Cn_sorted[0] > 1 + tol:
        edge_mask[idx[0]] = True
        details["left_edge"] = {"x": x_sorted[0], "Cn": Cn_sorted[0]}
    # Rightmost should not exceed small positive tolerance
    if Cn_sorted[-1] < -tol or Cn_sorted[-1] > 0 + 1e-2:
        edge_mask[idx[-1]] = True
        details["right_edge"] = {"x": x_sorted[-1], "Cn": Cn_sorted[-1]}
    return edge_mask, details


# ================================
# Main evaluation and plotting
# ================================
def evaluate_fx_iv_arbitrage_and_plot(
    iv_grid, maturities, strikes, S, rd, rf, plot=True, tol=1e-12
):
    """
    Inputs:
      iv_grid: [nT, nK] implied vols
      maturities: [nT]
      strikes: [nK], strictly increasing
      S, rd, rf: scalars
    Output:
      dict of prices and violation masks/details.
    """
    iv_grid = np.asarray(iv_grid, float)
    T = np.asarray(maturities, float)
    K = np.asarray(strikes, float)
    if np.any(np.diff(K) <= 0):
        raise ValueError("Strikes must be strictly increasing.")

    nT, nK = iv_grid.shape
    if nT != len(T) or nK != len(K):
        raise ValueError("iv_grid shape must match maturities and strikes.")

    # Price grid
    call_prices = np.zeros_like(iv_grid)
    for i in range(nT):
        call_prices[i, :] = fx_call_price_gk(S, K, T[i], rd, rf, iv_grid[i, :])

    # Checks (per maturity)
    vertical_mask = np.zeros_like(iv_grid, dtype=bool)
    butterfly_mask = np.zeros_like(iv_grid, dtype=bool)
    bounds_mask = np.zeros_like(iv_grid, dtype=bool)
    edge_mask = np.zeros_like(iv_grid, dtype=bool)

    vertical_details = []
    butterfly_details = []
    bounds_details = []
    edge_details = []

    for i in range(nT):
        vmask, vdet = check_vertical_spread_monotonicity(call_prices[i, :], K, tol)
        bmask, bdet = check_butterfly_convexity_triad(call_prices[i, :], K, tol)
        pmask, pdet = check_price_bounds(call_prices[i, :], K, T[i], S, rd, rf, tol)
        emask, edet = check_forward_normalized_limits(call_prices[i, :], K, T[i], S, rd, rf, tol)

        vertical_mask[i, :] = vmask
        butterfly_mask[i, :] = bmask
        bounds_mask[i, :] = pmask
        edge_mask[i, :] = emask

        vdet["T_index"] = i
        bdet["T_index"] = i
        pdet["T_index"] = i
        edet["T_index"] = i
        vertical_details.append(vdet)
        butterfly_details.append(bdet)
        bounds_details.append(pdet)
        edge_details.append(edet)

    # Calendar across maturities
    calendar_mask, calendar_details = check_calendar_monotonicity(call_prices, T, tol)

    any_violation = vertical_mask | butterfly_mask | calendar_mask | bounds_mask | edge_mask

    # Plots
    if plot:
        TT, KK = np.meshgrid(T, K, indexing='ij')
        fig = plt.figure(figsize=(16, 12))

        # 1) Call surface
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(TT, KK, call_prices, cmap=cm.viridis, alpha=0.85, linewidth=0)
        ax1.set_title("FX call price surface (Garman–Kohlhagen)")
        ax1.set_xlabel("Maturity T")
        ax1.set_ylabel("Strike K")
        ax1.set_zlabel("Call price")
        fig.colorbar(surf, shrink=0.5, aspect=12, pad=0.1)
        vt = TT[any_violation]
        vk = KK[any_violation]
        vc = call_prices[any_violation]
        ax1.scatter(vt, vk, vc, color='red', s=30, label='Violation')
        ax1.legend(loc='best')

        # 2) IV vs log-moneyness
        ax2 = fig.add_subplot(222)
        for i in range(nT):
            m = log_moneyness(S, K, T[i], rd, rf)
            ax2.plot(m, iv_grid[i, :], label=f"T={T[i]:.3f}", alpha=0.85)
            idx_viol = np.where(any_violation[i, :])[0]
            if idx_viol.size > 0:
                ax2.scatter(m[idx_viol], iv_grid[i, idx_viol], color='red', s=32)
        ax2.set_title("Implied volatility by log-moneyness ln(K/F)")
        ax2.set_xlabel("ln(K/F)")
        ax2.set_ylabel("Implied volatility")
        ax2.legend(ncol=2, fontsize=8)

        # 3) IV vs spot delta
        ax3 = fig.add_subplot(223)
        for i in range(nT):
            deltas = fx_call_delta_spot(S, K, T[i], rd, rf, iv_grid[i, :])
            ax3.plot(deltas, iv_grid[i, :], label=f"T={T[i]:.3f}", alpha=0.85)
            idx_viol = np.where(any_violation[i, :])[0]
            if idx_viol.size > 0:
                ax3.scatter(deltas[idx_viol], iv_grid[i, idx_viol], color='red', s=32)
        ax3.set_title("Implied volatility by spot delta")
        ax3.set_xlabel("Delta (spot)")
        ax3.set_ylabel("Implied volatility")
        ax3.legend(ncol=2, fontsize=8)

        # 4) Violation heatmap
        ax4 = fig.add_subplot(224)
        im = ax4.imshow(any_violation, aspect='auto', origin='lower',
                        extent=[K.min(), K.max(), T.min(), T.max()], cmap='Reds', alpha=0.75)
        ax4.set_title("Violation heatmap (any type)")
        ax4.set_xlabel("Strike K")
        ax4.set_ylabel("Maturity T")
        fig.colorbar(im, ax=ax4, shrink=0.8, pad=0.05)

        plt.tight_layout()
        plt.show()

    return {
        "call_prices": call_prices,
        "vertical_spread_violations_mask": vertical_mask,
        "butterfly_violations_mask": butterfly_mask,
        "calendar_violations_mask": calendar_mask,
        "bounds_violations_mask": bounds_mask,
        "edge_limits_violations_mask": edge_mask,
        "details": {
            "vertical": vertical_details,
            "butterfly": butterfly_details,
            "calendar": calendar_details,
            "bounds": bounds_details,
            "edge_limits": edge_details,
        },
        "any_violation_mask": any_violation
    }


# ================================
# Example (intentionally arbitrageous)
# ================================
if __name__ == "__main__":
    S = 1.25
    rd = 0.03
    rf = 0.01

    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    strikes = np.array([0.90, 1.00, 1.05, 1.10, 1.20, 1.30])

    # Build IV with a local "hump" to force triad convexity failure at the middle strike
    base = 0.12
    wing = np.array([+0.02, 0.00, +0.025, 0.00, -0.01, -0.02])  # hump at K=1.05
    iv_grid = np.vstack([
        base + 0.018 * wing,
        base + 0.015 * wing,
        base + 0.012 * wing,
        base + 0.010 * wing,
    ])

    res = evaluate_fx_iv_arbitrage_and_plot(iv_grid, maturities, strikes, S, rd, rf, plot=True, tol=1e-12)

    print("Vertical violations per T:", [float(np.sum(res['vertical_spread_violations_mask'][i])) for i in range(len(maturities))])
    print("Butterfly violations per T:", [float(np.sum(res['butterfly_violations_mask'][i])) for i in range(len(maturities))])
    print("Calendar violations per K:", [float(np.sum(res['calendar_violations_mask'][:, j])) for j in range(len(strikes))])
    print("Bound violations per T:", [float(np.sum(res['bounds_violations_mask'][i])) for i in range(len(maturities))])
