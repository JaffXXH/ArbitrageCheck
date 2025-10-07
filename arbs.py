"""
fx_iv_arbitrage.py

Single-file module:
- Smile reconstruction from ATM, RR, STR (BF) for 10Δ and 25Δ nodes
- Delta-to-strike conversion (Black forward delta)
- Butterfly arbitrage checks per tenor via call price convexity in strike
- Calendar arbitrage checks across tenors via total variance monotonicity and price-in-time
- Time interpolation methods for implied volatility (variance-linear, vol-linear, log-variance-linear)

Author: (c) 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math


# -----------------------------
# Core math: Normal, Black call
# -----------------------------

SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _norm_cdf(x: float) -> float:
    # Abramowitz-Stegun approximation (sufficient for checks; use mpmath for higher precision if needed)
    # For quantitative desk-grade precision, consider importing scipy.stats.norm.cdf.
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    poly = ((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k
    approx = 1.0 - _norm_pdf(abs(x)) * poly
    return approx if x >= 0 else 1.0 - approx

def black_call_price(F: float, K: float, T: float, sigma: float, DF: float = 1.0) -> float:
    """
    Black-76 forward call: C = DF * [ F N(d1) - K N(d2) ]
    """
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0 or DF <= 0:
        return max(DF * (F - K), 0.0)  # degenerate guard
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return DF * (F * _norm_cdf(d1) - K * _norm_cdf(d2))

def black_forward_delta_call(F: float, K: float, T: float, sigma: float) -> float:
    """
    Forward delta for a call under Black-76: Δ_fwd = N(d1)
    """
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0:
        return 1.0 if F > K else 0.0
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    return _norm_cdf(d1)

def strike_from_forward_delta_call(F: float, T: float, sigma: float, delta: float) -> float:
    """
    Invert Δ_fwd = N(d1) to get K:
    d1 = N^{-1}(delta), d1 = [ln(F/K) + 0.5 σ^2 T] / (σ √T)
    => ln(F/K) = d1 σ √T - 0.5 σ^2 T
    => K = F / exp(d1 σ √T - 0.5 σ^2 T)
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("Delta must be in (0,1) for call.")
    if sigma <= 0 or T <= 0 or F <= 0:
        raise ValueError("Invalid inputs for strike inversion.")
    # Inverse CDF via binary search (sufficient accuracy for desk checks)
    d1 = _inv_norm_cdf(delta)
    vol_sqrt_t = sigma * math.sqrt(T)
    ln_F_over_K = d1 * vol_sqrt_t - 0.5 * sigma * sigma * T
    return F / math.exp(ln_F_over_K)

def _inv_norm_cdf(p: float) -> float:
    """
    Approximation for inverse standard normal CDF (Moro's algorithm variant).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1).")
    # Coefficients
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    # Central region
    x = p - 0.5
    if abs(x) < 0.42:
        r = x * x
        num = x * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
        den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0
        return num / den
    # Tail region
    r = p if x > 0 else 1.0 - p
    s = math.log(-math.log(r))
    t = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * (c[5] + s * (c[6] + s * (c[7] + s * c[8])))))))
    return t if x > 0 else -t


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class TenorQuote:
    """
    FX desk-style quotes per tenor:
    - ATM: at-the-money implied volatility (log-moneyness ~ 0)
    - RR25: 25Δ risk reversal (vol_call25 - vol_put25)
    - STR25 (BF25): 25Δ butterfly = 0.5*(vol_call25 + vol_put25) - vol_ATM
    - RR10: 10Δ risk reversal
    - STR10 (BF10): 10Δ butterfly
    Conventions: Δ are forward call deltas 0.25 and 0.10 for 'call', with puts symmetric via RR.
    """
    ATM: float
    RR25: float
    STR25: float
    RR10: float
    STR10: float


@dataclass
class MarketTenor:
    """
    Market data required per tenor for pricing and checks:
    - T: time to maturity in years
    - F: forward price for the FX pair at this tenor
    - DF: discount factor to maturity (optional, default 1.0)
    """
    T: float
    F: float
    DF: float = 1.0


@dataclass
class SmileNode:
    """
    A node on the smile with:
    - label: e.g., '10P', '10C', '25P', '25C', 'ATM'
    - delta: forward call delta for calls; use 1 - delta for puts when mapping (we store canonical call delta)
    - K: strike corresponding to delta under Black, using node's vol
    - vol: implied volatility at this node
    """
    label: str
    delta: float
    K: float
    vol: float


# -----------------------------
# Smile construction
# -----------------------------

class SmileBuilder:
    """
    Construct discrete smile nodes (10Δ, 25Δ put/call and ATM) from ATM/RR/STR quotes.
    """

    def __init__(self, quote: TenorQuote, market: MarketTenor):
        self.q = quote
        self.mkt = market

    def build_nodes(self) -> List[SmileNode]:
        """
        Build smile nodes using standard FX vol quoting identities:

        Let:
        RR_x = vol_call_x - vol_put_x
        BF_x (STR_x) = 0.5*(vol_call_x + vol_put_x) - vol_ATM

        => vol_call_x = vol_ATM + BF_x + 0.5 * RR_x
        => vol_put_x  = vol_ATM + BF_x - 0.5 * RR_x

        Nodes:
        - 10C: delta=0.10, vol as above
        - 10P: symmetric put, we store as label '10P' but use call-delta 0.90 when converting (equivalently, for put we can use Δ_call=1-Δ_put)
        - 25C, 25P
        - ATM: label 'ATM' with delta ~ 0.5 (we set strike K=F under Black ATM; delta implied by sigma and T)
        """
        ATM = self.q.ATM

        vol_25c = ATM + self.q.STR25 + 0.5 * self.q.RR25
        vol_25p = ATM + self.q.STR25 - 0.5 * self.q.RR25
        vol_10c = ATM + self.q.STR10 + 0.5 * self.q.RR10
        vol_10p = ATM + self.q.STR10 - 0.5 * self.q.RR10

        # Construct strikes from deltas using each node's vol
        nodes = []

        # 10C: Δ_call = 0.10
        K_10c = strike_from_forward_delta_call(self.mkt.F, self.mkt.T, vol_10c, 0.10)
        nodes.append(SmileNode(label="10C", delta=0.10, K=K_10c, vol=vol_10c))

        # 10P: for a put with 10Δ_put, the corresponding call delta is 1 - Δ_put = 0.90
        K_10p = strike_from_forward_delta_call(self.mkt.F, self.mkt.T, vol_10p, 0.90)
        nodes.append(SmileNode(label="10P", delta=0.90, K=K_10p, vol=vol_10p))

        # 25C
        K_25c = strike_from_forward_delta_call(self.mkt.F, self.mkt.T, vol_25c, 0.25)
        nodes.append(SmileNode(label="25C", delta=0.25, K=K_25c, vol=vol_25c))

        # 25P: call-delta = 0.75
        K_25p = strike_from_forward_delta_call(self.mkt.F, self.mkt.T, vol_25p, 0.75)
        nodes.append(SmileNode(label="25P", delta=0.75, K=K_25p, vol=vol_25p))

        # ATM: set K = F, vol = ATM
        nodes.append(SmileNode(label="ATM", delta=0.5, K=self.mkt.F, vol=ATM))

        # Sort nodes by strike ascending to facilitate butterfly checks
        nodes.sort(key=lambda n: n.K)
        return nodes


# -----------------------------
# Arbitrage checks
# -----------------------------

class ArbitrageChecker:
    """
    Arbitrage checks for:
    - Butterfly arbitrage per tenor: call price is decreasing in strike and convex in strike.
    - Calendar arbitrage across tenors: total variance and call price monotonicity in maturity for fixed strike buckets.
    """

    def __init__(self, nodes_by_tenor: Dict[str, List[SmileNode]], market_by_tenor: Dict[str, MarketTenor]):
        """
        nodes_by_tenor: mapping tenor key -> list of SmileNode (sorted by strike)
        market_by_tenor: mapping tenor key -> MarketTenor (T, F, DF)
        """
        self.nodes_by_tenor = nodes_by_tenor
        self.market_by_tenor = market_by_tenor

    # ----- Butterfly per tenor -----

    def check_butterfly(self, epsilon_monotone: float = 1e-8, epsilon_convex: float = -1e-8) -> Dict[str, Dict]:
        """
        For each tenor:
        - Monotonicity: C(K_i) >= C(K_{i+1})  (call price decreases with strike)
        - Convexity: discrete second difference >= 0

        Returns per tenor:
            {
                'tenor': {
                    'monotone_ok': bool,
                    'convex_ok': bool,
                    'violations': [str messages]
                }
            }
        """
        result = {}
        for tenor, nodes in self.nodes_by_tenor.items():
            mkt = self.market_by_tenor[tenor]
            Ks = [n.K for n in nodes]
            vols = [n.vol for n in nodes]
            prices = [black_call_price(mkt.F, K, mkt.T, v, mkt.DF) for K, v in zip(Ks, vols)]

            violations = []

            # Monotonicity: non-increasing in strike
            monotone_ok = True
            for i in range(len(Ks) - 1):
                if prices[i] + epsilon_monotone < prices[i + 1]:
                    monotone_ok = False
                    violations.append(f"Monotonicity violated: C(K={Ks[i]:.6f}) < C(K_next={Ks[i+1]:.6f})")

            # Convexity: discrete second derivative >= 0
            convex_ok = True
            for i in range(1, len(Ks) - 1):
                K_prev, K_mid, K_next = Ks[i - 1], Ks[i], Ks[i + 1]
                C_prev, C_mid, C_next = prices[i - 1], prices[i], prices[i + 1]
                # Second finite difference scaled by spacing (nonuniform grid handling via divided differences)
                # Using standard test: sec_diff ≈ ( (C_next - C_mid)/(K_next-K_mid) - (C_mid - C_prev)/(K_mid-K_prev) )
                left_slope = (C_mid - C_prev) / (K_mid - K_prev)
                right_slope = (C_next - C_mid) / (K_next - K_mid)
                sec_diff = right_slope - left_slope
                if sec_diff < epsilon_convex:
                    convex_ok = False
                    violations.append(f"Convexity violated around K={K_mid:.6f}: second diff = {sec_diff:.6e}")

            result[tenor] = {
                'monotone_ok': monotone_ok,
                'convex_ok': convex_ok,
                'violations': violations
            }
        return result

    # ----- Calendar across tenors -----

    def _collect_strike_buckets(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Collect common strike buckets across tenors by node label.
        Returns mapping: label -> list of (T, K, vol) across tenors that have this node.
        """
        buckets: Dict[str, List[Tuple[float, float, float]]] = {}
        for tenor, nodes in self.nodes_by_tenor.items():
            T = self.market_by_tenor[tenor].T
            for n in nodes:
                buckets.setdefault(n.label, []).append((T, n.K, n.vol))
        # Sort by T within each bucket
        for label in buckets:
            buckets[label].sort(key=lambda x: x[0])
        return buckets

    def check_calendar(self, epsilon_var: float = -1e-12, epsilon_price: float = -1e-8,
                       forwards_by_tenor: Optional[Dict[str, float]] = None,
                       dfs_by_tenor: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        """
        Calendar no-arb:
        - Total variance monotonicity at fixed strike bucket: w(T) = σ^2(T) T should be non-decreasing in T.
        - Call price monotonicity at fixed (approximate) strike: C(T) should be non-decreasing in T, accounting for forward/DF per tenor.

        Returns per label:
            {
              'label': {
                 'variance_monotone_ok': bool,
                 'price_monotone_ok': bool,
                 'violations': [str]
              }
            }

        Note:
        - Buckets use node labels (ATM, 10C, 10P, 25C, 25P). Their strikes differ slightly across tenors due to delta-to-strike dependence on vol,
          but desk practice accepts price monotonicity checks on these canonical buckets as a pragmatic calendar sanity check.
        """
        buckets = self._collect_strike_buckets()
        result = {}

        # Build helper to find MarketTenor by T (we assume unique T per tenor key)
        # Map T -> (F, DF)
        T_to_mkt: Dict[float, Tuple[float, float]] = {}
        for tenor, m in self.market_by_tenor.items():
            T_to_mkt[m.T] = (m.F, m.DF)

        for label, series in buckets.items():
            violations = []
            # Variance monotonicity
            variances = [vol * vol * T for (T, _, vol) in series]
            variance_monotone_ok = True
            for i in range(len(variances) - 1):
                if variances[i + 1] + epsilon_var < variances[i]:
                    variance_monotone_ok = False
                    t0, t1 = series[i][0], series[i + 1][0]
                    violations.append(f"Variance decreased between T={t0:.6f} and T={t1:.6f}: w0={variances[i]:.6e}, w1={variances[i+1]:.6e}")

            # Price monotonicity across T using bucket strikes
            price_monotone_ok = True
            prices = []
            for (T, K, vol) in series:
                F, DF = T_to_mkt[T]
                prices.append(black_call_price(F, K, T, vol, DF))
            for i in range(len(prices) - 1):
                if prices[i + 1] + epsilon_price < prices[i]:
                    price_monotone_ok = False
                    t0, t1 = series[i][0], series[i + 1][0]
                    violations.append(f"Call price decreased between T={t0:.6f} and T={t1:.6f}: C0={prices[i]:.6e}, C1={prices[i+1]:.6e}")

            result[label] = {
                'variance_monotone_ok': variance_monotone_ok,
                'price_monotone_ok': price_monotone_ok,
                'violations': violations
            }
        return result


# -----------------------------
# Time interpolation of IV
# -----------------------------

class TimeInterpolator:
    """
    Time interpolation methods for implied volatility term structure.
    Provides:
    - Linear in total variance (w = σ^2 T)
    - Linear in volatility (σ)
    - Linear in log total variance (log w)
    """

    @staticmethod
    def linear_variance(T: float, T0: float, sigma0: float, T1: float, sigma1: float) -> float:
        """
        Interpolate linearly in total variance:
        w(T) = w0 + (w1 - w0) * ((T - T0) / (T1 - T0)), with w = σ^2 T, then σ(T) = sqrt(w(T)/T)

        This preserves calendar monotonicity if w0 <= w1 and T within [T0, T1].
        """
        if not (T0 < T < T1) or T <= 0:
            raise ValueError("Require T in (T0, T1) and T>0.")
        w0, w1 = sigma0 * sigma0 * T0, sigma1 * sigma1 * T1
        wT = w0 + (w1 - w0) * ((T - T0) / (T1 - T0))
        return math.sqrt(max(wT, 0.0) / T)

    @staticmethod
    def linear_volatility(T: float, T0: float, sigma0: float, T1: float, sigma1: float) -> float:
        """
        Interpolate linearly in volatility:
        σ(T) = σ0 + (σ1 - σ0) * ((T - T0) / (T1 - T0))

        Simple and common, but can violate calendar no-arb (w(T) non-monotone) when σ1 << σ0 and T grows.
        """
        if not (T0 < T < T1):
            raise ValueError("Require T in (T0, T1).")
        return sigma0 + (sigma1 - sigma0) * ((T - T0) / (T1 - T0))

    @staticmethod
    def linear_log_variance(T: float, T0: float, sigma0: float, T1: float, sigma1: float) -> float:
        """
        Interpolate linearly in log total variance:
        log w(T) = log w0 + (log w1 - log w0) * ((T - T0) / (T1 - T0)), w = σ^2 T

        Smooth and multiplicative; helps avoid negative variances and can temper curvature,
        but still relies on w0 <= w1 for no-arb within bracket.
        """
        if not (T0 < T < T1) or T <= 0 or T0 <= 0 or T1 <= 0:
            raise ValueError("Require T in (T0, T1) and positive T,T0,T1.")
        w0, w1 = sigma0 * sigma0 * T0, sigma1 * sigma1 * T1
        if w0 <= 0 or w1 <= 0:
            raise ValueError("Total variances must be positive.")
        lw0, lw1 = math.log(w0), math.log(w1)
        lwT = lw0 + (lw1 - lw0) * ((T - T0) / (T1 - T0))
        wT = math.exp(lwT)
        return math.sqrt(wT / T)


# -----------------------------
# Accuracy discussion helpers
# -----------------------------

def interpolation_accuracy_notes() -> str:
    """
    Returns a compact discussion of interpolation method accuracies.
    """
    return (
        "Accuracy and no-arbitrage considerations:\n"
        "- Linear in total variance: Typically the most robust for avoiding calendar arbitrage since w(T)=σ^2 T is additive "
        "over independent increments. If end-node variances are non-decreasing, the interpolated w(T) is non-decreasing inside the bracket. "
        "It aligns with diffusion scaling and produces sensible short-end behavior.\n"
        "- Linear in volatility: Intuitive but can break calendar no-arb; σ(T) linear does not guarantee w(T) monotonicity. "
        "Biases short maturities if σ1 >> σ0 (or vice versa), and may understate tail variance.\n"
        "- Linear in log variance: Multiplicative smoothing; guards against negative w and reduces sharp kinks. "
        "Still requires end-node monotonic variances to preserve no-arb. Often yields intermediate shapes between variance-linear and vol-linear.\n"
        "Practical guidance: Prefer variance-linear for production, enforce w-node monotonicity at calibration, and supplement with explicit "
        "calendar checks. Where aesthetics matter, log-variance can be a good compromise. Avoid vol-linear unless coupled with post-checks."
    )


# -----------------------------
# Example usage (for testing)
# -----------------------------

if __name__ == "__main__":
    # Example quotes and market data for two tenors
    q_1m = TenorQuote(ATM=0.12, RR25=-0.02, STR25=0.01, RR10=-0.03, STR10=0.015)
    q_3m = TenorQuote(ATM=0.11, RR25=-0.015, STR25=0.012, RR10=-0.025, STR10=0.016)

    mkt_1m = MarketTenor(T=1.0/12.0, F=1.25, DF=0.999)
    mkt_3m = MarketTenor(T=0.25, F=1.255, DF=0.997)

    # Build smiles
    sb_1m = SmileBuilder(q_1m, mkt_1m)
    sb_3m = SmileBuilder(q_3m, mkt_3m)

    nodes_1m = sb_1m.build_nodes()
    nodes_3m = sb_3m.build_nodes()

    nodes_by_tenor = {"1M": nodes_1m, "3M": nodes_3m}
    market_by_tenor = {"1M": mkt_1m, "3M": mkt_3m}

    # Arbitrage checks
    checker = ArbitrageChecker(nodes_by_tenor, market_by_tenor)
    butterfly_res = checker.check_butterfly()
    calendar_res = checker.check_calendar()

    print("Butterfly checks:")
    for tenor, res in butterfly_res.items():
        print(tenor, res)

    print("\nCalendar checks:")
    for label, res in calendar_res.items():
        print(label, res)

    # Interpolation demo between 1M and 3M ATM vols
    T_mid = 0.125  # 1.5M
    atm_mid_var_lin = TimeInterpolator.linear_variance(T_mid, mkt_1m.T, q_1m.ATM, mkt_3m.T, q_3m.ATM)
    atm_mid_vol_lin = TimeInterpolator.linear_volatility(T_mid, mkt_1m.T, q_1m.ATM, mkt_3m.T, q_3m.ATM)
    atm_mid_logw_lin = TimeInterpolator.linear_log_variance(T_mid, mkt_1m.T, q_1m.ATM, mkt_3m.T, q_3m.ATM)
    print("\nInterpolated ATM at T=0.125y:")
    print(f"Variance-linear: {atm_mid_var_lin:.6f}, Vol-linear: {atm_mid_vol_lin:.6f}, Log-variance-linear: {atm_mid_logw_lin:.6f}")

    print("\nNotes:")
    print(interpolation_accuracy_notes())