import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, bisect
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.stats import norm
from typing import Dict, Tuple, List, Optional
import warnings

class VolatilitySurfaceArbitrage:
    """
    Comprehensive volatility surface analysis with arbitrage checks
    for butterfly and calendar spreads, supporting ATM, RR, STR quotes
    for 10 and 25 delta instruments.
    """
    
    def __init__(self, spot: float = 1.0, risk_free_rate: float = 0.02, dividend_yield: float = 0.01):
        self.spot = spot
        self.r = risk_free_rate
        self.q = dividend_yield
        self.surface_data = None
        self.call_prices_cache = {}
        
    def norm_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal"""
        return norm.cdf(x)
    
    def norm_pdf(self, x: float) -> float:
        """Probability density function for standard normal"""
        return norm.pdf(x)
    
    def black_scholes_delta(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate Black-Scholes delta"""
        d1 = (np.log(S/K) + (self.r - self.q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return np.exp(-self.q * T) * self.norm_cdf(d1)
        else:
            return np.exp(-self.q * T) * (self.norm_cdf(d1) - 1)
    
    def find_strike_from_delta(self, delta: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """Find strike price for given delta using root finding"""
        def objective(K):
            computed_delta = self.black_scholes_delta(self.spot, K, T, sigma, option_type)
            return computed_delta - delta
        
        # Reasonable strike bounds
        K_min = self.spot * 0.1
        K_max = self.spot * 3.0
        
        try:
            return bisect(objective, K_min, K_max, xtol=1e-6)
        except:
            # Fallback: use approximation for ATM
            if abs(delta - 0.5) < 0.1:
                return self.spot * np.exp((self.r - self.q) * T)
            else:
                return self.spot * (1 + (1 - 2 * delta) * sigma * np.sqrt(T))
    
    def black_scholes_price(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price"""
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
            
        d1 = (np.log(S/K) + (self.r - self.q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-self.q * T) * self.norm_cdf(d1) - K * np.exp(-self.r * T) * self.norm_cdf(d2)
        else:
            price = K * np.exp(-self.r * T) * self.norm_cdf(-d2) - S * np.exp(-self.q * T) * self.norm_cdf(-d1)
            
        return max(price, 0)  # Ensure non-negative prices
    
    def setup_volatility_surface(self, tenors: List[float], 
                                atm_vols: List[float],
                                rr_25d: List[float], str_25d: List[float],
                                rr_10d: List[float], str_10d: List[float]) -> Dict:
        """
        Setup volatility surface from market quotes
        
        Parameters:
        - tenors: List of time to maturities in years
        - atm_vols: ATM volatilities for each tenor
        - rr_25d: 25-delta risk reversals
        - str_25d: 25-delta strangles  
        - rr_10d: 10-delta risk reversals
        - str_10d: 10-delta strangles
        """
        
        surface = {}
        
        for i, T in enumerate(tenors):
            surface[T] = {
                'ATM': atm_vols[i],
                'RR_25d': rr_25d[i],
                'STR_25d': str_25d[i],
                'RR_10d': rr_10d[i],
                'STR_10d': str_10d[i],
                
                # Calculate call and put volatilities for each delta
                'call_25d': atm_vols[i] + str_25d[i] + 0.5 * rr_25d[i],
                'put_25d': atm_vols[i] + str_25d[i] - 0.5 * rr_25d[i],
                'call_10d': atm_vols[i] + str_10d[i] + 0.5 * rr_10d[i],
                'put_10d': atm_vols[i] + str_10d[i] - 0.5 * rr_10d[i]
            }
            
            # Calculate strikes for each delta point
            surface[T]['strikes'] = {
                'ATM': self.find_strike_from_delta(0.5, T, atm_vols[i], 'call'),
                'call_25d': self.find_strike_from_delta(0.25, T, surface[T]['call_25d'], 'call'),
                'put_25d': self.find_strike_from_delta(0.25, T, surface[T]['put_25d'], 'put'),
                'call_10d': self.find_strike_from_delta(0.10, T, surface[T]['call_10d'], 'call'),
                'put_10d': self.find_strike_from_delta(0.10, T, surface[T]['put_10d'], 'put')
            }
        
        self.surface_data = surface
        self.tenors = tenors
        return surface
    
    def calculate_call_prices(self, T: float, strikes: List[float]) -> np.ndarray:
        """Calculate call prices for given tenor and strikes"""
        key = (T, tuple(strikes))
        if key in self.call_prices_cache:
            return self.call_prices_cache[key]
        
        # Get volatility at these strikes using interpolation
        vol_surface_T = self.surface_data[T]
        known_strikes = list(vol_surface_T['strikes'].values())
        known_vols = [
            vol_surface_T['ATM'],
            vol_surface_T['call_25d'],
            vol_surface_T['put_25d'], 
            vol_surface_T['call_10d'],
            vol_surface_T['put_10d']
        ]
        
        # Sort by strike
        sorted_indices = np.argsort(known_strikes)
        known_strikes_sorted = [known_strikes[i] for i in sorted_indices]
        known_vols_sorted = [known_vols[i] for i in sorted_indices]
        
        # Interpolate volatility smile
        try:
            vol_interpolator = CubicSpline(known_strikes_sorted, known_vols_sorted, extrapolate=False)
            interp_vols = vol_interpolator(strikes)
        except:
            # Fallback to linear interpolation
            interp_vols = np.interp(strikes, known_strikes_sorted, known_vols_sorted)
        
        # Calculate call prices
        call_prices = np.array([self.black_scholes_price(self.spot, K, T, sigma, 'call') 
                              for K, sigma in zip(strikes, interp_vols)])
        
        self.call_prices_cache[key] = call_prices
        return call_prices
    
    def check_butterfly_arbitrage_single_tenor(self, T: float, n_points: int = 100) -> Tuple[bool, float, Dict]:
        """
        Check for butterfly arbitrage in a single tenor
        
        Butterfly arbitrage exists if call price function is not convex in strike
        i.e., if the second derivative w.r.t. strike is negative
        """
        if T not in self.surface_data:
            raise ValueError(f"Tenor {T} not found in surface data")
        
        # Create a dense strike grid
        vol_surface_T = self.surface_data[T]
        min_strike = min(vol_surface_T['strikes'].values()) * 0.8
        max_strike = max(vol_surface_T['strikes'].values()) * 1.2
        strikes = np.linspace(min_strike, max_strike, n_points)
        
        # Calculate call prices
        call_prices = self.calculate_call_prices(T, strikes)
        
        # Calculate second derivative (convexity)
        # Using finite differences
        dK = strikes[1] - strikes[0]
        second_deriv = np.gradient(np.gradient(call_prices, dK), dK)
        
        # Check for negative convexity (butterfly arbitrage)
        min_convexity = np.min(second_deriv)
        arbitrage_detected = min_convexity < -1e-6
        
        # Calculate butterfly spread prices at key points
        butterfly_prices = {}
        key_strikes = list(vol_surface_T['strikes'].values())
        key_strike_names = list(vol_surface_T['strikes'].keys())
        
        for i, K in enumerate(key_strikes):
            if i == 0 or i == len(key_strikes) - 1:
                continue
                
            K_prev = key_strikes[i-1]
            K_next = key_strikes[i+1]
            
            # Butterfly spread: long 1 call at K_prev, long 1 call at K_next, short 2 calls at K
            C_prev = self.black_scholes_price(self.spot, K_prev, T, 
                                            self._get_volatility_at_strike(T, K_prev), 'call')
            C = self.black_scholes_price(self.spot, K, T, 
                                       self._get_volatility_at_strike(T, K), 'call')
            C_next = self.black_scholes_price(self.spot, K_next, T, 
                                            self._get_volatility_at_strike(T, K_next), 'call')
            
            butterfly_price = C_prev + C_next - 2 * C
            butterfly_prices[key_strike_names[i]] = butterfly_price
        
        details = {
            'min_convexity': min_convexity,
            'strikes': strikes,
            'call_prices': call_prices,
            'second_derivative': second_deriv,
            'butterfly_prices': butterfly_prices
        }
        
        return not arbitrage_detected, min_convexity, details
    
    def _get_volatility_at_strike(self, T: float, K: float) -> float:
        """Get interpolated volatility for given tenor and strike"""
        vol_surface_T = self.surface_data[T]
        known_strikes = list(vol_surface_T['strikes'].values())
        known_vols = [
            vol_surface_T['ATM'],
            vol_surface_T['call_25d'],
            vol_surface_T['put_25d'],
            vol_surface_T['call_10d'],
            vol_surface_T['put_10d']
        ]
        
        # Sort and interpolate
        sorted_indices = np.argsort(known_strikes)
        known_strikes_sorted = [known_strikes[i] for i in sorted_indices]
        known_vols_sorted = [known_vols[i] for i in sorted_indices]
        
        return np.interp(K, known_strikes_sorted, known_vols_sorted)
    
    def check_calendar_arbitrage(self, delta: float = 0.25, option_type: str = 'call') -> Tuple[bool, List[Tuple]]:
        """
        Check for calendar spread arbitrage
        
        Calendar arbitrage exists if call price is not increasing in maturity
        for fixed strike, or if total variance is not increasing in time
        """
        violations = []
        
        # Check total variance monotonicity
        total_variances = []
        for T in self.tenors:
            if option_type == 'call':
                if delta == 0.25:
                    vol = self.surface_data[T]['call_25d']
                elif delta == 0.10:
                    vol = self.surface_data[T]['call_10d']
                else:
                    vol = self.surface_data[T]['ATM']
            else:
                if delta == 0.25:
                    vol = self.surface_data[T]['put_25d']
                elif delta == 0.10:
                    vol = self.surface_data[T]['put_10d']
                else:
                    vol = self.surface_data[T]['ATM']
            
            total_variances.append(vol**2 * T)
        
        # Check if total variance is increasing
        for i in range(1, len(total_variances)):
            if total_variances[i] < total_variances[i-1]:
                violations.append((self.tenors[i-1], self.tenors[i], 
                                 total_variances[i-1], total_variances[i]))
        
        # Check call price monotonicity for fixed moneyness
        for strike_type in ['ATM', 'call_25d', 'put_25d', 'call_10d', 'put_10d']:
            prev_price = None
            prev_T = None
            
            for T in sorted(self.tenors):
                K = self.surface_data[T]['strikes'][strike_type]
                vol = self._get_volatility_for_strike_type(T, strike_type)
                price = self.black_scholes_price(self.spot, K, T, vol, 'call')
                
                if prev_price is not None and price < prev_price:
                    violations.append((prev_T, T, f"{strike_type}", prev_price, price))
                
                prev_price = price
                prev_T = T
        
        return len(violations) == 0, violations
    
    def _get_volatility_for_strike_type(self, T: float, strike_type: str) -> float:
        """Get volatility for specific strike type"""
        if strike_type == 'ATM':
            return self.surface_data[T]['ATM']
        elif strike_type == 'call_25d':
            return self.surface_data[T]['call_25d']
        elif strike_type == 'put_25d':
            return self.surface_data[T]['put_25d']
        elif strike_type == 'call_10d':
            return self.surface_data[T]['call_10d']
        elif strike_type == 'put_10d':
            return self.surface_data[T]['put_10d']
        else:
            raise ValueError(f"Unknown strike type: {strike_type}")
    
    def interpolate_volatility_time(self, target_tenor: float, method: str = 'linear') -> Dict:
        """
        Interpolate volatility surface in time dimension
        
        Methods:
        - 'linear': Linear interpolation in volatility space
        - 'cubic': Cubic spline interpolation in volatility space  
        - 'total_variance': Linear interpolation in total variance space
        - 'pchip': Piecewise Cubic Hermite Interpolating Polynomial
        """
        
        tenors = sorted(self.tenors)
        
        if target_tenor < min(tenors) or target_tenor > max(tenors):
            warnings.warn(f"Target tenor {target_tenor} outside available range [{min(tenors)}, {max(tenors)}]")
        
        result = {}
        vol_types = ['ATM', 'call_25d', 'put_25d', 'call_10d', 'put_10d']
        
        for vol_type in vol_types:
            vols = [self.surface_data[T][vol_type] for T in tenors]
            
            if method == 'linear':
                interp_vol = np.interp(target_tenor, tenors, vols)
            elif method == 'cubic':
                spline = CubicSpline(tenors, vols)
                interp_vol = spline(target_tenor)
            elif method == 'total_variance':
                # Interpolate in total variance space
                total_vars = [vol**2 * T for vol, T in zip(vols, tenors)]
                interp_total_var = np.interp(target_tenor, tenors, total_vars)
                interp_vol = np.sqrt(interp_total_var / target_tenor)
            elif method == 'pchip':
                pchip = PchipInterpolator(tenors, vols)
                interp_vol = pchip(target_tenor)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            result[vol_type] = interp_vol
        
        # Interpolate RR and STR similarly
        for rr_str_type in ['RR_25d', 'STR_25d', 'RR_10d', 'STR_10d']:
            values = [self.surface_data[T][rr_str_type] for T in tenors]
            
            if method == 'linear':
                result[rr_str_type] = np.interp(target_tenor, tenors, values)
            elif method == 'cubic':
                spline = CubicSpline(tenors, values)
                result[rr_str_type] = spline(target_tenor)
            elif method == 'total_variance':
                # For RR/STR, use linear interpolation as they are differences
                result[rr_str_type] = np.interp(target_tenor, tenors, values)
            elif method == 'pchip':
                pchip = PchipInterpolator(tenors, values)
                result[rr_str_type] = pchip(target_tenor)
        
        return result
    
    def evaluate_interpolation_accuracy(self, method: str = 'linear') -> Dict:
        """
        Evaluate interpolation accuracy using leave-one-out cross-validation
        """
        errors = {vol_type: [] for vol_type in ['ATM', 'call_25d', 'put_25d', 'call_10d', 'put_10d']}
        
        for i, test_tenor in enumerate(self.tenors):
            # Training tenors (all except test_tenor)
            train_tenors = [T for j, T in enumerate(self.tenors) if j != i]
            
            # Create temporary surface without test tenor
            temp_data = {}
            for T in train_tenors:
                temp_data[T] = self.surface_data[T]
            
            # Store original data
            original_data = self.surface_data.copy()
            self.surface_data = temp_data
            self.tenors = train_tenors
            
            try:
                # Interpolate at test tenor
                interp_result = self.interpolate_volatility_time(test_tenor, method)
                
                # Calculate errors
                for vol_type in errors.keys():
                    true_vol = original_data[test_tenor][vol_type]
                    interp_vol = interp_result[vol_type]
                    error = abs(interp_vol - true_vol)
                    errors[vol_type].append(error)
                    
            finally:
                # Restore original data
                self.surface_data = original_data
                self.tenors = sorted(original_data.keys())
        
        # Calculate statistics
        stats = {}
        for vol_type, error_list in errors.items():
            if error_list:
                stats[vol_type] = {
                    'mean_error': np.mean(error_list),
                    'max_error': np.max(error_list),
                    'rmse': np.sqrt(np.mean(np.array(error_list)**2))
                }
        
        return stats
    
    def comprehensive_arbitrage_check(self) -> Dict:
        """Run comprehensive arbitrage checks across entire surface"""
        results = {}
        
        # Check butterfly arbitrage for each tenor
        butterfly_results = {}
        for T in self.tenors:
            is_arb_free, min_conv, details = self.check_butterfly_arbitrage_single_tenor(T)
            butterfly_results[T] = {
                'arbitrage_free': is_arb_free,
                'min_convexity': min_conv,
                'butterfly_prices': details['butterfly_prices']
            }
        
        # Check calendar arbitrage
        calendar_arb_free, calendar_violations = self.check_calendar_arbitrage()
        
        results['butterfly_arbitrage'] = butterfly_results
        results['calendar_arbitrage'] = {
            'arbitrage_free': calendar_arb_free,
            'violations': calendar_violations
        }
        
        # Overall assessment
        all_butterfly_free = all(r['arbitrage_free'] for r in butterfly_results.values())
        results['overall_arbitrage_free'] = all_butterfly_free and calendar_arb_free
        
        return results
    
    def plot_volatility_surface(self):
        """Plot the volatility surface"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot volatility smiles for each tenor
        for T in self.tenors:
            strikes_data = self.surface_data[T]['strikes']
            vols_data = [
                self.surface_data[T]['ATM'],
                self.surface_data[T]['call_25d'],
                self.surface_data[T]['put_25d'],
                self.surface_data[T]['call_10d'],
                self.surface_data[T]['put_10d']
            ]
            
            strike_names = ['ATM', '25d Call', '25d Put', '10d Call', '10d Put']
            
            ax1.plot(strike_names, vols_data, 'o-', label=f'T={T}y', markersize=6)
        
        ax1.set_title('Volatility Smiles by Tenor')
        ax1.set_ylabel('Implied Volatility')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot term structure for each volatility type
        vol_types = ['ATM', 'call_25d', 'put_25d']
        for vol_type in vol_types:
            vols = [self.surface_data[T][vol_type] for T in self.tenors]
            ax2.plot(self.tenors, vols, 's-', label=vol_type, markersize=6)
        
        ax2.set_title('Volatility Term Structure')
        ax2.set_xlabel('Tenor (Years)')
        ax2.set_ylabel('Implied Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot total variance
        for vol_type in vol_types:
            total_vars = [self.surface_data[T][vol_type]**2 * T for T in self.tenors]
            ax3.plot(self.tenors, total_vars, '^-', label=vol_type, markersize=6)
        
        ax3.set_title('Total Variance Term Structure')
        ax3.set_xlabel('Tenor (Years)')
        ax3.set_ylabel('Total Variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot risk reversals and strangles
        ax4.plot(self.tenors, [self.surface_data[T]['RR_25d'] for T in self.tenors], 
                'o-', label='25d RR', markersize=6)
        ax4.plot(self.tenors, [self.surface_data[T]['STR_25d'] for T in self.tenors],
                's-', label='25d STR', markersize=6)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_title('Risk Reversals and Strangles')
        ax4.set_xlabel('Tenor (Years)')
        ax4.set_ylabel('Volatility')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def demonstrate_arbitrage_checks():
    """Demonstrate the arbitrage checking functionality"""
    
    # Create sample market data
    tenors = [0.1, 0.25, 0.5, 1.0, 2.0]  # years
    
    # Sample market quotes (real data would come from market sources)
    atm_vols = [0.15, 0.16, 0.165, 0.17, 0.175]
    rr_25d = [0.02, 0.025, 0.03, 0.035, 0.04]  # Typically positive for equity indices
    str_25d = [0.01, 0.012, 0.014, 0.016, 0.018]
    rr_10d = [0.04, 0.045, 0.05, 0.055, 0.06]
    str_10d = [0.02, 0.022, 0.024, 0.026, 0.028]
    
    # Initialize volatility surface
    vol_surface = VolatilitySurfaceArbitrage(spot=100.0, risk_free_rate=0.03, dividend_yield=0.02)
    surface_data = vol_surface.setup_volatility_surface(tenors, atm_vols, rr_25d, str_25d, rr_10d, str_10d)
    
    print("=== Volatility Surface Arbitrage Analysis ===\n")
    
    # Run comprehensive arbitrage checks
    arbitrage_results = vol_surface.comprehensive_arbitrage_check()
    
    print("1. BUTTERFLY ARBITRAGE CHECK:")
    for T, result in arbitrage_results['butterfly_arbitrage'].items():
        status = "✓ FREE" if result['arbitrage_free'] else "✗ ARBITRAGE DETECTED"
        print(f"   Tenor {T}y: {status} (min convexity: {result['min_convexity']:.6f})")
        
        # Show butterfly prices
        for strike_type, price in result['butterfly_prices'].items():
            if price < -1e-6:
                print(f"     WARNING: Negative butterfly at {strike_type}: {price:.6f}")
    
    print(f"\n2. CALENDAR ARBITRAGE CHECK:")
    calendar_status = "✓ FREE" if arbitrage_results['calendar_arbitrage']['arbitrage_free'] else "✗ ARBITRAGE DETECTED"
    print(f"   Calendar spreads: {calendar_status}")
    
    if arbitrage_results['calendar_arbitrage']['violations']:
        print("   Violations found:")
        for violation in arbitrage_results['calendar_arbitrage']['violations']:
            print(f"     {violation}")
    
    print(f"\n3. OVERALL ASSESSMENT:")
    overall_status = "ARBITRAGE-FREE" if arbitrage_results['overall_arbitrage_free'] else "ARBITRAGE DETECTED"
    print(f"   Surface is: {overall_status}")
    
    # Test time interpolation
    print(f"\n4. TIME INTERPOLATION ACCURACY:")
    methods = ['linear', 'cubic', 'total_variance', 'pchip']
    
    for method in methods:
        stats = vol_surface.evaluate_interpolation_accuracy(method)
        print(f"\n   {method.upper()} method:")
        for vol_type, error_stats in stats.items():
            print(f"     {vol_type}: RMSE={error_stats['rmse']:.6f}, Max={error_stats['max_error']:.6f}")
    
    # Demonstrate interpolation at specific tenor
    print(f"\n5. INTERPOLATION AT T=0.75 YEARS:")
    target_tenor = 0.75
    for method in methods:
        interp_result = vol_surface.interpolate_volatility_time(target_tenor, method)
        print(f"\n   {method.upper()}:")
        print(f"     ATM: {interp_result['ATM']:.4f}, 25d Call: {interp_result['call_25d']:.4f}")
    
    # Plot the surface
    print(f"\n6. GENERATING VOLATILITY SURFACE PLOTS...")
    vol_surface.plot_volatility_surface()
    
    return vol_surface, arbitrage_results

if __name__ == "__main__":
    # Run the demonstration
    vol_surface, results = demonstrate_arbitrage_checks()