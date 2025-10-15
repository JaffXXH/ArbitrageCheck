import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

@dataclass
class FXVolSurface:
    """
    A class to represent an FX Implied Volatility Surface and check for static arbitrage.
    Delta convention: Spot delta (non-premium adjusted) :cite[1]
    """
    times_to_maturity: np.ndarray
    delta_grid: np.ndarray
    implied_vols: np.ndarray
    spot: float
    domestic_rate: float
    foreign_rate: float

    def __post_init__(self):
        self.strike_grid = self._calculate_strike_from_delta()
        self.call_prices = self._calculate_call_prices()
        self.arbitrage_violations = {
            'butterfly': [],
            'calendar': [],
            'vertical': []
        }

    def _calculate_strike_from_delta(self) -> np.ndarray:
        """Convert delta to strike price using Black-Scholes-Garman-Kohlhagen formula"""
        strikes = np.zeros_like(self.implied_vols)
        for i, T in enumerate(self.times_to_maturity):
            for j, delta in enumerate(self.delta_grid):
                vol = self.implied_vols[i, j]
                # Solve for strike that gives target delta
                strike_guess = self.spot * np.exp(
                    -norm.ppf(abs(delta)) * vol * np.sqrt(T) +
                    (self.domestic_rate - self.foreign_rate + 0.5 * vol**2) * T
                )
                strikes[i, j] = strike_guess
        return strikes

    def calculate_call_price(self, K: float, T: float, sigma: float) -> float:
        """Black-Scholes-Garman-Kohlhagen formula for FX call options :cite[1]"""
        if T <= 0:
            return max(self.spot - K, 0)
        
        F = self.spot * np.exp((self.domestic_rate - self.foreign_rate) * T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = np.exp(-self.domestic_rate * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        return max(call_price, 0)  # Ensure non-negativity

    def _calculate_call_prices(self) -> np.ndarray:
        """Calculate call prices from implied volatilities"""
        call_prices = np.zeros_like(self.implied_vols)
        for i, T in enumerate(self.times_to_maturity):
            for j, K in enumerate(self.strike_grid[i]):
                sigma = self.implied_vols[i, j]
                call_prices[i, j] = self.calculate_call_price(K, T, sigma)
        return call_prices

    def check_butterfly_arbitrage(self, tolerance: float = 1e-5) -> bool:
        """
        Check for butterfly arbitrage (positive density condition)
        Proposition: The second derivative of call price wrt strike must be non-negative
        d²C/dK² ≥ 0 for all K > 0 :cite[1]
        """
        arbitrage_free = True
        for i, T in enumerate(self.times_to_maturity):
            if T <= 0:
                continue
                
            strikes = self.strike_grid[i]
            call_prices = self.call_prices[i]
            
            # Sort by strike
            sort_idx = np.argsort(strikes)
            strikes_sorted = strikes[sort_idx]
            calls_sorted = call_prices[sort_idx]
            
            # Calculate second derivative using finite differences
            for j in range(1, len(strikes_sorted)-1):
                K_prev, K_curr, K_next = strikes_sorted[j-1], strikes_sorted[j], strikes_sorted[j+1]
                C_prev, C_curr, C_next = calls_sorted[j-1], calls_sorted[j], calls_sorted[j+1]
                
                # Second derivative approximation
                second_deriv = (C_next - 2*C_curr + C_prev) / ((K_next - K_curr) * (K_curr - K_prev))
                
                if second_deriv < -tolerance:
                    arbitrage_free = False
                    self.arbitrage_violations['butterfly'].append(
                        (T, K_curr, second_deriv, f"Butterfly arbitrage at T={T:.3f}, K={K_curr:.4f}")
                    )
        return arbitrage_free

    def check_calendar_arbitrage(self, tolerance: float = 1e-5) -> bool:
        """
        Check for calendar arbitrage (monotonic total variance)
        Proposition: Total variance σ²T must be non-decreasing with maturity :cite[1]
        If σ₂²T₂ < σ₁²T₁ for T₂ > T₁, calendar arbitrage exists
        """
        arbitrage_free = True
        
        # Calculate total variance for ATM (delta=0.5)
        atm_vols = []
        valid_times = []
        
        for i, T in enumerate(self.times_to_maturity):
            if T > 0:
                # Find vol for delta closest to 0.5 (ATM)
                atm_idx = np.argmin(np.abs(self.delta_grid - 0.5))
                atm_vol = self.implied_vols[i, atm_idx]
                atm_vols.append(atm_vol)
                valid_times.append(T)
        
        # Sort by maturity
        sort_idx = np.argsort(valid_times)
        sorted_times = np.array(valid_times)[sort_idx]
        sorted_vols = np.array(atm_vols)[sort_idx]
        
        # Check monotonicity of total variance
        for i in range(1, len(sorted_times)):
            total_var_prev = sorted_vols[i-1]**2 * sorted_times[i-1]
            total_var_curr = sorted_vols[i]**2 * sorted_times[i]
            
            if total_var_curr < total_var_prev - tolerance:
                arbitrage_free = False
                self.arbitrage_violations['calendar'].append(
                    (sorted_times[i], total_var_curr, 
                     f"Calendar arbitrage: σ²T decreases from {total_var_prev:.6f} to {total_var_curr:.6f} at T={sorted_times[i]:.3f}")
                )
        return arbitrage_free

    def check_vertical_arbitrage(self, tolerance: float = 1e-5) -> bool:
        """
        Check for vertical (spread) arbitrage
        Proposition: Call prices must be non-increasing with strike :cite[1]
        C(K₁) ≥ C(K₂) for K₁ < K₂
        """
        arbitrage_free = True
        for i, T in enumerate(self.times_to_maturity):
            strikes = self.strike_grid[i]
            call_prices = self.call_prices[i]
            
            # Sort by strike
            sort_idx = np.argsort(strikes)
            strikes_sorted = strikes[sort_idx]
            calls_sorted = call_prices[sort_idx]
            
            # Check monotonicity
            for j in range(1, len(strikes_sorted)):
                if calls_sorted[j] > calls_sorted[j-1] + tolerance:
                    arbitrage_free = False
                    self.arbitrage_violations['vertical'].append(
                        (T, strikes_sorted[j], 
                         f"Vertical arbitrage: C({strikes_sorted[j]:.4f}) = {calls_sorted[j]:.6f} > C({strikes_sorted[j-1]:.4f}) = {calls_sorted[j-1]:.6f} at T={T:.3f}")
                    )
        return arbitrage_free

    def run_all_arbitrage_checks(self) -> Dict[str, bool]:
        """Run all arbitrage checks and return results"""
        results = {
            'butterfly': self.check_butterfly_arbitrage(),
            'calendar': self.check_calendar_arbitrage(),
            'vertical': self.check_vertical_arbitrage()
        }
        return results

    def plot_arbitrage_analysis(self):
        """Plot the three required surfaces with arbitrage violations highlighted"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FX Implied Volatility Surface Arbitrage Analysis', fontsize=16)
        
        self._plot_call_price_surface(axes[0, 0])
        self._plot_implied_vol_by_log_moneyness(axes[0, 1])
        self._plot_implied_vol_by_delta(axes[1, 0])
        self._plot_arbitrage_summary(axes[1, 1])
        
        plt.tight_layout()
        plt.show()

    def _plot_call_price_surface(self, ax):
        """Plot call price surface"""
        T_grid, K_grid = np.meshgrid(self.times_to_maturity, self.strike_grid[0], indexing='ij')
        call_surface = np.zeros_like(T_grid)
        
        for i in range(len(self.times_to_maturity)):
            for j in range(len(self.strike_grid[0])):
                call_surface[i, j] = self.call_prices[i, j]
        
        contour = ax.contourf(T_grid, K_grid, call_surface, levels=20, cmap='viridis')
        ax.set_xlabel('Time to Maturity (T)')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Call Price Surface C(T,K)')
        plt.colorbar(contour, ax=ax, label='Call Price')
        
        # Highlight arbitrage points
        self._highlight_arbitrage_points(ax, 'butterfly', 'red', 'Butterfly')
        self._highlight_arbitrage_points(ax, 'vertical', 'yellow', 'Vertical')

    def _plot_implied_vol_by_log_moneyness(self, ax):
        """Plot implied volatility by log-moneyness"""
        for i, T in enumerate(self.times_to_maturity):
            if T > 0:
                F = self.spot * np.exp((self.domestic_rate - self.foreign_rate) * T)
                log_moneyness = np.log(self.strike_grid[i] / F)
                
                # Sort by log moneyness
                sort_idx = np.argsort(log_moneyness)
                ax.plot(log_moneyness[sort_idx], self.implied_vols[i][sort_idx], 
                       marker='o', label=f'T={T:.2f}')
        
        ax.set_xlabel('Log-Moneyness ln(K/F)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title('Implied Volatility by Log-Moneyness')
        ax.legend()
        ax.grid(True)

    def _plot_implied_vol_by_delta(self, ax):
        """Plot implied volatility by delta"""
        for i, T in enumerate(self.times_to_maturity):
            if T > 0:
                ax.plot(self.delta_grid, self.implied_vols[i], 
                       marker='s', label=f'T={T:.2f}')
        
        ax.set_xlabel('Delta (Spot)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title('Implied Volatility by Delta')
        ax.legend()
        ax.grid(True)

    def _plot_arbitrage_summary(self, ax):
        """Plot summary of arbitrage violations"""
        violation_types = list(self.arbitrage_violations.keys())
        violation_counts = [len(self.arbitrage_violations[v_type]) for v_type in violation_types]
        
        bars = ax.bar(violation_types, violation_counts, color=['red' if count > 0 else 'green' for count in violation_counts])
        ax.set_ylabel('Number of Violations')
        ax.set_title('Arbitrage Violations Summary')
        
        for bar, count in zip(bars, violation_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom')

    def _highlight_arbitrage_points(self, ax, violation_type: str, color: str, label: str):
        """Highlight arbitrage points on plots"""
        violations = self.arbitrage_violations[violation_type]
        for violation in violations:
            T, K, *_, = violation
            ax.plot(T, K, marker='x', color=color, markersize=10, markeredgewidth=3, label=label if 'x' not in ax.get_legend_handles_labels()[1] else "")

def create_sample_vol_surface() -> FXVolSurface:
    """
    Create a sample volatility surface for testing
    This represents a typical FX vol surface with some potential arbitrage
    """
    # Sample data: times, deltas, and implied volatilities
    times_to_maturity = np.array([0.1, 0.25, 0.5, 1.0])  # years
    delta_grid = np.array([0.1, 0.25, 0.5, 0.75, 0.9])  # spot deltas
    
    # Sample implied vol surface (volatility matrix)
    implied_vols = np.array([
        [0.125, 0.120, 0.118, 0.121, 0.128],  # 1M
        [0.130, 0.125, 0.322, 0.126, 0.133],  # 3M  
        [0.135, 0.128, 0.125, 0.129, 0.136],  # 6M
        [0.140, 0.132, 0.128, 0.133, 0.140]   # 1Y
    ])
    
    # Market parameters
    spot = 1.1000  # EURUSD
    domestic_rate = 0.05   # USD rate
    foreign_rate = 0.03    # EUR rate
    
    return FXVolSurface(times_to_maturity, delta_grid, implied_vols, 
                       spot, domestic_rate, foreign_rate)

# Example usage and testing
if __name__ == "__main__":
    # Create sample volatility surface
    vol_surface = create_sample_vol_surface()
    
    # Run all arbitrage checks
    print("Running arbitrage checks...")
    results = vol_surface.run_all_arbitrage_checks()
    
    # Print results
    print("\n=== ARBITRAGE CHECK RESULTS ===")
    for check_type, is_arbitrage_free in results.items():
        status = "PASS" if is_arbitrage_free else "FAIL"
        print(f"{check_type.upper():<15}: {status}")
        
        # Print violation details
        violations = vol_surface.arbitrage_violations[check_type]
        for violation in violations:
            print(f"  - {violation[-1]}")
    
    # Plot comprehensive analysis
    print("\nGenerating arbitrage analysis plots...")
    vol_surface.plot_arbitrage_analysis()
