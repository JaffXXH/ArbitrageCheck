# ArbitrageCheck

################################################################
Butterfly Arbitrage (Positive Density): Ensures the call price is convex with respect to strike ((∂^2 C)/(∂K^2 )≥0). 
A violation implies negative probabilities, allowing arbitrage by buying a butterfly spread for a negative price (receive premium to hold it) .

Calendar Arbitrage (Monotonic Total Variance): Ensures total variance (σ^2 T) is non-decreasing with time to maturity. A violation allows arbitrage by selling a near-dated option and buying a far-dated option for a guaranteed profit .

Vertical (Spread) Arbitrage: Ensures call prices are non-increasing and convex with strike (C(K_1)≥C(K_2) for K_1<K_2). 
Violation allows arbitrage by buying a low-strike call and selling a high-strike call for a net credit 

################################################################
Arbitrage-free smile construction on FX option markets using Garman-Kohlhagen deltas and implied volatilities:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9483449

https://quant.stackexchange.com/questions/76366/option-pricing-for-illiquid-case/76367#76367
