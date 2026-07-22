This simulator calculates the optimal hedge for GPU compute rentals using forward curve and spot simulations.

1. PARAMETERS
  Choose GPU model (H100 / H200 / A100), number of GPUs, utilization (%), energy rate (¢/kWh), horizon (12–24 months), and min operating      margin %.

2. UNHEDGED FINANCIALS
  Every Month:
  
  gpu_hours = effective GPUs × 730
  revenue = gpu_hours × spot
  cost = depreciation + power + other opex
  operating margin = (revenue − cost) / revenue
  
3. FORWARD CURVE
  Synthetic $/GPU-hr curve from 1 to 24 months. Default shape is backwardation, because of GPU depreciation and new upgrade cycles:
  forward(months) ≈ spot₀ × (1 − 0.10 × months/12) × (1 + small noise)
  
  Breakeven spot is the min $/GPU-hr needed to hit the floor:
  spot ≥ [cost / (gpu_hours × (1 − margin))]
  
4. SIZING
  To protect the min operating profit from a near worst-case price drop, the model shorts enough GPU contracts to cover the single worst      month, making sure it never buys more protection than the total expected revenue.

  floor_rev = cost / (1 − margin)
  stress_spot = forward × exp(−z × vol × √(months/12))
  stress_shortfall = max(0, floor_rev − gpu_hours × stress_spot)
  n_t = stress_shortfall / ((forward − stress_spot) × contract_size)
  standing short = max(n_t) over the horizon
  
5. SELLING vs COVERING SHORT
  The model starts by opening the full short at the beginning based on the forward curve. Instead of constantly adjusting the position, it    looks at it once a month and estimates a near worst-case range for the future profit margins:

    SELL: If worst-case profit drops 2% below the absolute min margin. To avoid excessive trading, we only make further adjustments if                things get worse by at least another 1%.
    COVER: If the expected profit is above our min margin, we reduce the short for that month by 25%.
   
6. IMPACT OF HEDGE
  hedge PnL = n × 10,000 × (forward entry − realized spot)
  hedged income = operating profit + hedge PnL

  Spot Falls → Short Wins → Offsets Losses
  Spot Rises → Short Loses → Cost of Insurance
