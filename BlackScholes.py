import numpy as np
import scipy.stats as si

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the call option price using the Black-Scholes model.

    Parameters:
    S : float - Underlying asset price
    K : float - Strike price
    T : float - Time to maturity in years
    r : float - Risk-free rate (annual)
    sigma : float - Volatility (annual)
    
    Returns:
    float - Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    return call_price

def greeks_call(S, K, T, r, sigma):
    """
    Calculate the standard Greeks for a call option.

    Parameters:
    S, K, T, r, sigma : float - See black_scholes_call for descriptions
    
    Returns:
    dict: Dictionary with keys 'delta', 'gamma', 'theta', 'vega', 'rho'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Greeks calculations for the call option:
    delta = si.norm.cdf(d1)  # Call Delta
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        - (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * si.norm.cdf(d2)
    )
    vega = S * si.norm.pdf(d1) * np.sqrt(T)  # Vega per 1 unit change in volatility
    rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

def greeks_put(S, K, T, r, sigma):
    """
    Calculate the standard Greeks for a put option.

    Parameters:
    S : float - Underlying asset price
    K : float - Strike price
    T : float - Time to maturity in years
    r : float - Risk-free rate (annual)
    sigma : float - Volatility (annual)
    
    Returns:
    dict: Dictionary with keys 'delta', 'gamma', 'theta', 'vega', 'rho'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Greeks calculations for the put option:
    delta = si.norm.cdf(d1) - 1  # Put Delta: equivalent to N(d1) - 1
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
            + r * K * np.exp(-r * T) * si.norm.cdf(-d2)
    vega = S * si.norm.pdf(d1) * np.sqrt(T)
    rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

if __name__ == "__main__":
    # Test parameters
    S = 100      # Underlying asset price
    K = 100      # Strike price
    T = 1.0      # Time to maturity in years
    r = 0.05     # Annual risk-free rate
    sigma = 0.20 # Annual volatility

    call_price = black_scholes_call(S, K, T, r, sigma)
    call_greeks = greeks_call(S, K, T, r, sigma)
    put_greeks = greeks_put(S, K, T, r, sigma)

    print("Black-Scholes Call Price: {:.2f}".format(call_price))
    print("Call Option Greeks:")
    for greek, value in call_greeks.items():
        print("{}: {:.4f}".format(greek, value))
    
    print("\nPut Option Greeks:")
    for greek, value in put_greeks.items():
        print("{}: {:.4f}".format(greek, value))
