import sentimentMapping

# Helper functions for Greek classification

def classify_delta(delta, call_put="C"):
    if call_put == "C":
        if 0.5 <= delta <= 1:
            return "good"
        elif 0.3 < delta < 0.5:
            return "neutral"
        else:
            return "bad"
    else:  # Put
        if -1 <= delta <= -0.5:
            return "good"
        elif -0.5 < delta < -0.3:
            return "neutral"
        else:
            return "bad"

def classify_gamma(gamma):
    if gamma > 0.1:
        return "good"
    elif 0.05 < gamma <= 0.1:
        return "neutral"
    else:
        return "bad"

def classify_theta(theta, call_put="C"):
    # For long options, theta is usually negative
    if call_put == "C" or call_put == "P":
        if theta > -0.05:
            return "neutral" if theta > 0 else "good"
        elif theta <= -0.05:
            return "bad"
    # For short options, positive theta is good
    if theta > 0.05:
        return "good"
    elif 0 < theta <= 0.05:
        return "neutral"
    else:
        return "bad"

def classify_vega(vega):
    if vega > 0.1:
        return "good"
    elif 0.05 < vega <= 0.1:
        return "neutral"
    else:
        return "bad"

def interpret_delta(delta, call_put="C"):
    if call_put == "C":
        if delta > 0.9:
            return f"Delta ({delta:.2f}) is very close to one, indicating very high sensitivity to the underlying price and a high probability of being in the money at expiration."
        elif delta > 0.5:
            return f"Delta ({delta:.2f}) is above 0.5, showing strong sensitivity to the underlying price and a good chance of being in the money."
        elif delta > 0.3:
            return f"Delta ({delta:.2f}) is moderate, so the option price moves somewhat with the underlying, but the probability of being in the money is lower."
        else:
            return f"Delta ({delta:.2f}) is low, indicating little sensitivity to the underlying price and a low probability of being in the money."
    else:
        if delta < -0.9:
            return f"Delta ({delta:.2f}) is very close to minus one, indicating very high sensitivity to the underlying price decrease and a high probability of being in the money at expiration."
        elif delta < -0.5:
            return f"Delta ({delta:.2f}) is below -0.5, showing strong sensitivity to the underlying price decrease and a good chance of being in the money."
        elif delta < -0.3:
            return f"Delta ({delta:.2f}) is moderate, so the option price moves somewhat with the underlying, but the probability of being in the money is lower."
        else:
            return f"Delta ({delta:.2f}) is close to zero, indicating little sensitivity to the underlying price and a low probability of being in the money."

def interpret_gamma(gamma):
    if gamma > 0.2:
        return f"Gamma ({gamma:.4f}) is high, meaning delta will change significantly with underlying price movements, offering potential for increased profitability if the price moves favorably."
    elif gamma > 0.1:
        return f"Gamma ({gamma:.4f}) is moderate to high, so delta will adjust noticeably as the underlying price changes."
    elif gamma > 0.05:
        return f"Gamma ({gamma:.4f}) is moderate, indicating delta changes at a reasonable pace."
    else:
        return f"Gamma ({gamma:.4f}) is low, so delta changes slowly as the underlying price moves."

def interpret_theta(theta, call_put="C"):
    if theta > 0.05:
        return f"Theta ({theta:.4f}) is positive and high, which is beneficial for short options as you profit from time decay."
    elif theta > 0:
        return f"Theta ({theta:.4f}) is positive, so time decay is working in your favor if you are short the option."
    elif theta > -0.05:
        return f"Theta ({theta:.4f}) is slightly negative, so time decay is having a small impact on the option's value."
    else:
        return f"Theta ({theta:.4f}) is strongly negative, indicating the option is losing value quickly due to time decay, which is a concern for long positions."

def interpret_vega(vega):
    if vega > 0.1:
        return f"Vega ({vega:.4f}) is high, so the option's value is very sensitive to changes in implied volatility. If volatility rises, your option will benefit significantly."
    elif vega > 0.05:
        return f"Vega ({vega:.4f}) is moderate, so the option's value will change noticeably with volatility shifts."
    else:
        return f"Vega ({vega:.4f}) is low, so the option's value is not very sensitive to changes in implied volatility."

def generate_analysis(iv_new, greeks, call_put="C"):
    """
    Generate a paragraph of textual analysis based on the new implied volatility and option Greeks.
    """
    # Classify each Greek
    delta_class = classify_delta(greeks['delta'], call_put)
    gamma_class = classify_gamma(greeks['gamma'])
    theta_class = classify_theta(greeks['theta'], call_put)
    vega_class = classify_vega(greeks['vega'])
    
    classes = [delta_class, gamma_class, theta_class, vega_class]
    good_count = classes.count("good")
    neutral_count = classes.count("neutral")
    bad_count = classes.count("bad")

    if good_count > max(neutral_count, bad_count):
        recommendation = "Based on the current Greeks, you should go through with the trade."
    else:
        recommendation = "Based on the current Greeks, you should NOT go through with the trade."

    analysis = (
        f"The adjusted implied volatility (IV) is {iv_new:.4f}, reflecting the market's updated expectations for price movement. "
        f"{interpret_delta(greeks['delta'], call_put)} "
        f"{interpret_gamma(greeks['gamma'])} "
        f"{interpret_theta(greeks['theta'], call_put)} "
        f"{interpret_vega(greeks['vega'])} "
        f"{recommendation} "
    )
    return analysis

if __name__ == "__main__":
    pass 