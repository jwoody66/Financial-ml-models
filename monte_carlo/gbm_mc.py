import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


##This first model is GBM monte carlo simulation
####This acts as the baseline for monte carlo sim, well behaved, stationary and gaussian this gives uncertainty bands and probability of loss. 

##From this simulation, i can conclude the following: Assuming the returns are normally distributed with a constant volatility, the 60 Day future price has a 90% probability of being between x and y.
##As the bands in this simulation are widening, the uncertainty is increasing 

def fetch_prices(ticker: str, start="2018-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker or connection.") 
    return df["Close"].dropna() 


def estimate_mu_sigma(prices: pd.Series):
    log_returns = np.log(prices / prices.shift(1)).dropna() ##shifts the series by 1, gives the daily price ratio, converts to a log return, dropna() removes first Nan as day 1 has no prior day
    mu = float(log_returns.mean()) ##average log return daily drift estimate 
    sigma = float(log_returns.std(ddof=1))  # sample std
    return mu, sigma 


def simulate_gbm_paths(S0: float, mu: float, sigma: float, n_days: int, n_sims: int, seed: int = 42):
    rng = np.random.default_rng(seed) ##Repeatable

    # Z ~ N(0,1) shocks
    Z = rng.standard_normal(size=(n_days, n_sims))

    # Discrete-time GBM log returns look into GBM monte carlo and see why this works
    # r_t = (mu - 0.5*sigma^2) + sigma*Z_t
    r = (mu - 0.5 * sigma**2) + sigma * Z  ##This is the model for the MC, sigma* Z adds randomness scaled by volatility.
    # Convert returns into price paths:
    # S_t = S0 * exp(cumsum(r))
    log_price_rel = np.vstack([np.zeros((1, n_sims)), np.cumsum(r, axis=0)])
    price_paths = S0 * np.exp(log_price_rel)
    return price_paths


def summarize(price_paths: np.ndarray) -> dict:
    """
    Summary stats for the distribution of final prices.
    """
    final_prices = price_paths[-1]
    return {
        "expected_final": float(final_prices.mean()),
        "median_final": float(np.median(final_prices)),
        "p05_final": float(np.percentile(final_prices, 5)),
        "p95_final": float(np.percentile(final_prices, 95)),
        "prob_loss": float(np.mean(final_prices < price_paths[0, 0]))
    }


def plot_paths(price_paths: np.ndarray, ticker: str, n_plot: int = 50) -> None:
    plt.figure()
    plt.plot(price_paths[:, :n_plot])
    plt.title(f"Monte Carlo GBM Paths: {ticker}")
    plt.xlabel("Trading Day")
    plt.ylabel("Price")
    plt.show()


def plot_fan_chart(price_paths: np.ndarray, ticker: str) -> None:
    p5 = np.percentile(price_paths, 5, axis=1)
    p50 = np.percentile(price_paths, 50, axis=1)
    p95 = np.percentile(price_paths, 95, axis=1)

    plt.figure()
    plt.plot(p50)
    plt.fill_between(np.arange(len(p50)), p5, p95, alpha=0.2)
    plt.title(f"Fan Chart (5/50/95): {ticker}")
    plt.xlabel("Trading Day")
    plt.ylabel("Price")
    plt.show()


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2018-01-01"
    n_days = 60       # forecast horizon (trading days)
    n_sims = 5000     # number of simulated paths
    seed = 42

    prices = fetch_prices(ticker, start=start_date)
    mu, sigma = estimate_mu_sigma(prices)

    S0 = float(prices.iloc[-1])
    paths = simulate_gbm_paths(S0, mu, sigma, n_days=n_days, n_sims=n_sims, seed=seed)
    stats = summarize(paths)

    print(f"Ticker: {ticker}")
    print(f"Last price (S0): {S0:.2f}")
    print(f"Estimated mu (daily): {mu:.6f}")
    print(f"Estimated sigma (daily): {sigma:.6f}")
    print("\nDistribution of final price after", n_days, "days:")
    print(f"  Expected: {stats['expected_final']:.2f}")
    print(f"  Median:   {stats['median_final']:.2f}")
    print(f"  5th pct:  {stats['p05_final']:.2f}")
    print(f"  95th pct: {stats['p95_final']:.2f}")
    print(f"  Prob(final < S0): {stats['prob_loss']:.3f}")

    plot_paths(paths, ticker, n_plot=50)
    plot_fan_chart(paths, ticker)
