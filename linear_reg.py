import numpy as np 
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


##Get data from yahoo finance
def fetch_prices(ticker: str, start = "2018-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df = df.dropna()
    return df

##Create features
def make_features(df:pd.DataFrame):
    out = df.copy()
    #log returns
    out["ret_1"] = np.log(out["Close"] / out["Close"].shift(1))
    ##Lag returns
    out["ret_lag1"] = out["ret_1"].shift(1)
    out["ret_lag2"] = out["ret_1"].shift(2)
    out["ret_lag5"] = out["ret_1"].shift(5)

    # Rolling volatility (predictors)
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    # Trend features (predictors)
    sma_10 = out["Close"].rolling(10).mean()
    sma_50 = out["Close"].rolling(50).mean()
    out["sma10_dist"] = (out["Close"] - sma_10) / out["Close"]
    out["sma50_dist"] = (out["Close"] - sma_50) / out["Close"]

    return out

def make_target(df: pd.DataFrame):
    out = df.copy()
    out["y_next_ret"] = out["ret_1"].shift(-1)
    return out

def walk_forward_rmse(X, y, model, n_splits=5):  ##Finding RootMeanSquaredError on past information
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        rmses.append(rmse)

    return float(np.mean(rmses)), rmses

def backtest_signals(df, preds, fee_bps=1.0, threshold=0.0):
    signal = (preds > threshold).astype(int)

    strat_ret = signal * df["y_next_ret"].values

    # simple transaction cost:
    turnover = np.abs(np.diff(signal, prepend=signal[0]))
    cost = turnover * (fee_bps / 10000.0)

    net_ret = strat_ret - cost
    equity = (1 + net_ret).cumprod()

    return pd.DataFrame({
        "pred": preds,
        "signal": signal,
        "net_ret": net_ret,
        "equity": equity
    }, index=df.index)

def walk_forward_predict(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model.fit(X_train, y_train)
        preds.iloc[test_idx] = model.predict(X_test)

    return preds

def sharpe(returns, periods=252):
    r = pd.Series(returns).dropna()
    if r.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * r.mean() / r.std())

def max_drawdown(equity):
    e = pd.Series(equity).dropna()
    peak = e.cummax()
    dd = (e / peak) - 1.0
    return float(dd.min())

def run_linear(ticker="AAPL", start="2018-01-01", fee_bps=1.0, threshold=0.0, n_splits=5):
    df = fetch_prices(ticker, start=start)
    df = make_features(df)
    df = make_target(df)
    df = df.dropna()

    feature_cols = ["ret_lag1","ret_lag2","ret_lag5","vol_10","vol_20","sma10_dist","sma50_dist"]
    X = df[feature_cols]
    y = df["y_next_ret"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    mean_rmse, rmses = walk_forward_rmse(X, y, model, n_splits=n_splits)
    preds = walk_forward_predict(X, y, model, n_splits=n_splits)

    bt = backtest_signals(df.loc[preds.index], preds.values, fee_bps=fee_bps)
    bh = (1 + df.loc[bt.index, "y_next_ret"]).cumprod()

    summary = {
        "ticker": ticker,
        "model": "Ridge (regression)",
        "mean_rmse": float(mean_rmse),
        "final_equity": float(bt["equity"].iloc[-1]),
        "sharpe": sharpe(bt["net_ret"]),
        "max_dd": max_drawdown(bt["equity"]),
        "bh_final_equity": float(bh.iloc[-1]),
        "bh_sharpe": sharpe(df.loc[bt.index, "y_next_ret"]),
        "bh_max_dd": max_drawdown(bh),
        "last_pred": float(bt["pred"].iloc[-1]),
        "last_signal": int(bt["signal"].iloc[-1]),
    }

    return summary, bt


if __name__ == "__main__":
    ticker = "AAPL"
    start = "2018-01-01"

    # --- Data pipeline ---
    df = fetch_prices(ticker, start=start)
    df = make_features(df)
    df = make_target(df)
    df = df.dropna()

    feature_cols = ["ret_lag1","ret_lag2","ret_lag5","vol_10","vol_20","sma10_dist","sma50_dist"]
    X = df[feature_cols]
    y = df["y_next_ret"]

    # --- Model ---
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    mean_rmse, rmses = walk_forward_rmse(X, y, model, n_splits=5)
    print(f"{ticker} mean walk-forward RMSE: {mean_rmse:.6f}")
    print("Fold RMSEs:", [float(r) for r in rmses])

    # --- Generate predictions ---
    preds = walk_forward_predict(X, y, model, n_splits=5)

    # --- Backtest ---
    bt = backtest_signals(df.loc[preds.index], preds.values, fee_bps=1.0)

    print("Final equity:", float(bt["equity"].iloc[-1]))

    # --- Buy & Hold comparison ---
    bh = (1 + df.loc[bt.index, "y_next_ret"]).cumprod()
    print("Buy&Hold final equity:", float(bh.iloc[-1]))

    # --- Metrics ---
    print("Strategy Sharpe:", sharpe(bt["net_ret"]))
    print("Strategy Max DD:", max_drawdown(bt["equity"]))
    print("Buy&Hold Sharpe:", sharpe(df.loc[bt.index, "y_next_ret"]))
    print("Buy&Hold Max DD:", max_drawdown(bh))

    # --- Plot ---
    bt["equity"].plot(title=f"{ticker} Equity Curve (Ridge signal)")
    plt.show()

