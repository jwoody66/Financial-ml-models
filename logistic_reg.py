import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score


def fetch_prices(ticker: str, start="2018-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return df.dropna()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = np.log(out["Close"] / out["Close"].shift(1))

    out["ret_lag1"] = out["ret_1"].shift(1)
    out["ret_lag2"] = out["ret_1"].shift(2)
    out["ret_lag5"] = out["ret_1"].shift(5)

    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    sma_10 = out["Close"].rolling(10).mean()
    sma_50 = out["Close"].rolling(50).mean()
    out["sma10_dist"] = (out["Close"] - sma_10) / out["Close"]
    out["sma50_dist"] = (out["Close"] - sma_50) / out["Close"]

    return out


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y_next_ret"] = out["ret_1"].shift(-1) 
    out["y_up"] = (out["y_next_ret"] > 0).astype(int)
    return out


def sharpe(returns, periods=252) -> float:
    r = pd.Series(returns).dropna()
    if r.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * r.mean() / r.std())


def max_drawdown(equity) -> float:
    e = pd.Series(equity).dropna()
    peak = e.cummax()
    dd = (e / peak) - 1.0
    return float(dd.min())


def walk_forward_auc(X, y, model, n_splits=5) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        aucs.append(float(roc_auc_score(y_test, proba)))
    return float(np.mean(aucs))


def walk_forward_predict_proba(X, y, model, n_splits=5) -> pd.Series:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    probs = pd.Series(index=X.index, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        model.fit(X_train, y_train)
        probs.iloc[test_idx] = model.predict_proba(X_test)[:, 1]
    return probs


def backtest_proba(df: pd.DataFrame, probs: np.ndarray, p_threshold=0.55, fee_bps=1.0) -> pd.DataFrame:
    signal = (probs > p_threshold).astype(int)  # 1 = long, 0 = flat
    strat_ret = signal * df["y_next_ret"].values

    turnover = np.abs(np.diff(signal, prepend=signal[0]))
    cost = turnover * (fee_bps / 10000.0)

    net_ret = strat_ret - cost
    equity = (1 + net_ret).cumprod()

    return pd.DataFrame({
        "proba_up": probs,
        "signal": signal,
        "net_ret": net_ret,
        "equity": equity
    }, index=df.index)


def run_logistic(ticker="AAPL", start="2018-01-01", fee_bps=1.0, p_threshold=0.55, n_splits=5):
    df = fetch_prices(ticker, start=start)
    df = make_features(df)
    df = make_targets(df)
    df = df.dropna()

    feature_cols = ["ret_lag1", "ret_lag2", "ret_lag5", "vol_10", "vol_20", "sma10_dist", "sma50_dist"]
    X = df[feature_cols]
    y = df["y_up"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000))
    ])

    mean_auc = walk_forward_auc(X, y, model, n_splits=n_splits)
    probs = walk_forward_predict_proba(X, y, model, n_splits=n_splits)

    bt = backtest_proba(df.loc[probs.index], probs.values, p_threshold=p_threshold, fee_bps=fee_bps)
    bh = (1 + df.loc[bt.index, "y_next_ret"]).cumprod()

    summary = {
        "ticker": ticker,
        "model": "Logistic (classification)",
        "mean_auc": mean_auc,
        "final_equity": float(bt["equity"].iloc[-1]),
        "sharpe": sharpe(bt["net_ret"]),
        "max_dd": max_drawdown(bt["equity"]),
        "bh_final_equity": float(bh.iloc[-1]),
        "bh_sharpe": sharpe(df.loc[bt.index, "y_next_ret"]),
        "bh_max_dd": max_drawdown(bh),
        "last_proba_up": float(bt["proba_up"].iloc[-1]),
        "last_signal": int(bt["signal"].iloc[-1]),
    }

    return summary, bt
