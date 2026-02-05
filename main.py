from linear_reg import run_linear
from logistic_reg import run_logistic

def print_summary(title: str, s: dict):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for k, v in s.items():
        if isinstance(v, float):
            print(f"{k:18s}: {v:.4f}")
        else:
            print(f"{k:18s}: {v}")


if __name__ == "__main__":
    ticker = "AAPL"
    start = "2018-01-01"

    
    fee_bps = 1.0
    ridge_threshold = 0.00005      
    p_threshold = 0.55            

    lin_summary, lin_bt = run_linear(
        ticker=ticker, start=start, fee_bps=fee_bps, threshold=ridge_threshold, n_splits=5
    )
    log_summary, log_bt = run_logistic(
        ticker=ticker, start=start, fee_bps=fee_bps, p_threshold=p_threshold, n_splits=5
    )

    print_summary("LINEAR / RIDGE MODEL RESULTS", lin_summary)
    print_summary("LOGISTIC MODEL RESULTS", log_summary)

    print("\n" + "-" * 60)
    print("TODAY'S SIGNALS (based on most recent bar in dataset)")
    print("-" * 60)
    print(f"Ridge predicted next return: {lin_summary['last_pred']:.4f} -> signal={lin_summary['last_signal']} (1=LONG,0=FLAT)")
    print(f"Logistic P(up):               {log_summary['last_proba_up']:.4f} -> signal={log_summary['last_signal']} (1=LONG,0=FLAT)")

    agree = (lin_summary["last_signal"] == log_summary["last_signal"])
    print(f"\nModels agree? {agree}")
