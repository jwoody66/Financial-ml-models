\# Financial ML Trading Models



Quantitative research project implementing machine learning techniques to provide trading signals and stochastic price simulations using real historical market data from Yahoo Finance.



---



\## Project Overview



This repository explores whether simple statistical and machine learning models can generate tradable signals in financial markets.



Key components:



\- Ridge Linear Regression for next day return prediction

\- Logistic Regression for next day direction classification

\- Walk forward time-series validation to avoid look-ahead bias

\- Transaction-cost-aware backtesting

\- Monte Carlo Geometric Brownian Motion simulation for future price uncertainty



The goal is to replicate a basic quantitative research workflow used in systematic trading.



---



\## Models Implemented



\### 1. Ridge Regression (Return Forecasting)

Predicts the magnitude of tomorrows log return using:



\- Lagged returns

\- Rolling volatility

\- Moving-average distance features



Performance is evaluated using:



\- Walk-forward RMSE

\- Sharpe ratio

\- Maximum drawdown

\- Comparison vs buy-and-hold



---



\### 2. Logistic Regression (Direction Prediction)



Classifies whether tomorrow’s return is positive (up) or not (down/flat).



Outputs:



\- Probability of upward move

\- Threshold based trading signal

\- Walk-forward AUC

\- Backtested equity curve and risk metrics



---



\### 3. Monte Carlo GBM Simulation



Simulates thousands of potential future price paths using

Geometric Brownian Motion (GBM):



\- Drift and volatility estimated from historical log returns

\- Generates probabilistic price distributions

\- Produces percentile bands and probability of loss



This mirrors techniques used in risk management and derivatives pricing.



---



\## Backtesting Methodology



To ensure realistic evaluation:



\- Time-series walk-forward validation (no data leakage)

\- Out-of-sample predictions only

\- Simple transaction costs included (bps per signal change).

\- Benchmarked against buy and hold



This approximates a simplified systematic trading research pipeline.



---



\## Example Metrics Produced



\- Final equity curve

\- Sharpe ratio

\- Maximum drawdown

\- RMSE (regression)

\- AUC (classification)



---



\## How to Run



pip install -r requirements.txt

python main.py



For MonteCarlo,

python gbm\_mc.py

