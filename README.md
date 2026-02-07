# Deep Learning Portfolio Optimization

This model explores the application of **Transformer** and **LSTM** architectures to automate asset allocation. Unlike traditional models that predict prices, this framework directly outputs portfolio weights to maximize the risk-adjusted return (Sharpe Ratio).

## Project Architecture
The pipeline transforms raw market data into optimized weights through a high-fidelity deep learning stack:

1. **Data Ingestion**: Multi-asset universe including Tech Equities (NVDA, TSLA, AAPL, MSFT) and Broad Market ETFs (SPY, QQQ, IWM, RSP).
2. **Preprocessing**: 
    - **Feature Engineering**: Calculates rolling volatility as well as moving averages.
    - **Scaling**: StandardScaler application to prevent look-ahead bias and leakage.
    - **Windowing**: 16-week sequence generation for temporal context.
3. **Optimization**: 
    - **Loss Function**: Negative Sharpe Ratio.
    - **Output Layer**: Softmax activation for constrained weight distribution (summing to 1).

## Key Findings
- **Attention vs. Recurrence**: The Transformer's attention mechanism effectively captured cross-asset interactions (e.g., NVDA/QQQ correlation) more robustly than the LSTM's sequential state.
- **Feature Importance**: Random Forest analysis identified **Volatility** as the primary predictive driver for weight rebalancing.
- **Performance**: Transformers exhibited higher stability and less sensitivity to hyperparameter variance compared to LSTMs.

## Tech Stack
- **Deep Learning**: Transformers, LSTMs
- **Optimization**: Sharpe Ratio Maximization
- **Analysis**: Random Forest (Feature Importance)
- **Data**: Financial Time-Series (Prices, Volume)

## Future Roadmap
- Integration of **Reinforcement Learning (RL)** for agent-based trading policies.
- Incorporation of **Alternative Data** (Sentiment analysis via NLP).
