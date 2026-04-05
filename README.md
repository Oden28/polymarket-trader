# Polymarket Crypto Trading Bot

An automated trading bot for [Polymarket](https://polymarket.com) prediction markets, focused on crypto/financial markets. It uses real-time price data and options-implied volatility to find mispriced markets and trade the edge.

## How It Works

The bot runs a continuous loop:

1. **Discover markets** — Fetches active crypto prediction markets from Polymarket's Gamma API (sorted by 24h volume)
2. **Enrich with orderbook data** — Pulls real-time bid/ask spreads from the CLOB API for accurate pricing
3. **Parse each market question** — Extracts the asset (BTC, ETH, SOL, etc.), price target, direction, and expiry date from question text
4. **Fetch live market data** — Gets the current spot price from CoinGecko and implied volatility from Deribit DVOL
5. **Calculate fair probability** — Uses a Black-Scholes / log-normal model to compute the mathematically fair chance of the outcome
6. **Find mispriced markets** — Compares the oracle's fair probability to what Polymarket is pricing — if the gap exceeds your minimum edge threshold, it generates a signal
7. **Size the position** — Applies fractional Kelly criterion to determine how much to bet
8. **Execute (or log)** — In dry-run mode, logs the signal. In live mode, places a limit order through the CLOB

### Probability Model

For a market like "Will BTC be above $70,000 on April 10?", the oracle calculates:

```
P(S_T > K) = N(d2)

where d2 = [ln(S/K) + (μ - σ²/2) * T] / (σ * √T)
```

- **S** = current spot price (live from CoinGecko)
- **K** = target price (parsed from the market question)
- **T** = time to expiry in years
- **σ** = annualized implied volatility (live from Deribit DVOL)
- **μ** = drift (set to 0 for risk-neutral pricing)

This is the same model used by professional options traders.

### Supported Market Types

| Pattern | Example | Method |
|---------|---------|--------|
| Above $X | "Will BTC be above $70,000 on April 10?" | `P(S_T > K)` |
| Below $X | "Will ETH be below $2,000 on April 10?" | `1 - P(S_T > K)` |
| Between $X and $Y | "Will BTC be between $65k and $70k?" | `P(S_T > lower) - P(S_T > upper)` |
| Reach $X | "Will BTC reach $100,000 in April?" | Barrier option approximation |
| Dip to $X | "Will BTC dip to $50,000?" | Barrier option approximation |
| Up / Down | "Bitcoin up on April 5?" | `P(S_T > S_0)` |

### Supported Assets

BTC, ETH, SOL, XRP, DOGE, ADA, MATIC, AVAX, LINK, LTC — any crypto with a CoinGecko listing.

## Quick Start

### Prerequisites

- Python 3.10+
- A Polymarket account with USDC deposited
- Your Polygon wallet private key ([how to export](https://reveal.polymarket.com))

### Installation

```bash
git clone <repo-url> && cd polymarket-bot
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and add your private key:

```env
POLY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
```

If you use email/Magic link login on Polymarket, also set:

```env
POLY_SIGNATURE_TYPE=1
POLY_FUNDER_ADDRESS=0xYOUR_POLYMARKET_DEPOSIT_ADDRESS
```

### Run

```bash
# See what signals the bot would generate (no trades, no wallet needed)
python main.py test

# List active markets with prices and volume
python main.py scan

# Start the trading loop (DRY_RUN=true by default — no real trades)
python main.py

# Show help
python main.py help
```

## Configuration Reference

All settings are in `.env`. Defaults are tuned for ~$500 capital.

### Authentication

| Variable | Description | Default |
|----------|-------------|---------|
| `POLY_PRIVATE_KEY` | Your Polygon wallet private key (hex, 0x prefix) | *(required)* |
| `POLY_SIGNATURE_TYPE` | `0` = EOA/MetaMask, `1` = email/Magic, `2` = browser wallet | `0` |
| `POLY_FUNDER_ADDRESS` | Proxy wallet address (required for types 1 and 2) | — |

### Risk Management

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_POSITION_SIZE_USDC` | Maximum USDC per trade | `25` |
| `MAX_DAILY_LOSS_USDC` | Stop trading after this daily loss | `50` |
| `MAX_POSITIONS` | Maximum simultaneous open positions | `10` |
| `MAX_TRADES_PER_DAY` | Daily trade limit | `20` |

### Strategy

| Variable | Description | Default |
|----------|-------------|---------|
| `MIN_EDGE_PERCENTAGE` | Minimum mispricing to trade (%) | `5.0` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence score (0–1) | `0.65` |
| `KELLY_FRACTION` | Fraction of Kelly criterion to bet (lower = safer) | `0.15` |

### Market Filters

| Variable | Description | Default |
|----------|-------------|---------|
| `MIN_LIQUIDITY` | Skip markets with less than this USDC liquidity | `10000` |
| `MIN_VOLUME_24H` | Skip markets with less than this 24h volume | `5000` |
| `MAX_SPREAD` | Skip markets with a wider bid-ask spread | `0.05` |

### Operational

| Variable | Description | Default |
|----------|-------------|---------|
| `UPDATE_INTERVAL_SECONDS` | How often to scan for new signals | `60` |
| `DRY_RUN` | `true` = log signals only, `false` = place real orders | `true` |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) | `INFO` |

## Project Structure

```
├── main.py              # CLI entry point (main, test, scan, help)
├── config.py            # Environment config loader with validation
├── trader.py            # Polymarket trader: market discovery, orderbook, execution
├── oracle.py            # Data oracle: live prices, volatility, probability math
├── smart_strategy.py    # Oracle-backed strategy (Black-Scholes fair pricing)
├── strategy.py          # Base strategy class with Kelly sizing
├── risk_manager.py      # Risk limits: daily loss, position size, trade count
├── backtest.py          # Backtesting framework
├── scan_crypto.py       # Standalone crypto market scanner utility
├── tests/               # 76 unit tests
│   ├── test_oracle.py
│   ├── test_strategy.py
│   ├── test_trader.py
│   ├── test_risk_manager.py
│   ├── test_backtest.py
│   └── test_config.py
├── .env.example         # Configuration template
├── .gitignore           # Excludes .env and caches
└── requirements.txt     # Python dependencies
```

## Data Sources

All free, no API keys required:

| Source | Data | Rate Limit |
|--------|------|------------|
| [CoinGecko](https://www.coingecko.com/en/api) | Spot prices for all supported cryptos | 30 req/min |
| [Deribit](https://docs.deribit.com/) | BTC/ETH implied volatility (DVOL), index prices (fallback) | Generous |
| [Polymarket Gamma API](https://gamma-api.polymarket.com) | Market discovery, metadata, volume | Generous |
| [Polymarket CLOB API](https://docs.polymarket.com) | Orderbook, order execution | Generous |

The oracle caches prices for 30 seconds and volatility for 10 minutes to stay well within rate limits.

## Safety

- **DRY_RUN is on by default.** No real orders are placed until you explicitly set `DRY_RUN=false`.
- **Live mode requires confirmation.** When `DRY_RUN=false`, the bot asks you to type "yes" before it starts.
- **Daily loss circuit breaker.** Trading halts automatically when daily losses hit your limit.
- **Conservative Kelly.** At `KELLY_FRACTION=0.15`, the bot bets ~15% of the theoretically optimal size.
- **Near-expiry filter.** Markets within 2 hours of settlement are skipped (likely already priced in, no real edge).
- **Never commit your `.env` file.** It contains your private key.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Disclaimer

This is experimental software. Trading on prediction markets carries risk of loss. Past signals do not guarantee future performance. Use at your own risk and never trade more than you can afford to lose.
