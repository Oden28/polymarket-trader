#!/usr/bin/env python3
"""
Polymarket Automated Trading Bot

A professional trading system for Polymarket prediction markets with:
- Real-time market analysis via Gamma + CLOB APIs
- Edge detection and signal generation
- Risk-managed order execution
- Portfolio tracking
"""
import asyncio
import sys
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

from config import config
from trader import PolymarketTrader

logger = structlog.get_logger()


def check_config() -> bool:
    """Verify configuration is valid."""
    issues = config.validate()
    if issues:
        print("\n" + "=" * 60)
        print("Configuration Issues")
        print("=" * 60)
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease create a .env file with your configuration:")
        print("\n1. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("\n2. Edit .env and add your:")
        print("   - POLY_PRIVATE_KEY: Your Polygon wallet private key")
        print("   - POLY_FUNDER_ADDRESS: Your proxy wallet (if using email login)")
        print("   - Trading parameters (optional)")
        print("\n  Never commit your .env file!")
        print("=" * 60 + "\n")
        return False
    return True


async def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Polymarket Trading Bot")
    print("=" * 60)

    # Check configuration
    if not check_config():
        sys.exit(1)

    # Display configuration summary
    print(f"\nConfiguration:")
    print(f"  Max Position Size: ${config.max_position_size_usdc} USDC")
    print(f"  Max Daily Loss: ${config.max_daily_loss_usdc} USDC")
    print(f"  Min Edge: {config.min_edge_percentage}%")
    print(f"  Confidence Threshold: {config.confidence_threshold}")
    print(f"  Update Interval: {config.update_interval_seconds}s")
    print(f"  Chain ID: {config.poly_chain_id}")
    print(f"  Signature Type: {config.poly_signature_type}")
    print(f"  Dry Run: {config.dry_run}")
    print(f"  API URL: {config.poly_api_url}")
    print("\n" + "=" * 60)

    # Safety confirmation
    if not config.dry_run:
        print("\n  WARNING: DRY_RUN is OFF – this bot will trade with REAL MONEY!")
        response = input("\nDo you want to start trading? (yes/no): ").lower().strip()
        if response not in ("yes", "y"):
            print("\nTrading cancelled.")
            sys.exit(0)
    else:
        print("\n  Running in DRY RUN mode – no real trades will be placed.")

    # Initialize and start trader
    trader = PolymarketTrader(config)

    try:
        await trader.start()
    except KeyboardInterrupt:
        print("\nShutting down …")
        await trader.stop()
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        raise


async def test_mode():
    """Run in test mode – scan markets and show signals without trading."""
    print("\n" + "=" * 60)
    print("Polymarket Trading Bot – TEST MODE")
    print("=" * 60)
    print("\nScanning live markets for signals (no trades) …\n")

    # Force dry-run regardless of .env
    config.dry_run = True

    if not config.is_configured:
        print("No private key configured – using read-only mode.\n")
        # Create a read-only trader that only fetches public data
        config.poly_private_key = ""

    trader = PolymarketTrader(config)

    try:
        signals = await trader.run_once()

        print(f"\n{'=' * 60}")
        print(f"Found {len(signals)} trading signals")
        print(f"{'=' * 60}\n")

        for i, signal in enumerate(signals[:10], 1):
            print(f"\n{i}. {signal.market_slug}")
            print(f"   Question: {signal.reasoning.split('|')[0].strip()}")
            print(f"   Side: {signal.side}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Edge: {signal.edge_percentage:.2f}%")
            print(f"   Current Price: {signal.current_price:.4f}")
            print(f"   Fair Price: {signal.fair_probability:.4f}")
            print(f"   Suggested Size: ${signal.suggested_size:.2f} USDC")

        print(f"\n{'=' * 60}")
        print("Test mode complete – no trades were executed")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"Error in test mode: {e}")
        import traceback
        traceback.print_exc()


async def scan_markets():
    """Scan and display all active markets with key metrics."""
    print("\n" + "=" * 60)
    print("Polymarket Market Scanner")
    print("=" * 60 + "\n")

    trader = PolymarketTrader(config)
    markets = await trader.fetch_markets()

    print(f"Found {len(markets)} active markets\n")
    print(f"{'#':<4} {'Question':<50} {'Price':>7} {'Vol24h':>10} {'Liq':>10}")
    print("-" * 85)

    for i, m in enumerate(markets[:30], 1):
        q = m.question[:48] + "…" if len(m.question) > 48 else m.question
        print(
            f"{i:<4} {q:<50} {m.current_price:>7.3f} "
            f"${m.volume_24h:>9,.0f} ${m.liquidity:>9,.0f}"
        )

    print(f"\n{'=' * 60}\n")


def print_help():
    """Print usage information."""
    print("""
Polymarket Trading Bot

Usage:
    python main.py              Start the trading bot
    python main.py test         Run in test mode (analyse only)
    python main.py scan         Scan active markets
    python main.py dashboard  Local web UI (5m Up/Down + wallet)
    python main.py help         Show this help message

Environment:
    Set configuration in .env file (copy from .env.example)

Key Variables:
    POLY_PRIVATE_KEY         Your Polygon wallet private key
    POLY_FUNDER_ADDRESS      Your proxy/funder wallet address
    POLY_SIGNATURE_TYPE      0=EOA, 1=email/Magic, 2=browser wallet
    MAX_POSITION_SIZE_USDC   Maximum position size (default: 100)
    MAX_DAILY_LOSS_USDC      Daily loss limit (default: 500)
    MIN_EDGE_PERCENTAGE      Minimum edge to trade (default: 2.0)
    CONFIDENCE_THRESHOLD     Minimum confidence (default: 0.65)
    DRY_RUN                  true/false – skip order execution (default: true)

Safety:
    - DRY_RUN=true by default – no real trades until you opt in
    - Start with small position sizes
    - Test mode first to see signals
    - Monitor daily loss limits
    - Keep your private key secure
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower().strip("-")
        if cmd in ("test", "t"):
            asyncio.run(test_mode())
        elif cmd in ("scan", "markets"):
            asyncio.run(scan_markets())
        elif cmd in ("dashboard", "dash", "ui"):
            from dashboard_server import run_dashboard

            run_dashboard()
        elif cmd in ("help", "h"):
            print_help()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print_help()
            sys.exit(1)
    else:
        asyncio.run(main())
