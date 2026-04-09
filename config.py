"""Configuration management for Polymarket trading bot."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent / ".env", encoding="utf-8-sig")


def _load_poly_private_key() -> str:
    """Support POLY_PRIVATE_KEY (preferred) or PRIVATE_KEY; trim whitespace / wrapping quotes."""
    v = (os.getenv("POLY_PRIVATE_KEY") or os.getenv("PRIVATE_KEY") or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    return v


@dataclass
class TradingConfig:
    """Trading bot configuration loaded from environment variables."""

    # ── Polymarket connection ──────────────────────────────────────────
    poly_private_key: str = field(default_factory=_load_poly_private_key)
    poly_api_url: str = field(
        default_factory=lambda: os.getenv(
            "POLY_API_URL", "https://clob.polymarket.com"
        )
    )
    gamma_api_url: str = field(
        default_factory=lambda: os.getenv(
            "GAMMA_API_URL", "https://gamma-api.polymarket.com"
        )
    )
    poly_chain_id: int = field(
        default_factory=lambda: int(os.getenv("POLY_CHAIN_ID", "137"))
    )
    # Signature type: 0 = EOA, 1 = email/Magic, 2 = browser wallet proxy
    poly_signature_type: int = field(
        default_factory=lambda: int(os.getenv("POLY_SIGNATURE_TYPE", "0"))
    )
    # Funder / proxy wallet address (required for signature_type 1 or 2)
    poly_funder_address: str = field(
        default_factory=lambda: os.getenv("POLY_FUNDER_ADDRESS", "")
    )

    # ── Risk parameters ───────────────────────────────────────────────
    max_position_size_usdc: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE_USDC", "100"))
    )
    max_daily_loss_usdc: float = field(
        default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_USDC", "500"))
    )
    max_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_POSITIONS", "10"))
    )
    max_trades_per_day: int = field(
        default_factory=lambda: int(os.getenv("MAX_TRADES_PER_DAY", "50"))
    )

    # ── Strategy parameters ───────────────────────────────────────────
    min_edge_percentage: float = field(
        default_factory=lambda: float(os.getenv("MIN_EDGE_PERCENTAGE", "2.0"))
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    )
    kelly_fraction: float = field(
        default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.25"))
    )

    # ── Market filters ────────────────────────────────────────────────
    min_liquidity: float = field(
        default_factory=lambda: float(os.getenv("MIN_LIQUIDITY", "5000"))
    )
    min_volume_24h: float = field(
        default_factory=lambda: float(os.getenv("MIN_VOLUME_24H", "1000"))
    )
    max_spread: float = field(
        default_factory=lambda: float(os.getenv("MAX_SPREAD", "0.10"))
    )

    # ── Operational ───────────────────────────────────────────────────
    update_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("UPDATE_INTERVAL_SECONDS", "60"))
    )
    dry_run: bool = field(
        default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    @property
    def is_configured(self) -> bool:
        """Check whether minimum configuration is present."""
        return bool(self.poly_private_key) and len(self.poly_private_key) > 10

    @property
    def needs_funder(self) -> bool:
        """Whether the chosen signature type requires a funder address."""
        return self.poly_signature_type in (1, 2)

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty = OK)."""
        issues: list[str] = []
        if not self.is_configured:
            issues.append(
                "POLY_PRIVATE_KEY (or PRIVATE_KEY) is missing or too short"
            )
        if self.needs_funder and not self.poly_funder_address:
            issues.append(
                f"POLY_FUNDER_ADDRESS required for signature_type={self.poly_signature_type}"
            )
        if self.max_position_size_usdc <= 0:
            issues.append("MAX_POSITION_SIZE_USDC must be > 0")
        if self.max_daily_loss_usdc <= 0:
            issues.append("MAX_DAILY_LOSS_USDC must be > 0")
        if not (0 < self.confidence_threshold < 1):
            issues.append("CONFIDENCE_THRESHOLD must be between 0 and 1")
        if self.min_edge_percentage < 0:
            issues.append("MIN_EDGE_PERCENTAGE must be >= 0")
        return issues


# Singleton – importable as `from config import config`
config = TradingConfig()
