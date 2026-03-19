"""Data layer — providers, storage, quality, and catalog."""

from engine.data.storage import read_bars, write_bars
from engine.data.quality import DataCleaner
from engine.data.contracts import ContractStitcher
from engine.data.catalog import DataCatalog


def get_provider(name: str):
    """Factory: return a DataProvider instance by name.

    Args:
        name: One of "databento", "yahoo", "motivewave".

    Returns:
        An instantiated DataProvider subclass.
    """
    name = name.lower()
    if name == "databento":
        from engine.data.providers.databento import DatabentoProvider
        return DatabentoProvider()
    elif name == "yahoo":
        from engine.data.providers.yahoo import YahooProvider
        return YahooProvider()
    elif name == "motivewave":
        from engine.data.providers.motivewave import MotiveWaveProvider
        return MotiveWaveProvider()
    else:
        raise ValueError(
            f"Unknown provider: {name!r}. "
            f"Available: 'databento', 'yahoo', 'motivewave'"
        )


__all__ = [
    "get_provider",
    "read_bars",
    "write_bars",
    "DataCleaner",
    "ContractStitcher",
    "DataCatalog",
]
