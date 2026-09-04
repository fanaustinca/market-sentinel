"""Real market data, presented through the same interface as the sandbox.

This is rung 2 of the Reality Ladder. The only thing that changes when climbing
here is where the prices come from -- every strategy, the engine, the risk layer
and the evaluation harness are the identical code paths the synthetic markets
exercised. That is the entire point of the `MarketData` interface: any change in
behaviour at this rung is caused by reality, not by a different code path.
"""

from sentinel.data.yahoo import CACHE_DIR, load_prices, universe_history

__all__ = ["CACHE_DIR", "load_prices", "universe_history"]
