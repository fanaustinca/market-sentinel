"""The technology universe, and the bias that choosing it introduces.

Narrowing to one sector is a reasonable request with an unreasonable failure
mode. "Focus on tech" almost always becomes a list of the companies that turned
out to be the winners -- Nvidia, Apple, Microsoft -- and a backtest on that list
cannot lose. The stocks were selected *because* they went up. Any strategy that
holds them looks brilliant, and the brilliance belongs to the person who wrote
the list in 2026, not to the model.

This module keeps the two honest instruments separate from the compromised one.

`SECTOR_ETFS` are index funds. Their membership was chosen by a rule at the time,
not by hindsight, and when a component collapsed the fund ate the loss. XLK
starting in 1998 contains the dot-com bust in full, including the companies that
never came back. This is the universe for any claim about returns.

`LONG_LIVED_TECH` is a basket of large technology companies that were already
large before the dot-com peak. It still has survivorship bias -- every name on it
survived to be typed -- but the bias is far weaker than a list of recent winners,
because these were selected for being big in 1999 rather than big now, and
several (Intel, Cisco, IBM) spent the following twenty years underperforming
badly. That is the point: a defensible tech basket must contain its own losers.

The event study uses `LONG_LIVED_TECH`, and it controls for the remaining bias by
shuffling event dates *within each stock*. That comparison holds the stock's own
drift fixed, so whatever survivorship advantage a name carries is present in both
the real and the shuffled arm and cancels out. What survives the comparison is
timing around the event, which is the only thing being claimed.
"""

from __future__ import annotations

#: Broad technology index funds. Rule-based membership, losers included.
SECTOR_ETFS = {
    "XLK": "US technology sector (1998)",
    "QQQ": "Nasdaq-100, tech-heavy (1999)",
    "SMH": "semiconductors (2000)",
}

#: Large technology companies that were already large before the 2000 peak.
#: Chosen for being significant in 1999, which is why the list includes names
#: that subsequently did badly -- Intel and Cisco never regained their peaks.
LONG_LIVED_TECH = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "IBM": "IBM",
    "INTC": "Intel",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "TXN": "Texas Instruments",
    "QCOM": "Qualcomm",
    "AMD": "AMD",
    "NVDA": "Nvidia",
    "ADBE": "Adobe",
    "AMAT": "Applied Materials",
    "MU": "Micron",
    "HPQ": "HP",
    "DELL": "Dell",
    "WDC": "Western Digital",
}

#: Added later, and named separately so a result can be recomputed without them.
#: Including these is the step that introduces real hindsight -- each was picked
#: knowing it became enormous.
MODERN_TECH = {
    "GOOGL": "Alphabet (2004)",
    "AMZN": "Amazon (1997)",
    "META": "Meta (2012)",
    "CRM": "Salesforce (2004)",
    "AVGO": "Broadcom (2009)",
    "NFLX": "Netflix (2002)",
}
