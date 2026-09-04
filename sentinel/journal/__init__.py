"""Recording what the system decided, before anyone knows whether it was right.

`plan.md` Phase 5 requires that every decision is logged *before* the outcome is
known, and the requirement is not bookkeeping. A decision recalled after the fact
is not evidence about anything -- the memory reshapes itself around what happened,
reliably and without anyone noticing. Only a prediction written down in advance
can be checked, and only a checkable prediction can distinguish "the model is
behaving as expected" from "the model is behaving, and I am explaining it".

This is also the difference between paper trading that means something and paper
trading that does not. Six months of watching a strategy and nodding proves
nothing. Six months of dated files saying what was expected, compared afterwards
against what happened, is a real test -- and any divergence is a bug to find
rather than a story to tell.
"""

from sentinel.journal.signals import SignalReport, current_signals, write_journal_entry

__all__ = ["SignalReport", "current_signals", "write_journal_entry"]
