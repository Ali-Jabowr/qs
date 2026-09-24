"""`NullAnnunciator`: the headless `Annunciator` of replay and the tests (docs/11 §4).

It records every `(frame, verdict)` it is shown, in order, and can be told to raise on every
call (F12). An event that has no frame — a NO_VERDICT before capture — is shown with `None`:
the Pipeline shows every event, and a frame is not always there to show (#68, step 8).
"""

from __future__ import annotations

from qs.contracts import Frame, Verdict


class NullAnnunciator:
    """Shows nothing; keeps the calls. `fail=True` raises `RuntimeError` on every `show()`."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[Frame | None, Verdict]] = []
        self.fail = fail
        self.closed = False

    def show(self, frame: Frame | None, verdict: Verdict) -> None:
        self.calls.append((frame, verdict))
        if self.fail:
            raise RuntimeError("the annunciator failed to draw (injected)")

    def close(self) -> None:
        self.closed = True
