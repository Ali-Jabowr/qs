# ADR-0004: The marking station is sold alongside and ships first; appearance is the R&D

**Status:** Accepted · 2026-09-02 · Ali · backfilled 2026-09-21 from CLAUDE.md §7 and
`10-architecture.md` §1

## Context

The client's date code is printed on a different conveyor, and a batch has already been returned
over marking (Q21). Checking a date code is deterministic, needs no training data, and is identical
at every future customer. Appearance inspection is genuine machine learning on data that does not
exist yet.

## Decision

Sell Station M (marking) alongside Station A (appearance). **Station M ships first; Station A's
method is prototyped first.** Both are one program with a different Inspector and config.

## Consequences

- The shared core — pipeline, evidence, failure handling — is built once and hardened by whichever
  station runs first.
- Station M is blocked on the client's printer and date-code format (M0). This week's MVP is Station
  A's bench prototype (ADR-0007). That is not a contradiction: it is the only one of the two that
  can start without the client.
