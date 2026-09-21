# ADR-0003: Marketplace parcel dimensioning is not V1

**Status:** Accepted · 2026-09-01 · Ali · backfilled 2026-09-21 from CLAUDE.md §7

## Context

A parked track, the "QC Vision Box" for Wildberries and Ozon sellers, would measure parcels. Parcel
dimensioning is a solved commodity abroad. Under the factory client's own answers it has no user.

## Decision

Dimensioning as a product is not V1. It becomes active only if the WB/Ozon seller track does.

## Not to be confused with

**The geometry check of ADR-0008**, which compares a моти's silhouette with good units — pass/fail
against a golden reference, in pixels — to catch the client's shape and size defects. That is part of
Station A's inspection, not a dimensioning product, and this ADR does not restrict it.

## Consequences

- No engineering time on parcel measurement.
- Revisit only when the marketplace track has a paying user.
