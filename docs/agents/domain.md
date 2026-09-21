# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root: the domain glossary.
- **`decisions/`**: this repo's ADR log, `ADR-NNNN-<slug>.md`, indexed in `decisions/README.md`.
  Read the ADRs that touch the area you're about to work in.
- **`docs/05`–`docs/17`**: concept, requirements and the Phase 2 design. Read the one that covers your area.

If `CONTEXT.md` doesn't exist, **proceed silently**. Don't flag its absence; don't suggest creating it
upfront. The `/domain-modeling` skill creates it lazily when terms actually get resolved.

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── decisions/
│   ├── README.md                  ← the index
│   ├── ADR-0001-smartbot-closed-qs-replaces-it.md
│   └── …
├── docs/
└── src/qs/
```

New ADRs go in `decisions/`, numbered on from the last entry in `decisions/README.md`, and are added to
that index. Never create `docs/adr/`. A decision that changes is superseded by a new ADR, not edited.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test
name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the
project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0008 (geometry check in code), but worth reopening because…_
