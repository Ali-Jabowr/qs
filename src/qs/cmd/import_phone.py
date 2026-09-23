"""`qs import-phone`: phone photos into the session, as `qs shoot` would have saved them.

ADR-0022: the phone shoots offline, following the printed list (`qs split-plan --shot-list`);
the photos reach the laptop over USB. Photo k, in capture-time order, is shot k. The import
writes the same `raw/<session>/` as `qs shoot` — rule-named PNGs (docs/12 §2), `frames.csv`,
`session.yml` (docs/13 §8) — plus `phone/`, the originals byte for byte, and `import.csv`.

It fails closed: the EXIF lock, the order, the mapping, the count and a full decode of every
photo all run before the first write, and any failure writes nothing. `--from` is only read.

`--check` writes nothing anywhere. Without `--session` it is the phone's day-zero test (EXIF lock
and grey card); with it, the mid-session "where am I".
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, timezone
from pathlib import Path

from qs.config import StationConfig, load_config
from qs.contracts import CameraReadback, TriggerEvent
from qs.errors import ConfigError, LightingFault
from qs.names import DEGREES, SUFFIX, FrameNameError, parse_name
from qs.phone import (
    PATCH_LIMIT,
    Mapping,
    PhoneRefused,
    Photo,
    Unsupported,
    check_lock,
    decode,
    gap_warnings,
    in_order,
    list_photos,
    map_photos,
    next_step,
    patch_variation,
    read_photo,
    utc_offset,
)
from qs.session import (
    PLAN_FILE,
    FrameLog,
    ImportLog,
    Product,
    SessionRecord,
    copy_original,
    read_session_plan,
    save_frame,
    save_rejected,
    session_started,
    write_session,
)
from qs.sources.webcam import measure_patch
from qs.splitplan import (
    PilotPlan,
    Plan,
    PlanError,
    Shot,
    Substitution,
    discard_broken,
    pilot_shot_list,
    shot_list,
    substitute_spares,
)

_SOURCE_ID = "phone"


def offset(text: str) -> timezone:
    """argparse type: `--utc-offset ±HH:MM`."""
    try:
        return utc_offset(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def redo_list(text: str) -> list[int | str]:
    """argparse type: `--redo K[,K…]` — row K of the printed list, from 1 — or a frame name
    (`u053_good_G2_r1`), for a spare's photo, which has no row; a name is kept with `.png`."""
    items: list[int | str] = []
    for part in text.split(","):
        if part.isdigit() and int(part) >= 1:
            items.append(int(part))
            continue
        file = part if part.endswith(SUFFIX) else part + SUFFIX
        try:
            parse_name(file)
        except FrameNameError:
            raise argparse.ArgumentTypeError(
                f"{text!r}: rows of the printed list from 1, or frame names — e.g. 40,41 or "
                "u053_good_G2_r1"
            ) from None
        items.append(file)
    return items


def broken_unit(text: str) -> tuple[str, int]:
    """argparse type: `--broken uNNN:N` or `pNN:N`, the unit and the photos it was shot in."""
    unit, _, count = text.partition(":")
    if not unit.startswith(("u", "p")) or not count.isdigit():
        raise argparse.ArgumentTypeError(
            f"{text!r}: a dataset or pilot unit and its photos before it broke, e.g. u012:2 "
            "or p03:4"
        )
    return unit, int(count)


@dataclass(frozen=True, slots=True)
class _PhoneFrame:
    """A saved photo as the frame log and `session.yml` keep it (`session.Recorded`)."""

    event: TriggerEvent
    reference_patch: float | None
    camera: CameraReadback


def _read(paths: Sequence[Path], args: argparse.Namespace) -> list[Photo]:
    return in_order([read_photo(path, args.utc_offset) for path in paths])


def _readings(photos: Sequence[Photo], config: StationConfig) -> list[float]:
    """Decode every photo once, check its size, and read the grey card on it (FR-14)."""
    roi = config.lighting.reference_patch.roi
    size = (config.camera.width, config.camera.height)
    readings = []
    for photo in photos:
        try:
            pixels = decode(photo.path)
        except (OSError, SyntaxError, ValueError) as exc:
            raise PhoneRefused(f"{photo.name}: could not be decoded — {exc}") from None
        if (pixels.shape[1], pixels.shape[0]) != size:
            raise PhoneRefused(
                f"{photo.name}: decodes to {pixels.shape[1]}x{pixels.shape[0]}, not "
                f"{size[0]}x{size[1]}"
            )
        try:
            readings.append(measure_patch(pixels, roi))
        except LightingFault as exc:
            raise PhoneRefused(f"{photo.name}: {exc}") from None
    return readings


def _drift(readings: Sequence[float]) -> str:
    first, last = readings[0], readings[-1]
    text = f"grey card {last:.4f}"
    if first:
        text += f" · drift {(last - first) / first * 100:+.2f}% since the first photo"
    return text


def _day_zero(photos: list[Photo], config: StationConfig) -> int:
    """The phone's day-zero test (ADR-0022): the lock over every photo and the grey card
    varying by less than 1%. 0 when both pass."""
    lock = check_lock(photos, config.camera)
    print("\n".join(lock.lines))
    try:
        readings = _readings(photos, config)
    except PhoneRefused as exc:
        print(f"FAIL reference patch: {exc}")
        return 1
    variation = patch_variation(readings)
    patch_ok = variation < PATCH_LIMIT
    print(f"{'PASS' if patch_ok else 'FAIL'} reference patch: varies {variation * 100:.2f}% "
          f"(limit {PATCH_LIMIT * 100:g}%) over {len(readings)} photos")
    return 0 if lock.passed and patch_ok else 1


def _printed_rows(plan: Plan | PilotPlan, geometry: str) -> list[str]:
    """The frame file of each row of the printed list, row 1 first — the list as planned,
    before any unit broke, because that is the paper in Ali's hand (F2)."""
    steps = pilot_shot_list(plan) if isinstance(plan, PilotPlan) else shot_list(plan, geometry)
    return [step.file for step in steps if isinstance(step, Shot)]


def _redo_files(
    redo: Sequence[int | str], printed: Sequence[str], shot: Substitution
) -> list[str]:
    """Each `--redo` as the frame it spoiled. Row K is the printed row, not the K-th shot as
    shot: after a broken unit the two differ, and a guess would misname every frame after the
    redo. `PhoneRefused` for a row past the list, or a frame this session never shot."""
    shot_files = {step.file for step in shot.steps if isinstance(step, Shot)}
    files: list[str] = []
    for item in redo:
        if isinstance(item, int):
            if item > len(printed):
                raise PhoneRefused(f"--redo {item}: the printed list has rows 1 to "
                                   f"{len(printed)}")
            file = printed[item - 1]
            if file not in shot_files:
                raise PhoneRefused(f"--redo {item}: row {item} is {file}, which was never shot "
                                   "— its unit broke before it")
        else:
            file = item
            if file not in shot_files:
                raise PhoneRefused(f"--redo {file.removesuffix(SUFFIX)}: no such shot in this "
                                   "session")
        files.append(file)
    return files


def _where_am_i(
    photos: list[Photo], config: StationConfig, shot: Substitution, redo: Sequence[str],
    printed: Sequence[str],
) -> int:
    """The mid-session check: the lock so far, the mapping, the next photo, the drift. A shot
    is named by its printed row, as the paper numbers it."""
    lock = check_lock(photos, config.camera)
    print("\n".join(lock.lines))
    if not lock.passed:
        return 1
    mapping = map_photos(len(photos), shot.steps, redo, _discarded(shot))
    if len(photos) > mapping.expected_photos:
        print(f"check: {len(photos)} photos, but the session has {mapping.expected_photos} "
              "— more photos than shots")
        return 1
    prompts, following = next_step(shot.steps, mapping.shots_taken)
    row = {file: n for n, file in enumerate(printed, start=1)}
    line = f"{mapping.shots_taken} of {mapping.shots_total} photos mapped"
    if following is None:
        line += " — every shot is taken"
    else:
        frame = following.frame
        what = f"{frame.unit} {frame.state} {DEGREES[frame.rotation]}°"
        n = row.get(following.file)
        line += (f" — next: #{n} {what}" if n is not None else
                 f" — next: {what} — a spare's photo, not on the printed list")
    if mapping.photos and mapping.photos[-1].status == "rejected":
        spoiled = mapping.photos[-1].shot.file
        line += f" (a retake of {'#' + str(row[spoiled]) if spoiled in row else spoiled})"
    print(line)
    for prompt in prompts:
        print(f"first: {prompt.instruction}")
    print(_drift(_readings(photos, config)))
    return 0


def _discarded(shot: Substitution) -> frozenset[str]:
    return frozenset(file for files in shot.discarded.values() for file in files)


def _product(args: argparse.Namespace) -> Product | None:
    fields = (args.flavour, args.shop, args.best_before, args.purchased_at)
    if any(value is None for value in fields) or (args.brand is None and not args.client_product):
        return None
    return Product(
        brand=None if args.client_product else args.brand,
        flavour=args.flavour,
        shop=args.shop,
        best_before=args.best_before,
        purchased_at=args.purchased_at,
    )


def _plan(folder: Path, session: str) -> Plan | PilotPlan | None:
    if not (folder / PLAN_FILE).is_file():
        print(f"plan: no {folder / PLAN_FILE} — run qs split-plan --session {session} first")
        return None
    try:
        return read_session_plan(folder)
    except PlanError as exc:
        print(f"plan: {exc}")
        return None


def _printed_list(folder: Path, geometry: str) -> str | None:
    """Why a dataset session cannot be mapped under the config's `geometry`, or None.

    Photo k is row k of the printed `shots-<G>.md`, and G is in every frame name, so the list
    in Ali's hand and `lighting.geometry` must be the same, and the only one (F1).
    """
    printed = sorted(path.name for path in folder.glob("shots-*.md"))
    wanted = f"shots-{geometry}.md"
    if printed == [wanted]:
        return None
    if not printed:
        return (f"no photo list in {folder} for lighting.geometry {geometry} — the import maps "
                f"photo k to row k of the printed list: qs split-plan --shot-list --geometry "
                f"{geometry}")
    if wanted not in printed:
        return (f"the config's lighting.geometry {geometry} has no printed list — {folder} "
                f"has {', '.join(printed)}; set lighting.geometry to the geometry you shot "
                "under, or every frame is named under the wrong one")
    return (f"{folder} has {' and '.join(printed)} — which one was shot is a guess; the "
            f"config's lighting.geometry {geometry} says {wanted}, so remove the list that was "
            "not printed")


def main(args: argparse.Namespace) -> int:
    """Import, or `--check`; 0 done or PASS, 1 refused or FAIL, 2 could not run."""
    try:
        config = load_config(Path(args.config)).config
    except ConfigError as exc:
        print(f"config: {exc}")
        return 2
    if config.camera.type != "phone":
        print(f"config: camera.type {config.camera.type!r} — qs import-phone takes a phone "
              "config (camera.type: phone, the phone's frame size, both ROIs in its pixels; "
              "ADR-0022)")
        return 2
    source = Path(args.source)
    if not source.is_dir():
        print(f"photos: {source} is not a folder")
        return 2
    try:
        paths = list_photos(source)
    except Unsupported as exc:
        print(f"photos: {exc}")
        return 2
    except OSError as exc:
        print(f"photos: {exc}")
        return 2
    if not paths:
        print(f"photos: no .jpg or .jpeg in {source}")
        return 2
    product = _product(args)
    if args.check:
        given = (args.flavour, args.shop, args.best_before, args.purchased_at, args.brand)
        if any(value is not None for value in given) or args.client_product:
            print("check: --check writes nothing; the product fields go with the import")
            return 2
    elif args.session is None or product is None:
        print("import: needs --session, --flavour, --shop, --best-before, --purchased-at and "
              "--brand or --client-product (or --check)")
        return 2
    if args.check and args.session is None:
        if args.redo or args.broken:
            print("check: --redo and --broken go with --session")
            return 2
        try:
            return _day_zero(_read(paths, args), config)
        except PhoneRefused as exc:
            print(f"FAIL {exc}")
            return 1

    folder = Path(args.root) / args.session
    plan = _plan(folder, args.session)
    if plan is None:  # the session must hold plan.yml, or the import refuses (#53)
        return 1
    if isinstance(plan, Plan):
        problem = _printed_list(folder, config.lighting.geometry)
        if problem is not None:
            print(f"plan: {problem} — nothing written")
            return 1
    try:
        if isinstance(plan, PilotPlan):
            # The pilot carries its own geometries; the config's lighting.geometry is the
            # dataset's, and is not used here (#34). No spares: a broken unit is discarded.
            shot = discard_broken(plan, args.broken)
        else:
            shot = substitute_spares(plan, config.lighting.geometry, args.broken)
    except PlanError as exc:
        print(f"broken: {exc}")
        return 2
    printed = _printed_rows(plan, config.lighting.geometry)
    try:
        redo = _redo_files([k for group in args.redo for k in group], printed, shot)
        photos = _read(paths, args)
        if args.check:
            return _where_am_i(photos, config, shot, redo, printed)
        if session_started(folder):
            print(f"import: {folder} already has frames — refusing to import over a started "
                  "session")
            return 1
        return _import(args, folder, config, plan, shot, redo, photos, product)
    except PhoneRefused as exc:
        print(f"import: {exc} — nothing written" if not args.check else f"check: {exc}")
        return 1


def _import(
    args: argparse.Namespace,
    folder: Path,
    config: StationConfig,
    plan: Plan | PilotPlan,
    shot: Substitution,
    redo: Sequence[str],
    photos: list[Photo],
    product: Product | None,
) -> int:
    assert product is not None
    lock = check_lock(photos, config.camera)
    if not lock.passed:
        print("\n".join(lock.lines))
        print(f"import: the EXIF lock failed — nothing written to {folder} (ADR-0022)")
        return 1
    mapping = map_photos(len(photos), shot.steps, redo, _discarded(shot))
    if len(photos) != mapping.expected_photos:
        redone = f" + {len(redo)} redo" if redo else ""
        print(f"import: expected {mapping.expected_photos} photos ({mapping.shots_total} "
              f"shots{redone}), found {len(photos)} — nothing written")
        return 1
    # Every photo is decoded once here, before any write, so a photo that cannot be decoded
    # refuses the whole import; the write pass decodes it again rather than hold hundreds of
    # full-size frames in memory. Decoding is deterministic, so both passes see equal pixels.
    readings = _readings(photos, config)
    # The paired-shooting check (ADR-0014) is the dataset's: a pilot unit's good and dmg
    # photos are a whole lamp round apart by design (docs/13 §5).
    for line in [] if isinstance(plan, PilotPlan) else gap_warnings(mapping, photos):
        print(line)

    first = photos[0]
    device = f"phone:{first.make} {first.model}"
    assert first.exposure_s is not None  # the lock passed: every photo has one
    exposure_us = int(first.exposure_s * 1_000_000)
    camera = CameraReadback(exposure_us, first.iso, None, None, verified=False)
    try:
        saved = _write(folder, photos, mapping, readings, camera)
    except OSError as exc:
        print(f"import: {exc} — stopped part-way; what was written is kept in {folder}")
        return 1
    record = SessionRecord(
        session_id=args.session,
        geometry=config.lighting.geometry,
        product=product,
        device=device,
        camera_at_open=camera,
        plan=plan,
        frames=tuple(saved),
        substitution=shot if shot.discarded else None,
    )
    try:
        path = write_session(folder, record, by="qs import-phone")
    except OSError as exc:
        print(f"import: {exc}")
        return 1
    counts = {status: sum(m.status == status for m in mapping.photos)
              for status in ("saved", "rejected", "discarded")}
    print(f"import: {len(photos)} photos → {counts['saved']} frames saved, "
          f"{counts['rejected']} rejected, {counts['discarded']} discarded · {path.name} in "
          f"{folder}")
    for unit, spare in shot.replaced.items():
        if isinstance(plan, PilotPlan):
            print(f"import: {unit} broke — discarded, because the pilot has no spares")
            continue
        split = _split(unit, plan, shot)
        if spare is None:
            print(f"import: {unit} broke and no spare was left — {split} is one short")
        else:
            print(f"import: {unit} broke — {spare} took its place in {split}")
    print(f"Copy {folder}/ to the phone NOW, before anything else (T11) — with its cloud "
          "backup still off (ADR-0022)")
    return 0


def _write(
    folder: Path,
    photos: Sequence[Photo],
    mapping: Mapping,
    readings: Sequence[float],
    camera: CameraReadback,
) -> list[_PhoneFrame]:
    """The write pass: originals, PNGs, `frames.csv`, `import.csv`. The saved frames."""
    saved: list[_PhoneFrame] = []
    with FrameLog(folder) as frames, ImportLog(folder) as manifest:
        for entry in mapping.photos:
            photo = photos[entry.index]
            assert photo.taken is not None
            original_sha = copy_original(folder, photo.path)
            pixels = decode(photo.path)
            if entry.status == "rejected":
                file, png_sha = save_rejected(folder, entry.shot.frame, entry.nth, pixels)
            else:
                file, png_sha = entry.file, save_frame(folder, entry.shot.frame, pixels)
            wall = photo.taken.astimezone(UTC)
            event = TriggerEvent(entry.index + 1, 0.0, wall, _SOURCE_ID)
            frame = _PhoneFrame(event, readings[entry.index], camera)
            frames.append_row(file, entry.shot.frame, frame, None, None, png_sha, entry.status)
            manifest.append(photo.name, original_sha, photo.taken, file, png_sha, entry.status)
            if entry.status == "saved":
                saved.append(frame)
    return saved


def _split(unit: str, plan: Plan, shot: Substitution) -> str:
    """The split a broken unit stood in — a spare's is the one of the unit it replaced."""
    while unit in plan.spare:
        unit = next(broken for broken, spare in shot.replaced.items() if spare == unit)
    return plan.split_of(unit)
