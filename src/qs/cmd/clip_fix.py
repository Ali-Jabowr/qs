"""`qs clip-fix VIDEO`: renumber a clip's frames so a frame number means one picture (#173).

    qs clip-fix VIDEO

When frames are missing, or the container's count disagrees with the frames that decode (a true
frame rate off the nominal one does it too), a frame read in order is not the frame read by
seeking. It fixes a clip with either fault and refuses one with neither. This writes
`<name>-fixed.mkv` beside the clip with `ffmpeg -c copy -bsf:v setts=ts=N/(FPS*TB)`: every
picture is kept, nothing is re-encoded, and the timestamps become evenly spaced. The sidecar is copied to `<name>-fixed.json` with the new
`sha256`, `fixed_from` (the original's `sha256`) and `missing_frames`. The original is never
changed and an existing output is never overwritten. The output is counted; if its decoded count
still differs from its container's it is deleted and the exit code is 2. `missing_frames` is the
count from timestamps, so a rate mismatch reads 0.

Exit codes: 0 written · 2 refused or failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qs.cmd.clip_check import ClipCheckError, _sidecar
from qs.inspector.model import sha256_of
from qs.lane.split import ClipCounts, count_clip


def add_parser(commands: Any) -> None:
    """Register `clip-fix` on the `qs` subcommands."""
    parser = commands.add_parser("clip-fix", help="renumber a clip's frames without re-encoding")
    parser.add_argument("video", metavar="VIDEO", help="the clip; its sidecar sits beside it")


def _run_ffmpeg(argv: list[str]) -> int:
    try:
        return subprocess.run(argv, check=False).returncode
    except OSError as exc:
        raise ClipCheckError(f"cannot start ffmpeg: {exc}") from exc


def _fps(sidecar: dict[str, Any], path: Path) -> int:
    fps = sidecar.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ClipCheckError(f"the sidecar {path} has no whole-number fps")
    return fps


def main(
    args: argparse.Namespace,
    *,
    run_ffmpeg: Callable[[list[str]], int] = _run_ffmpeg,
    count: Callable[[Path], ClipCounts | None] = count_clip,
    file_hash: Callable[[Path], str] = sha256_of,
) -> int:
    video = Path(args.video)
    out = video.with_name(f"{video.stem}-fixed.mkv")
    out_sidecar = out.with_suffix(".json")
    try:
        for path in (out, out_sidecar):
            if path.exists():
                raise ClipCheckError(f"{path} already exists; nothing overwritten")
        sidecar = _sidecar(video)
        fps = _fps(sidecar, video.with_suffix(".json"))
        counts = count(video)
        if counts is None:
            raise ClipCheckError(f"cannot open the video {video}")
        missing = counts.missing
        if missing == 0 and counts.decoded == counts.container:
            raise ClipCheckError(
                f"{video} has no missing frames and its counts agree; nothing to fix"
            )
        code = run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-c", "copy", "-bsf:v", f"setts=ts=N/({fps}*TB)", str(out),
        ])
        if code != 0:
            out.unlink(missing_ok=True)
            raise ClipCheckError(f"ffmpeg exited with code {code}")
        after = count(out)
        if after is None or after.decoded != after.container:
            out.unlink(missing_ok=True)
            seen = "cannot be opened" if after is None else (
                f"decodes {after.decoded} frames but its container says {after.container}"
            )
            raise ClipCheckError(f"{out} {seen} after the fix; output deleted")
        body = {
            **sidecar,
            "sha256": file_hash(out),
            "fixed_from": sidecar.get("sha256"),
            "missing_frames": missing,
        }
        out_sidecar.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    except (ClipCheckError, OSError) as exc:
        print(f"clip-fix: {exc}")
        return 2
    print(f"clip-fix: {missing} missing frames; wrote {out} and {out_sidecar}")
    return 0
