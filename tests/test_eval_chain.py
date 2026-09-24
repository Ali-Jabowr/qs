"""The whole P08 chain, end to end, twice (NFR-08, A5, FR-55, FR-56; ticket #83).

The synthetic lean session goes through every command's own `main()` in Saturday's order:
`qs build` → P07's `qs train` with the numpy PatchCore → `qs calibrate` → `qs evaluate`, then
`qs evaluate` again on the same golden set, bundle and config. The two reports must be
identical outside the latency section, apart from the changed-units line of the second
saying "no unit changed"; the two CSVs byte-identical. The raw session and the golden set
are hashed file by file and must be untouched, and the working directory is a scratch one,
so a default path that slipped through would land in `tmp_path`, never in the repo.

The session is `test_evaluate`'s: the lean plan drawn so that no two units share bytes (E1
compares the bundle's image hashes with the golden frames') and the derived area range is
wider than one pixel. The scorer is its blot scorer, the one the calibration was set with.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
from pathlib import Path

import pytest

from conftest import FakePatchCore
from qs.__main__ import build_parser
from qs.cmd import build, evaluate, train
from test_calibrate import PRODUCT
from test_evaluate import make_scorer, run_calibrate, write_eval_session

REPO = Path(__file__).resolve().parents[1]


def _hashes(root: Path) -> dict[str, str]:
    """Every file under `root`, by relative path, to its SHA-256."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _repo_outputs() -> set[str]:
    """Every path under the repo's own output directories."""
    return {str(p) for name in ("reports", "models", "data")
            for p in (REPO / name).rglob("*")}


def _main(command: object, argv: list[str], **seams: object) -> int:
    args = build_parser().parse_args(argv)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code: int = command.main(args, **seams)  # type: ignore[attr-defined]
    assert "Traceback" not in buffer.getvalue(), buffer.getvalue()
    assert code == 0 or argv[0] == "evaluate", (argv[0], buffer.getvalue())
    return code


def _split(text: str) -> tuple[str, str]:
    """(the body before the latency section, the changed-units section)."""
    body = text.split("## Latency — measured, not reproducible")[0]
    changed = body.split("## Changed units\n")[1].split("\n\n")[0]
    return body, changed


def test_build_train_calibrate_evaluate_twice_reproduces_the_report(
    tmp_path: Path, bench_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before_repo = _repo_outputs()
    monkeypatch.chdir(tmp_path)
    _, session = write_eval_session(tmp_path / "raw")
    raw = _hashes(session)
    config = tmp_path / "input.yml"
    config.write_text(bench_config.read_text(encoding="utf-8").replace(
        "product: mochi-<flavour>-v1", f"product: {PRODUCT}"), encoding="utf-8")
    data = tmp_path / "data"
    models = data / "models"

    _main(build, ["build", "--session", str(session), "--product", PRODUCT,
                  "--data", str(data)])
    golden = data / "golden" / "mvp-2026-09-25"
    dataset = data / "datasets" / PRODUCT / "G2"
    frozen = _hashes(golden)
    assert frozen and "manifest.sha256" in frozen

    _main(train, ["train", "--config", str(config), "--dataset", str(dataset),
                  "--window-px", "64", "--input-px", "32", "--product", PRODUCT,
                  "--models", str(models)],
          make_patchcore=lambda device: FakePatchCore(), cuda_available=lambda: True)
    [model_id] = [p.name for p in models.iterdir() if p.is_dir()]

    calibrated = data / "station.yml"
    assert run_calibrate(config, dataset, model_id, models, calibrated,
                         data / "reports" / "calib") == 0

    reports = data / "reports" / "eval"
    argv = ["evaluate", "--config", str(calibrated), "--golden", str(golden),
            "--models", str(models), "--report-dir", str(reports)]
    seams = {"make_scorer": make_scorer, "cuda_available": lambda: True}
    assert _main(evaluate, argv, **seams) == 0
    assert _main(evaluate, argv, **seams) == 0

    [first, second] = sorted(reports.glob("*.md"), key=lambda p: len(p.name))
    assert second.name == f"{first.stem}-2.md"
    body1, changed1 = _split(first.read_text(encoding="utf-8"))
    body2, changed2 = _split(second.read_text(encoding="utf-8"))
    assert changed1 == "- no previous run on this golden set."
    assert changed2.startswith(f"- no unit changed since `{first.stem}.csv`")
    assert "\n" not in changed2
    assert body1.replace(changed1, changed2) == body2  # NFR-08: identical but for that line
    assert first.with_suffix(".csv").read_bytes() == second.with_suffix(".csv").read_bytes()

    assert _hashes(golden) == frozen  # the golden set is untouched by the whole chain
    assert _hashes(session) == raw  # and the raw session
    assert _repo_outputs() == before_repo  # nothing written into the repo
    assert os.getcwd() == str(tmp_path)
