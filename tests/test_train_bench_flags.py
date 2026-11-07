"""`qs train` on a bench config refuses the line-only flags instead of ignoring them (#108)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import EXAMPLE, FakePatchCore
from qs.__main__ import build_parser
from qs.cmd import train


@pytest.mark.parametrize("flag", [["--footage", "footage"], ["--gap-s", "1"]])
def test_a_bench_config_refuses_a_line_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: list[str]
) -> None:
    args = build_parser().parse_args([
        "train", "--config", str(EXAMPLE), "--dataset", str(tmp_path / "dataset"),
        "--window-px", "400", "--product", "mochi", "--models", str(tmp_path / "models"), *flag,
    ])
    assert train.main(args, make_patchcore=lambda device: FakePatchCore(),
                      cuda_available=lambda: True) == 2
    assert capsys.readouterr().out.strip() == (
        "--footage and --gap-s are for a line config only (inspector.type: line)"
    )
