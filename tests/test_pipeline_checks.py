"""A station's own list of checks (`inspector.checks`) decides what the Pipeline's guard
accepts and what the stub inspector emits. In memory, with the same seam as test_pipeline.py."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from qs.config import LoadedConfig, load_config
from qs.contracts import CheckResult, Frame, Label, Verdict
from qs.inspector.stub import StubInspector
from station_fixtures import FakeClock, ScriptedSource, config_mapping, event, loaded_config
from test_pipeline import _pipeline

GEOMETRY = CheckResult(
    name="geometry", passed=True, value=None, limit=None, measures=(), failed_on=None
)


def _station(root: Path, checks: list[str], inspector_type: str = "stub") -> LoadedConfig:
    """The test config with `inspector.checks` set to `checks`."""
    data = config_mapping(root, inspector_type=inspector_type)
    data["inspector"]["checks"] = checks
    path = root / "station.yml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(path)


class Carrying:
    """An inspector that returns the stub's verdict with `checks` and `label` swapped in."""

    def __init__(self, inner: StubInspector, checks: tuple[str, ...], label: Label) -> None:
        self.inner = inner
        self.checks = checks
        self.label = label

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def load(self, config: object) -> None:
        self.inner.load(config)  # type: ignore[arg-type]

    def inspect(self, frame: Frame) -> Verdict:
        verdict = self.inner.inspect(frame)
        by_name = {GEOMETRY.name: GEOMETRY} | {c.name: c for c in verdict.checks}
        return dataclasses.replace(
            verdict, label=self.label, checks=tuple(by_name[n] for n in self.checks)
        )


def test_a_passed_appearance_check_alone_is_a_pass_on_an_appearance_only_station(
    tmp_path: Path,
) -> None:
    loaded = _station(tmp_path, ["appearance"])
    inspector = Carrying(StubInspector(loaded.config_hash), ("appearance",), Label.PASS)
    pipeline, _, _ = _pipeline(loaded, inspector)
    assert pipeline.handle(event(1)).verdict.label is Label.PASS


def test_a_verdict_carrying_geometry_is_refused_on_an_appearance_only_station(
    tmp_path: Path,
) -> None:
    loaded = _station(tmp_path, ["appearance"])
    inspector = Carrying(
        StubInspector(loaded.config_hash), ("geometry", "appearance"), Label.PASS
    )
    pipeline, _, _ = _pipeline(loaded, inspector)
    record = pipeline.handle(event(1))
    assert record.verdict.label is Label.NO_VERDICT
    assert record.verdict.reason == "inference"


@pytest.mark.parametrize("present", [("appearance",), ("geometry",)])
def test_a_verdict_missing_a_listed_check_is_never_a_pass(
    tmp_path: Path, present: tuple[str, ...]
) -> None:
    loaded = _station(tmp_path, ["geometry", "appearance"])
    pipeline, _, _ = _pipeline(
        loaded, Carrying(StubInspector(loaded.config_hash), present, Label.PASS)
    )
    record = pipeline.handle(event(1))
    assert record.verdict.label is Label.NO_VERDICT
    assert record.verdict.reason == "inference"


@pytest.mark.parametrize(
    ("checks", "script", "names", "label"),
    [
        (["appearance"], {}, ("appearance",), Label.PASS),
        (["appearance"], {1: "defect:appearance"}, ("appearance",), Label.DEFECT),
        (["geometry"], {}, ("geometry",), Label.PASS),
        (["geometry"], {1: "defect:geometry"}, ("geometry",), Label.DEFECT),
        (["geometry", "appearance"], {}, ("geometry", "appearance"), Label.PASS),
        (["geometry", "appearance"], {1: "defect:geometry"}, ("geometry", "appearance"),
         Label.DEFECT),
    ],
)
def test_the_stub_emits_only_the_listed_checks_and_labels_from_them(
    tmp_path: Path,
    checks: list[str],
    script: dict[int, str],
    names: tuple[str, ...],
    label: Label,
) -> None:
    loaded = _station(tmp_path, checks)
    stub = StubInspector(loaded.config_hash, script)
    stub.load(loaded.config)
    verdict = stub.inspect(_pipeline_frame())
    assert tuple(c.name for c in verdict.checks) == names
    assert verdict.label is label


def _pipeline_frame() -> Frame:
    """A frame as the Pipeline's scripted source hands it over."""
    return ScriptedSource(FakeClock(start=0.0), captures={}).capture(event(1))


def test_the_bench_config_still_gets_both_checks_from_the_stub(tmp_path: Path) -> None:
    loaded = loaded_config(tmp_path)
    pipeline, _, _ = _pipeline(loaded, StubInspector(loaded.config_hash))
    record = pipeline.handle(event(1))
    assert [c.name for c in record.verdict.checks] == ["geometry", "appearance"]
    assert record.verdict.label is Label.PASS
