"""The validator in tools/render.py is the only thing standing between project.yml
and a plan that quietly rots. These tests keep each of its rules honest.

Run with:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import copy
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import render  # noqa: E402


@pytest.fixture(scope="module")
def doc():
    return yaml.safe_load((ROOT / "project.yml").read_text(encoding="utf-8"))


@pytest.fixture
def d(doc):
    return copy.deepcopy(doc)


def errors_of(document):
    return render.validate(document)[0]


def notes_of(document):
    return render.validate(document)[1]


def find_stage(document, stage_id):
    return next(s for s in document["stages"] if s["id"] == stage_id)


# ---------------------------------------------------------------- the real file

def test_the_committed_project_file_is_valid(doc):
    assert errors_of(doc) == []


def test_every_id_is_unique(doc):
    stage_ids = [s["id"] for s in doc["stages"]]
    assert len(stage_ids) == len(set(stage_ids))
    task_ids = [t["id"] for s in doc["stages"] for t in (s.get("tasks") or [])]
    assert len(task_ids) == len(set(task_ids))


def test_a_task_id_is_prefixed_by_its_stage(doc):
    for stage in doc["stages"]:
        for task in stage.get("tasks") or []:
            assert task["id"].startswith(stage["id"] + ".")


def test_all_three_views_render(doc, tmp_path):
    notes = notes_of(doc)
    assert render.render_status(doc, notes).startswith("<!-- GENERATED")
    assert "Отчёт о ходе работ" in render.render_report_ru(doc, notes)
    assert render.render_timeline(doc).splitlines()[1] == "gantt"


def test_russian_report_contains_no_stray_english_task_text(doc):
    """Every task surfaced in the report must come from a *_ru field."""
    report = render.render_report_ru(doc, notes_of(doc))
    for stage in doc["stages"]:
        for task in stage.get("tasks") or []:
            if task["text_en"] != task["text_ru"] and task["text_en"] in report:
                pytest.fail(f"{task['id']} leaked English into the Russian report")


# ---------------------------------------------------------------- each rule

def test_done_without_evidence_is_an_error(d):
    task = find_stage(d, "S0")["tasks"][0]
    task["status"] = "done"
    task["evidence"] = None
    assert any("no evidence" in e for e in errors_of(d))


def test_done_with_evidence_is_accepted(d):
    task = find_stage(d, "S0")["tasks"][0]
    task["status"] = "done"
    task["evidence"] = "commit abc1234"
    assert not any("no evidence" in e for e in errors_of(d))


def test_missing_russian_twin_is_an_error(d):
    del find_stage(d, "S1")["tasks"][0]["text_ru"]
    assert any("language policy" in e for e in errors_of(d))


def test_stage_marked_done_with_open_tasks_is_an_error(d):
    find_stage(d, "S1")["status"] = "done"
    assert any("marked done but" in e for e in errors_of(d))


def test_unknown_gate_reference_is_an_error(d):
    find_stage(d, "S1")["blocked_by_gate"] = "ZZ"
    assert any("unknown gate" in e for e in errors_of(d))


def test_gate_blocking_unknown_stage_is_an_error(d):
    d["gates"][0]["blocks"] = ["S99"]
    assert any("unknown stage" in e for e in errors_of(d))


def test_blocked_stage_must_say_what_blocks_it(d):
    stage = find_stage(d, "S1")
    stage["status"] = "blocked"
    assert any("does not say on what" in e for e in errors_of(d))


def test_blocked_reason_needs_a_russian_twin(d):
    stage = find_stage(d, "S1")
    stage["status"] = "blocked"
    stage["blocked_on"] = "something"
    assert any("blocked_on_ru" in e for e in errors_of(d))


# ---------------------------------------------------------------- drift warnings

def test_a_stage_past_its_planned_end_raises_a_warning(d):
    stage = find_stage(d, "S1")
    stage["planned_end"] = date.today() - timedelta(days=15)
    assert any("passed its planned end" in n.en for n in notes_of(d))
    assert any("срок окончания" in n.ru for n in notes_of(d))


def test_an_overdue_gate_raises_a_warning(d):
    d["gates"][0]["due"] = date.today() - timedelta(days=3)
    d["gates"][0]["status"] = "pending"
    assert any("was due" in n.en for n in notes_of(d))


def test_a_passed_gate_in_the_past_is_not_a_warning(d):
    d["gates"][0]["due"] = date.today() - timedelta(days=3)
    d["gates"][0]["status"] = "passed"
    assert not any("was due" in n.en for n in notes_of(d))


def test_unsent_client_questions_raise_a_warning(d):
    assert any("never been sent" in n.en for n in notes_of(d))


def test_sent_client_questions_do_not(d):
    for q in d["open_questions"]:
        q["asked_on"] = "2026-09-17"
    assert not any("never been sent" in n.en for n in notes_of(d))


def test_every_warning_is_bilingual(d):
    find_stage(d, "S1")["planned_end"] = date.today() - timedelta(days=1)
    d["gates"][0]["due"] = date.today() - timedelta(days=1)
    for note in notes_of(d):
        assert note.en and note.ru and note.en != note.ru
