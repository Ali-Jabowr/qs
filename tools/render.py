#!/usr/bin/env python3
"""Render project.yml into every status view, and validate it on the way.

    python3 tools/render.py           # validate, then write all outputs
    python3 tools/render.py --check   # validate only; non-zero exit on error

Outputs
    STATUS.md                      engineering view, English
    reports/status/<today>.md      supervisor progress report, Russian
    docs/diagrams/timeline.mmd     Mermaid gantt of the plan

project.yml is the only place status is written by hand. If a number appears in
two documents, one of them is already wrong; this script is how we avoid that.

The validator is the point of the exercise as much as the rendering is. It fails
the build on a task marked done with no evidence, and it raises a warning the
moment a stage slips past its planned end or a gate goes past due. The plan
written on 2026-09-02 slipped 15 days without anything noticing. This notices.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TODAY = date.today()

STAGE_MARK = {
    "done": "[x]", "in_progress": "[~]", "blocked": "[!]",
    "not_started": "[ ]", "dropped": "[-]",
}
TASK_MARK = {"done": "x", "doing": "~", "blocked": "!", "todo": " ", "dropped": "-"}

RU_STAGE = {
    "done": "выполнено", "in_progress": "в работе", "blocked": "заблокировано",
    "not_started": "не начато", "dropped": "снято",
}
RU_TASK = {
    "done": "выполнено", "doing": "в работе", "blocked": "заблокировано",
    "todo": "не начато", "dropped": "снято",
}
RU_GATE = {"pending": "ожидает", "passed": "пройден", "failed": "не пройден", "retired": "снят"}
RU_SEV = {"high": "высокий", "medium": "средний", "low": "низкий"}
RU_OWNER = {"client": "заказчик", "us": "мы"}


@dataclass
class Note:
    """A validator message. Bilingual, because it surfaces in both reports."""
    en: str
    ru: str


# ---------------------------------------------------------------- helpers

def parse_date(value):
    if not value:
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def flat(text):
    """Collapse a YAML folded block into one line."""
    return " ".join(str(text or "").split())


def counts(stage):
    tasks = stage.get("tasks") or []
    return sum(1 for t in tasks if t["status"] == "done"), len(tasks)


def pct(done, total):
    return 0 if total == 0 else round(100 * done / total)


def bar(done, total, width=12):
    filled = 0 if total == 0 else round(width * done / total)
    return "█" * filled + "·" * (width - filled)


def window(stage):
    start, end = parse_date(stage.get("planned_start")), parse_date(stage.get("planned_end"))
    if start and end:
        return f"{start:%d.%m} – {end:%d.%m}"
    if start:
        return f"{start:%d.%m} – ?"
    return "—"


def group_by_track(doc):
    out = {}
    for stage in doc["stages"]:
        out.setdefault(stage["track"], []).append(stage)
    return out


# ---------------------------------------------------------------- validation

# Every field here surfaces in the Russian supervisor report, so each one needs a
# Russian twin. Enforcing it here is what keeps the language policy from rotting.
RU_REQUIRED = {
    "stage": ["name_ru", "goal_ru"],
    "task": ["text_ru"],
    "gate": ["name_ru", "statement_ru"],
    "risk": ["text_ru", "response_ru"],
    "question": ["text_ru"],
}


def validate(doc):
    """Return (errors, notes). Errors make the build fail; notes are warnings."""
    errors: list[str] = []
    notes: list[Note] = []

    stage_ids = {s["id"] for s in doc["stages"]}
    gate_ids = {g["id"] for g in doc["gates"]}
    track_ids = {t["id"] for t in doc["tracks"]}

    for stage in doc["stages"]:
        sid = stage["id"]
        if stage["track"] not in track_ids:
            errors.append(f"{sid} belongs to unknown track '{stage['track']}'.")
        for field in RU_REQUIRED["stage"]:
            if not stage.get(field):
                errors.append(f"{sid} is missing '{field}' (language policy).")

        for task in stage.get("tasks") or []:
            for field in RU_REQUIRED["task"]:
                if not task.get(field):
                    errors.append(f"{sid}/{task['id']} is missing '{field}' (language policy).")
            if task["status"] == "done" and not task.get("evidence"):
                errors.append(
                    f"{sid}/{task['id']} is marked done with no evidence. "
                    f"Add a path, a commit hash or a filename, or set it back."
                )

        done, total = counts(stage)
        if stage["status"] == "done" and total and done < total:
            errors.append(f"{sid} is marked done but {total - done} of {total} tasks are not.")
        if total and done == total and stage["status"] != "done":
            notes.append(Note(
                f"{sid}: every task is done but the stage is still '{stage['status']}'.",
                f"{sid}: все задачи выполнены, но статус этапа — «{RU_STAGE[stage['status']]}».",
            ))

        end = parse_date(stage.get("planned_end"))
        if end and end < TODAY and stage["status"] not in ("done", "dropped"):
            days = (TODAY - end).days
            notes.append(Note(
                f"{sid} passed its planned end {end:%Y-%m-%d} ({days} days ago) "
                f"and is '{stage['status']}'.",
                f"{sid}: срок окончания {end:%d.%m.%Y} прошёл {days} дн. назад, "
                f"статус — «{RU_STAGE[stage['status']]}».",
            ))

        blocked_by = stage.get("blocked_by_gate")
        if blocked_by and blocked_by not in gate_ids:
            errors.append(f"{sid} is blocked by unknown gate '{blocked_by}'.")
        if stage["status"] == "blocked" and not (stage.get("blocked_on") or blocked_by):
            errors.append(f"{sid} is blocked but does not say on what.")
        if stage.get("blocked_on") and not stage.get("blocked_on_ru"):
            errors.append(f"{sid} is missing 'blocked_on_ru' (language policy).")

    for gate in doc["gates"]:
        for field in RU_REQUIRED["gate"]:
            if not gate.get(field):
                errors.append(f"Gate {gate['id']} is missing '{field}' (language policy).")
        for blocked in gate.get("blocks") or []:
            if blocked not in stage_ids:
                errors.append(f"Gate {gate['id']} blocks unknown stage '{blocked}'.")
        due = parse_date(gate.get("due"))
        if due and due < TODAY and gate["status"] == "pending":
            days = (TODAY - due).days
            notes.append(Note(
                f"Gate {gate['id']} was due {due:%Y-%m-%d} ({days} days ago) and is still pending.",
                f"Рубеж {gate['id']}: срок {due:%d.%m.%Y} прошёл {days} дн. назад, всё ещё ожидает.",
            ))

    for risk in doc["risks"]:
        for field in RU_REQUIRED["risk"]:
            if not risk.get(field):
                errors.append(f"Risk {risk['id']} is missing '{field}' (language policy).")

    for q in doc["open_questions"]:
        for field in RU_REQUIRED["question"]:
            if not q.get(field):
                errors.append(f"Question {q['id']} is missing '{field}' (language policy).")

    unsent = [q for q in doc["open_questions"]
              if q["status"] == "open" and q["owner"] == "client" and not q.get("asked_on")]
    if unsent:
        ids = ", ".join(q["id"] for q in unsent)
        notes.append(Note(
            f"{len(unsent)} client question(s) are open but have never been sent: {ids}. "
            f"Every one of them blocks a decision.",
            f"Вопросов к заказчику, ни разу не отправленных: {len(unsent)} ({ids}). "
            f"Каждый из них блокирует решение.",
        ))

    return errors, notes


# ---------------------------------------------------------------- English status

def render_status(doc, notes):
    m, out = doc["meta"], []
    w = out.append

    all_done = sum(counts(s)[0] for s in doc["stages"])
    all_total = sum(counts(s)[1] for s in doc["stages"])

    w("<!-- GENERATED by tools/render.py from project.yml. Do not edit by hand. -->")
    w(f"# {m['name']} — status")
    w("")
    w(f"**{m['title_en']}**  ")
    w(f"Owner: {m['owner']}, {m['location']} · Rendered {TODAY:%Y-%m-%d}")
    w("")
    w(f"Overall: **{all_done}/{all_total} tasks** "
      f"`{bar(all_done, all_total, 24)}` {pct(all_done, all_total)}%")
    w("")
    w(f"> {doc['client']['commercial_status_en']}  ")
    w(f"> Acceptance criterion: **{doc['client']['acceptance_criterion_en']}**")
    w("")

    if notes:
        w("## ⚠ Attention")
        w("")
        for n in notes:
            w(f"- {n.en}")
        w("")

    w("## Gates")
    w("")
    w("| Gate | Due | Status | Blocks | Statement |")
    w("|---|---|---|---|---|")
    for g in doc["gates"]:
        due = parse_date(g.get("due"))
        w(f"| **{g['id']}** — {g['name_en']} "
          f"| {f'{due:%Y-%m-%d}' if due else '—'} "
          f"| `{g['status']}` "
          f"| {', '.join(g.get('blocks') or []) or '—'} "
          f"| {flat(g.get('statement_en'))} |")
    w("")

    w("## Stages")
    w("")
    by_track = group_by_track(doc)
    for track in doc["tracks"]:
        stages = by_track.get(track["id"])
        if not stages:
            continue
        w(f"### {track['name_en']}")
        w("")
        w(f"*{flat(track['role_en'])}*")
        w("")
        w("| | Stage | Window | Progress | Status |")
        w("|---|---|---|---|---|")
        for s in stages:
            done, total = counts(s)
            if s.get("blocked_on"):
                note = f" — blocked on {s['blocked_on']}"
            elif s.get("blocked_by_gate"):
                note = f" — needs Gate {s['blocked_by_gate']}"
            else:
                note = ""
            w(f"| `{STAGE_MARK[s['status']]}` | **{s['id']}** {s['name_en']} | {window(s)} "
              f"| `{bar(done, total)}` {done}/{total} | {s['status']}{note} |")
        w("")

    w("## Task detail")
    w("")
    for stage in doc["stages"]:
        tasks = stage.get("tasks") or []
        if not tasks:
            continue
        done, total = counts(stage)
        w(f"**{stage['id']} · {stage['name_en']}** — {done}/{total}")
        w("")
        for t in tasks:
            ev = f"  ↳ `{t['evidence']}`" if t.get("evidence") else ""
            w(f"- [{TASK_MARK[t['status']]}] `{t['id']}` {t['text_en']}{ev}")
        w("")

    w("## Open questions")
    w("")
    w("| | Owner | Sent | Question | What it blocks |")
    w("|---|---|---|---|---|")
    for q in doc["open_questions"]:
        if q["status"] != "open":
            continue
        w(f"| `{q['id']}` | {q['owner']} | {q.get('asked_on') or '**never**'} "
          f"| {q['text_en']} | {flat(q.get('blocks_en'))} |")
    w("")

    w("## Risks")
    w("")
    w("| | Severity | Risk | Signal | Response |")
    w("|---|---|---|---|---|")
    for r in doc["risks"]:
        w(f"| `{r['id']}` | {r['severity']} | {r['text_en']} "
          f"| {flat(r['signal_en'])} | {flat(r['response_en'])} |")
    w("")
    w("---")
    w("")
    w("Regenerate with `python3 tools/render.py`. Source: [`project.yml`](project.yml).")
    w("")
    return "\n".join(out)


# ---------------------------------------------------------------- Russian report

def render_report_ru(doc, notes):
    m, out = doc["meta"], []
    w = out.append

    all_done = sum(counts(s)[0] for s in doc["stages"])
    all_total = sum(counts(s)[1] for s in doc["stages"])

    w("<!-- Сгенерировано tools/render.py из project.yml. Не редактировать вручную. -->")
    w(f"# Отчёт о ходе работ — {TODAY:%d.%m.%Y}")
    w("")
    w(f"**Проект:** {m['title_ru']}  ")
    w(f"**Исполнитель:** {m['owner_ru']}, {m['location_ru']}  ")
    w(f"**Заказчик:** {doc['client']['label_ru']}")
    w("")
    w(f"**Общий прогресс:** {all_done} из {all_total} задач "
      f"`{bar(all_done, all_total, 24)}` {pct(all_done, all_total)}%")
    w("")
    w(f"> {doc['client']['commercial_status_ru']}  ")
    w(f"> Критерий приёмки со слов заказчика: **{doc['client']['acceptance_criterion_ru']}**  ")
    w(f"> Модель поставки: {doc['client']['commercial_model_ru']}")
    w("")

    w("## 1. Что сделано")
    w("")
    any_done = False
    for stage in doc["stages"]:
        finished = [t for t in (stage.get("tasks") or []) if t["status"] == "done"]
        if not finished:
            continue
        any_done = True
        w(f"**{stage['id']} · {stage['name_ru']}**")
        w("")
        for t in finished:
            w(f"- {t['text_ru']} — *подтверждение: {t['evidence']}*")
        w("")
    if not any_done:
        w("Завершённых задач пока нет: проект находится на этапе проектирования.")
        w("")

    w("## 2. Что в работе")
    w("")
    active = [s for s in doc["stages"] if s["status"] == "in_progress"]
    if active:
        for stage in active:
            done, total = counts(stage)
            w(f"**{stage['id']} · {stage['name_ru']}** ({window(stage)}) — "
              f"выполнено {done} из {total}")
            w("")
            w(f"*Цель этапа.* {flat(stage['goal_ru'])}")
            w("")
            for t in stage.get("tasks") or []:
                if t["status"] in ("doing", "todo", "blocked"):
                    w(f"- {t['text_ru']} — {RU_TASK[t['status']]}")
            w("")
    else:
        w("Активных этапов нет.")
        w("")

    w("## 3. Что запланировано дальше")
    w("")
    upcoming = [s for s in doc["stages"] if s["status"] == "not_started"][:3]
    if upcoming:
        for stage in upcoming:
            w(f"**{stage['id']} · {stage['name_ru']}** ({window(stage)})")
            w("")
            w(f"*Цель этапа.* {flat(stage['goal_ru'])}")
            w("")
    else:
        w("Не начатых этапов нет.")
        w("")

    w("## 4. План по этапам")
    w("")
    w("| Этап | Направление | Сроки | Прогресс | Статус |")
    w("|---|---|---|---|---|")
    track_ru = {t["id"]: t["name_ru"] for t in doc["tracks"]}
    for s in doc["stages"]:
        done, total = counts(s)
        w(f"| **{s['id']}** {s['name_ru']} | {track_ru[s['track']]} | {window(s)} "
          f"| {done} / {total} | {RU_STAGE[s['status']]} |")
    w("")

    w("## 5. Контрольные рубежи")
    w("")
    for g in doc["gates"]:
        due = parse_date(g.get("due"))
        w(f"**Рубеж {g['id']} — {g['name_ru']}**  ")
        w(f"Срок: {f'{due:%d.%m.%Y}' if due else 'по готовности'} · "
          f"Состояние: {RU_GATE[g['status']]}" +
          (f" · Блокирует этапы: {', '.join(g['blocks'])}" if g.get("blocks") else ""))
        w("")
        w(flat(g["statement_ru"]))
        w("")
        if g.get("on_fail_ru"):
            w(f"*Если не пройден.* {flat(g['on_fail_ru'])}")
            w("")

    w("## 6. Что блокирует работу")
    w("")
    blocked_stages = [s for s in doc["stages"] if s["status"] == "blocked"]
    if blocked_stages:
        w("**Этапы, ожидающие внешних условий**")
        w("")
        for s in blocked_stages:
            reason = s.get("blocked_on_ru") or f"рубеж {s.get('blocked_by_gate')}"
            w(f"- {s['id']} · {s['name_ru']} — ожидает: {reason}")
        w("")
    open_q = [q for q in doc["open_questions"] if q["status"] == "open"]
    if open_q:
        w("**Открытые вопросы**")
        w("")
        w("| | Кому | Отправлен | Вопрос | Что блокирует |")
        w("|---|---|---|---|---|")
        for q in open_q:
            sent = q.get("asked_on") or "**не отправлен**"
            w(f"| `{q['id']}` | {RU_OWNER.get(q['owner'], q['owner'])} | {sent} "
              f"| {q['text_ru']} | {flat(q.get('blocks_ru'))} |")
        w("")

    w("## 7. Риски")
    w("")
    w("| | Уровень | Риск | Признак | Реакция |")
    w("|---|---|---|---|---|")
    for r in doc["risks"]:
        w(f"| `{r['id']}` | {RU_SEV[r['severity']]} | {r['text_ru']} "
          f"| {flat(r['signal_ru'])} | {flat(r['response_ru'])} |")
    w("")

    if notes:
        w("## 8. Замечания системы контроля плана")
        w("")
        for n in notes:
            w(f"- {n.ru}")
        w("")

    w("---")
    w("")
    w(f"Отчёт сформирован автоматически из `project.yml` {TODAY:%d.%m.%Y}. "
      f"Показатели прогресса не вводятся вручную.")
    w("")
    return "\n".join(out)


# ---------------------------------------------------------------- timeline

def render_timeline(doc):
    out = []
    w = out.append
    w("%% GENERATED by tools/render.py from project.yml. Do not edit by hand.")
    w("gantt")
    w("    title QS — plan")
    w("    dateFormat YYYY-MM-DD")
    w("    axisFormat %d.%m")
    w("    todayMarker stroke-width:2px,stroke:#d33,opacity:0.6")

    mermaid_state = {"done": "done", "in_progress": "active", "blocked": "crit"}
    by_track = group_by_track(doc)

    for track in doc["tracks"]:
        stages = [s for s in by_track.get(track["id"], []) if parse_date(s.get("planned_start"))]
        if not stages:
            continue  # an empty section renders as a stray label
        w(f"    section {track['name_en']}")
        for s in stages:
            start, end = parse_date(s["planned_start"]), parse_date(s.get("planned_end"))
            tag = mermaid_state.get(s["status"])
            prefix = f"{tag}, " if tag else ""
            label = f"{s['id']} {s['name_en']}".replace(":", " -")
            span = f"{end:%Y-%m-%d}" if end else "7d"
            w(f"    {label} :{prefix}{s['id'].lower()}, {start:%Y-%m-%d}, {span}")

    dated_gates = [g for g in doc["gates"] if parse_date(g.get("due"))]
    if dated_gates:
        w("    section Gates")
        for g in dated_gates:
            due = parse_date(g["due"])
            w(f"    Gate {g['id']} — {g['name_en']} "
              f":milestone, gate{g['id'].lower()}, {due:%Y-%m-%d}, 0d")
    w("")
    return "\n".join(out)


# ---------------------------------------------------------------- main

def main():
    check_only = "--check" in sys.argv
    doc = yaml.safe_load((ROOT / "project.yml").read_text(encoding="utf-8"))

    errors, notes = validate(doc)
    for e in errors:
        print(f"ERROR   {e}", file=sys.stderr)
    for n in notes:
        print(f"warning {n.en}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} error(s). Nothing was written.", file=sys.stderr)
        return 1
    if check_only:
        print(f"project.yml is valid. {len(notes)} warning(s).")
        return 0

    for path, content in [
        (ROOT / "STATUS.md", render_status(doc, notes)),
        (ROOT / "reports" / "status" / f"{TODAY:%Y-%m-%d}.md", render_report_ru(doc, notes)),
        (ROOT / "docs" / "diagrams" / "timeline.mmd", render_timeline(doc)),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote   {path.relative_to(ROOT)}")

    print(f"\n{len(notes)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
