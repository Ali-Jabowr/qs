"""The six security tests of docs/17 §6, one per item, none needing a camera.

1. **Offline run.** `qs station --replay` runs 50 frames in a child process whose audit hook
   kills it on any attempt to create a socket, connect, resolve a name or open a URL with
   `urllib` — installed before `qs` is imported. The stub variant is in `make check`; the
   variant with the appearance Inspector and the real scorer, on a bundle `qs train` makes
   in the test, is marked ``env``.
2. **No network code.** No `socket`, `http`, `urllib` or `requests` import in `src/qs` outside
   `src/qs/sources/`, the `FrameSource` implementations (NFR-07).
3. **Bundle mismatch.** A bundle changed after it was written stops the station at start-up,
   before any evidence is written.
4. **Validation-only refused.** A validation-only bundle named by the config is refused, and
   `qs station` has no validation mode.
5. **Append-only.** `UPDATE` and `DELETE` on `inspection` and `inspection_check` fail.
6. **Nothing tracked.** The repo's `.gitignore` ignores `data/`, `evidence/`, `models/`,
   `wheelhouse/`, `.torch/` and `.hf/` and a file inside each, and nothing under them but
   `data/.gitkeep` is tracked.

Tests 1, 2 and 6 each show they can fail: the hook kills a process that opens a socket, a
planted import in a copy of the tree is found, and a `.gitignore` missing a line leaves its
paths unignored.
"""

from __future__ import annotations

import ast
import gc
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from evidence_records import fast_sqlite
from qs.__main__ import build_parser
from qs.cmd.station import main
from qs.inspector.model import Scored
from station_fixtures import (
    disc_frame,
    rows,
    station_args,
    write_config,
    write_fake_bundle,
    write_frames,
    write_png,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "qs"

# --- 1 · offline run ------------------------------------------------------------------------

NETWORK_EXIT = 97
NETWORK_EVENTS = (
    "socket.__new__", "socket.connect", "socket.getaddrinfo", "urllib.Request",
    "socket.bind", "socket.sendto", "socket.sendmsg", "socket.gethostbyname",
    "socket.gethostbyaddr", "http.client.connect",
)
# Run as `python -c HARNESS <qs arguments>`, or `HARNESS --probe <kind>` to open the network.
HARNESS = textwrap.dedent(
    f"""
    import os
    import sys

    BLOCKED = frozenset({NETWORK_EVENTS!r})

    def refuse(event, args):
        if event in BLOCKED:
            os.write(2, f"network attempt: {{event}} {{args!r}}\\n".encode())
            os._exit({NETWORK_EXIT})

    sys.addaudithook(refuse)
    if sys.argv[1] == "--probe":
        kind = sys.argv[2]
        if kind == "socket":
            import socket
            socket.socket()
        elif kind == "resolve":
            import socket
            socket.getaddrinfo("localhost", 80)
        elif kind == "request":
            import urllib.request
            urllib.request.urlopen("http://localhost/")
        sys.exit(0)
    from qs.__main__ import run

    sys.exit(run(sys.argv[1:]))
    """
)


def offline(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """`HARNESS` in a child process from `cwd`, on this checkout's `src`."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [env.get("PYTHONPATH")])]
    )
    return subprocess.run(
        [sys.executable, "-c", HARNESS, *argv],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=600,
    )


@pytest.mark.parametrize("kind", ["socket", "resolve", "request"])
def test_the_offline_harness_kills_a_process_that_reaches_for_the_network(
    tmp_path: Path, kind: str
) -> None:
    done = offline(["--probe", kind], tmp_path)
    assert done.returncode == NETWORK_EXIT, done.stderr
    assert "network attempt" in done.stderr


def test_offline_run_50_replayed_frames_through_qs_station_open_no_socket(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 50)
    done = offline(
        ["station", "--config", str(config), "--replay", str(tmp_path / "shoot")], tmp_path
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "network attempt" not in done.stderr
    records = rows(tmp_path / "evidence" / "qs.db", "SELECT seq FROM inspection ORDER BY seq")
    assert [r["seq"] for r in records] == list(range(1, 51))


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, str]]:
    """A working directory with a real bundle in `models/`, trained through `qs train` on 10
    good synthetic discs; the GPU memory the training held is released before the station
    starts in its own process."""
    from qs.cmd import train
    from station_fixtures import EXAMPLE

    workdir = tmp_path_factory.mktemp("offline-appearance")
    good = workdir / "dataset" / "train" / "good"
    good.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for unit in range(1, 11):
        centre = (960 + int(rng.integers(-40, 41)), 540 + int(rng.integers(-40, 41)))
        write_png(good / f"u{unit:03d}_good_G2_r0.png",
                  disc_frame(int(rng.integers(185, 196)), centre))
    args = build_parser().parse_args([
        "train", "--config", str(EXAMPLE), "--dataset", str(workdir / "dataset"),
        "--window-px", "400", "--product", "mochi", "--models", str(workdir / "models"),
    ])
    assert train.main(args) == 0
    [bundle] = (workdir / "models").iterdir()
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    yield workdir, bundle.name


@pytest.mark.env
def test_offline_run_with_the_appearance_inspector_and_the_real_scorer(
    trained: tuple[Path, str],
) -> None:
    from appearance_fixtures import config_naming

    workdir, model_id = trained
    config_naming(workdir, model_id)  # the example config, naming the bundle
    shoot = workdir / "shoot"
    shoot.mkdir()
    rng = np.random.default_rng(11)
    for index in range(1, 51):
        centre = (960 + int(rng.integers(-40, 41)), 540 + int(rng.integers(-40, 41)))
        write_png(shoot / f"f{index:04d}.png", disc_frame(int(rng.integers(185, 196)), centre))
    done = offline(
        ["station", "--config", str(workdir / "station.yml"), "--replay", str(shoot)], workdir
    )
    print(done.stdout)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "network attempt" not in done.stderr
    records = rows(workdir / "evidence" / "qs.db",
                   "SELECT seq, label, model_id FROM inspection ORDER BY seq")
    assert [r["seq"] for r in records] == list(range(1, 51))
    assert {r["model_id"] for r in records} == {model_id}
    assert all(r["label"] in ("pass", "defect") for r in records)  # every unit was judged


# --- 2 · no network code --------------------------------------------------------------------

NETWORK_MODULES = frozenset({"socket", "http", "urllib", "requests"})
FRAME_SOURCES = "sources"  # src/qs/sources/: the FrameSource implementations


def _imported(node: ast.AST) -> list[str]:
    """The absolute modules a node imports: `import`, `from … import`, and `__import__` or
    `importlib.import_module` of a string literal."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.level == 0 and node.module else []
    if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
        func = node.func
        dynamic = (isinstance(func, ast.Name) and func.id in ("__import__", "import_module")) or (
            isinstance(func, ast.Attribute) and func.attr == "import_module"
        )
        if dynamic and isinstance(node.args[0].value, str):
            return [node.args[0].value]
    return []


def network_imports(package: Path) -> list[str]:
    """Every import of a network module in `package` outside its `sources/`, as
    `path:line module`, sorted."""
    found = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if relative.parts[0] == FRAME_SOURCES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for module in _imported(node):
                if module.split(".")[0] in NETWORK_MODULES:
                    found.append(f"{relative.as_posix()}:{node.lineno} {module}")
    return found


def test_no_network_import_in_src_qs_outside_the_frame_sources() -> None:
    assert (PACKAGE / FRAME_SOURCES / "webcam.py").is_file()  # the tree is where we look
    assert network_imports(PACKAGE) == []


@pytest.mark.parametrize(
    ("planted", "text", "module"),
    [
        ("planted.py", "import socket\n", "socket"),
        ("inspector/planted.py", "from urllib.request import urlopen\n", "urllib.request"),
        ("cmd/planted.py", "def later():\n    import http.client as client\n", "http.client"),
        ("planted.py", "import os, requests\n", "requests"),
        ("planted.py", "import importlib\nimportlib.import_module('socket')\n", "socket"),
        ("planted.py", "__import__('urllib.parse')\n", "urllib.parse"),
    ],
)
def test_a_planted_network_import_in_a_copy_of_the_tree_is_found(
    tmp_path: Path, planted: str, text: str, module: str
) -> None:
    copy = tmp_path / "qs"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / planted).write_text(text, encoding="utf-8")
    [finding] = network_imports(copy)
    assert finding.startswith(f"{planted}:") and finding.endswith(f" {module}")


def test_a_network_import_inside_the_frame_sources_is_allowed(tmp_path: Path) -> None:
    copy = tmp_path / "qs"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / FRAME_SOURCES / "gige.py").write_text("import socket\n", encoding="utf-8")
    assert network_imports(copy) == []


# --- 3 and 4 · the station runs only the bundle its config names ----------------------------


class NeverScorer:
    """A scorer factory that records being asked; the refusals come before it."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, device: str, bank: np.ndarray) -> NeverScorer:
        self.calls += 1
        return self

    def score(self, window_input: np.ndarray) -> Scored:
        anomaly_map = np.zeros(window_input.shape[1:], dtype=np.float32)
        return Scored(anomaly_map, 0.0, backbone_ms=1.0, search_ms=2.0)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`tmp_path` as the working directory, so `models/` is there; three frames in `shoot/`."""
    fast_sqlite(monkeypatch)
    monkeypatch.chdir(tmp_path)
    write_frames(tmp_path / "shoot", 3)
    return tmp_path


def _refused(workdir: Path, config: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Run the station; assert exit 2, one line, no scorer, no evidence; return the line."""
    scorer = NeverScorer()
    capsys.readouterr()
    code = main(station_args(config, workdir / "shoot"), make_scorer=scorer,
                cuda_available=lambda: True)
    out = capsys.readouterr().out
    assert code == 2, out
    assert out.count("\n") == 1 and "Traceback" not in out, out
    assert scorer.calls == 0
    assert not (workdir / "evidence").exists()  # refused before any evidence was written
    return out


def test_a_bundle_changed_after_it_was_written_stops_the_station_at_start_up(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_id = write_fake_bundle(workdir / "models")
    config = write_config(workdir, inspector_type="appearance", model=model_id)
    (workdir / "control").mkdir()
    control = write_config(workdir / "control", inspector_type="appearance", model=model_id)
    assert main(station_args(control, workdir / "shoot"), make_scorer=NeverScorer(),
                cuda_available=lambda: True) == 0  # the untouched bundle starts

    bank = workdir / "models" / model_id / "memory_bank.pt"
    data = bytearray(bank.read_bytes())
    data[len(data) // 2] ^= 0x01  # one byte
    bank.write_bytes(bytes(data))

    out = _refused(workdir, config, capsys)
    assert model_id in out
    assert "does not match its model_id" in out


def test_a_validation_only_bundle_is_refused_and_there_is_no_validation_mode(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_id = write_fake_bundle(workdir / "models", data_source="external:mvtec-ad")
    config = write_config(workdir, inspector_type="appearance", model=model_id)
    out = _refused(workdir, config, capsys)
    assert model_id in out
    assert "validation-only" in out

    for flag in ("--validation", "--validation-mode", "--valid"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["station", "--config", str(config), "--replay", str(workdir / "shoot"), flag]
            )
    capsys.readouterr()
    with pytest.raises(SystemExit):
        build_parser().parse_args(["station", "--help"])
    usage = capsys.readouterr().out
    assert "--replay" in usage and "valid" not in usage


# --- 5 · append-only ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE inspection SET label = 'pass'",
        "DELETE FROM inspection",
        "UPDATE inspection_check SET passed = 1",
        "DELETE FROM inspection_check",
    ],
)
def test_the_evidence_a_replay_just_wrote_is_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statement: str
) -> None:
    fast_sqlite(monkeypatch)
    config = write_config(tmp_path)
    write_frames(tmp_path / "shoot", 3)
    assert main(station_args(config, tmp_path / "shoot")) == 0
    database = tmp_path / "evidence" / "qs.db"
    before = rows(database, "SELECT * FROM inspection_check ORDER BY inspection_id, name")
    assert len(before) == 6

    conn = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="is append-only"):
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()
    after = rows(database, "SELECT * FROM inspection_check ORDER BY inspection_id, name")
    assert [tuple(r) for r in after] == [tuple(r) for r in before]
    assert len(rows(database, "SELECT id FROM inspection")) == 3


# --- 6 · nothing tracked --------------------------------------------------------------------

MUST_IGNORE = (
    # data/ itself holds the tracked data/.gitkeep, so what is ignored is everything in it
    "data/raw/", "data/raw/2026-09-25-mvp/u001_good_G2_r0.png", "data/golden.png",
    "evidence/", "evidence/qs.db",
    "models/", "models/patchcore-mochi-v1-3f9a1c2b7d4e/memory_bank.pt",
    "wheelhouse/", "wheelhouse/torch-2.5.1-cp312-cp312-linux_x86_64.whl",
    ".torch/", ".torch/hub/checkpoints/weights.pth",
    ".hf/", ".hf/hub/model.safetensors",
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """git in `repo`, with no user-wide excludes: only the repo's own rules count."""
    return subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", *args],
        cwd=repo, capture_output=True, text=True, timeout=30,
    )


def not_ignored(repo: Path, paths: tuple[str, ...]) -> list[str]:
    """Each of `paths` — created first, a trailing `/` as a directory — that git does not
    ignore in `repo`."""
    for path in paths:
        target = repo / path
        if path.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x89PNG not really")
    done = _git(repo, "check-ignore", "--", *paths)
    assert done.returncode in (0, 1), done.stderr
    ignored = set(done.stdout.splitlines())
    return [path for path in paths if path not in ignored]


def _scratch_repo(tmp_path: Path, gitignore: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    done = _git(repo, "init", "-q")
    assert done.returncode == 0, done.stderr
    (repo / ".gitignore").write_text(gitignore, encoding="utf-8")
    return repo


@pytest.fixture(scope="module")
def work_tree() -> Path:
    """The repo root, or a skip when the tests do not run in a git work tree."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    done = _git(ROOT, "rev-parse", "--is-inside-work-tree")
    if done.returncode != 0 or done.stdout.strip() != "true":
        pytest.skip("not in a git work tree: nothing can be tracked")
    return ROOT


def test_the_repos_gitignore_ignores_every_data_directory_and_a_file_in_each(
    work_tree: Path, tmp_path: Path
) -> None:
    # The rule is checked on real directories in a scratch repo, so what happens to exist
    # in this checkout (a symlinked .hf, an absent wheelhouse) cannot change the answer.
    rules = (work_tree / ".gitignore").read_text(encoding="utf-8")
    assert not_ignored(_scratch_repo(tmp_path, rules), MUST_IGNORE) == []


def test_nothing_under_the_data_directories_is_tracked(work_tree: Path) -> None:
    done = _git(work_tree, "ls-files", "--", "data", "evidence", "models", "wheelhouse",
                ".torch", ".hf")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["data/.gitkeep"]


@pytest.mark.parametrize("dropped", ["models/", ".hf", "data/*"])
def test_a_gitignore_missing_a_line_leaves_its_paths_unignored(
    work_tree: Path, tmp_path: Path, dropped: str
) -> None:
    rules = (work_tree / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert dropped in rules
    repo = _scratch_repo(tmp_path, "\n".join(r for r in rules if r != dropped) + "\n")
    missing = not_ignored(repo, MUST_IGNORE)
    prefix = dropped.rstrip("*")
    assert missing and all(path.startswith(prefix) for path in missing)
    assert missing == [path for path in MUST_IGNORE if path.startswith(prefix)]
