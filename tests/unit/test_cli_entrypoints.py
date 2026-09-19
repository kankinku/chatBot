"""CLI 진입점의 오프라인 실행 계약 테스트."""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )


def test_evaluate_qa_help_uses_local_scripts_package():
    result = _run_help("evaluate_qa_unified.py")

    assert result.returncode == 0, result.stderr
    assert "usage:" in (result.stdout or "")


def test_interactive_help_uses_local_scripts_package():
    result = _run_help("test_chatbot_interactive.py")

    assert result.returncode == 0, result.stderr
    assert "usage:" in (result.stdout or "")


def test_build_corpus_help_is_available_offline():
    result = _run_help("build_corpus.py")

    assert result.returncode == 0, result.stderr
    assert "usage:" in (result.stdout or "")


def test_local_scripts_package_is_resolved_before_installed_package():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.unified_evaluation as module; print(module.__file__)",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(PROJECT_ROOT / "scripts").lower() in (result.stdout or "").lower()


def test_local_scripts_package_wins_against_shadow_package():
    probe = """
from pathlib import Path
import sys
import tempfile

project_root = Path(sys.argv[1])
with tempfile.TemporaryDirectory() as directory:
    shadow_root = Path(directory)
    shadow_scripts = shadow_root / "scripts"
    shadow_scripts.mkdir()
    (shadow_scripts / "__init__.py").write_text("SHADOW = True", encoding="utf-8")
    (shadow_scripts / "unified_evaluation.py").write_text("SHADOW = True", encoding="utf-8")
    sys.path[:] = [str(project_root), str(shadow_root)] + [entry for entry in sys.path if entry not in {str(project_root), str(shadow_root)}]
    import scripts.unified_evaluation as module
    print(module.__file__)
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", probe, str(PROJECT_ROOT)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(PROJECT_ROOT / "scripts").lower() in (result.stdout or "").lower()
