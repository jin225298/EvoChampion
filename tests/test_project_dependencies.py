import tomllib
from pathlib import Path


def test_symbolic_judgment_dependencies_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    names = {dependency.split(">=", 1)[0].split("==", 1)[0] for dependency in dependencies}

    assert "math-verify" in names
    assert "sympy" in names
