"""The command line the SDK sends with a build, for the dashboard."""

from manhattan_reasoning_gym._app import _command_line


def test_mrg_cli():
    assert _command_line(["/home/a/.venv/bin/mrg", "run", "sat.py", "--top", "t"]) == (
        "mrg run sat.py --top t"
    )


def test_windows_exe():
    argv = [r"C:\venv\Scripts\mrg.exe", "run", "sat.py"]
    assert _command_line(argv) == "mrg run sat.py"


def test_python_script():
    assert _command_line(["/home/a/sat.py", "--n", "64"]) == "python sat.py --n 64"


def test_command_line_redacts_api_key():
    assert _command_line(["mrg", "run", "a.py", "--api-key", "s3cret"]) == (
        "mrg run a.py --api-key '***'"
    )
    assert _command_line(["mrg", "run", "a.py", "--api-key=s3cret"]) == (
        "mrg run a.py '--api-key=***'"
    )


def test_quotes_arguments_with_spaces():
    assert _command_line(["mrg", "run", "my design.py"]) == "mrg run 'my design.py'"


def test_empty_argv():
    assert _command_line([]) == "python"
