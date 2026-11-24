"""The command line the SDK sends with a build, for the dashboard."""

import shlex

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


def test_command_line_redacts_shortened_api_key():
    # argparse accepts any unambiguous start of --api-key as the key.
    assert _command_line(["mrg", "run", "a.py", "--api-k", "s3cret"]) == (
        "mrg run a.py --api-k '***'"
    )
    assert _command_line(["mrg", "run", "a.py", "--api-ke=s3cret"]) == (
        "mrg run a.py '--api-ke=***'"
    )
    assert _command_line(["mrg", "run", "a.py", "--api-url", "http://x"]) == (
        "mrg run a.py --api-url http://x"
    )


def test_hides_other_secret_values():
    argv = [
        "/home/a/sat.py",
        "--token",
        "abc",
        "--db-password=pw",
        "--secret-key",
        "k",
        "TOKEN=x",
        "--n",
        "64",
    ]
    assert _command_line(argv) == (
        "python sat.py --token '***' '--db-password=***' --secret-key '***'"
        " 'TOKEN=***' --n 64"
    )


def test_a_flag_after_a_secret_flag_is_not_hidden():
    argv = ["mrg", "run", "a.py", "--api-key", "--top", "t"]
    assert _command_line(argv) == "mrg run a.py --api-key --top t"


def test_a_notebook_sends_no_local_paths():
    argv = [
        "/x/site-packages/ipykernel_launcher.py",
        "-f",
        "/Users/me/Library/Jupyter/runtime/kernel-1.json",
    ]
    assert _command_line(argv, module="ipykernel_launcher") == "jupyter"
    assert _command_line(argv) == "jupyter"


def test_python_dash_c_sends_no_code():
    assert _command_line(["-c", "x"]) == "python -c"


def test_python_dash_m_shows_the_module():
    argv = ["/home/a/pkg/mod.py", "--n", "64"]
    assert _command_line(argv, module="pkg.mod") == "python -m pkg.mod --n 64"
    assert _command_line(["/home/a/pkg/__main__.py"], module="pkg.__main__") == (
        "python -m pkg"
    )


def test_a_long_command_drops_whole_arguments():
    argv = ["mrg", "run", "a.py"] + ["one two three"] * 200
    line = _command_line(argv)
    assert len(line) <= 1000
    assert shlex.split(line)[-1] == "..."
    assert shlex.split(line)[:4] == ["mrg", "run", "a.py", "one two three"]
