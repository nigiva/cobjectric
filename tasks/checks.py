"""Code quality checks tasks."""

from invoke import task
from rich import print


@task(name="check")
def check(c):
    """
    Run all code quality checks (typos, mypy, ruff)
    """
    print("[bold blue]Running typos checker...[/bold blue]")
    c.run("typos", pty=True)
    print("[bold blue]Running mypy on src...[/bold blue]")
    c.run("mypy src", pty=True)
    print("[bold blue]Running ruff check...[/bold blue]")
    c.run("ruff check --config=./ruff.toml .", pty=True)
