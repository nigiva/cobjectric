"""Test tasks."""

from invoke import task
from rich import print


@task(name="test")
def test(c):
    """
    Run tests with pytest and coverage
    """
    print("[bold blue]Running pytest with coverage...[/bold blue]")
    c.run("pytest", pty=True)
