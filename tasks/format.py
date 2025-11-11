"""Format tasks."""

from invoke import task
from rich import print


@task(name="format")
def format_code(c):
    """
    Format the code using isort and black (excludes Jupyter notebooks)
    """
    print("[bold blue]Running isort...[/bold blue]")
    c.run("isort .", pty=True)
    print("[bold blue]Running black...[/bold blue]")
    c.run("black .", pty=True)
