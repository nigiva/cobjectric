"""Pre-commit tasks."""

from invoke import task
from rich import print


@task(
    name="precommit",
    aliases=["pc"],
    help={
        "all-files": "Run pre-commit on all files (default: only staged files)",
    },
)
def precommit(c, all_files=False):
    """
    Run pre-commit hooks (simulates pre-push stage)
    """
    print("[bold blue]Running pre-commit checks...[/bold blue]")
    if all_files:
        c.run("pre-commit run --all-files --hook-stage push", pty=True)
    else:
        c.run("pre-commit run --hook-stage push", pty=True)
