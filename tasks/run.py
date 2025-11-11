"""Combined run tasks."""

from invoke import task
from rich import print

from tasks.checks import check
from tasks.format import format_code
from tasks.test import test


@task(
    name="all",
    pre=[format_code, check, test],
)
def run_all(c):
    """
    Run format, check, and test (inv format + inv check + inv test)
    """
    print("[bold green]✓ All tasks completed successfully![/bold green]")
