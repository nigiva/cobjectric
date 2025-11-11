"""
Invoke tasks for cobjectric project.
"""

from invoke import Collection

from tasks.checks import check
from tasks.format import format_code
from tasks.precommit import precommit
from tasks.run import run_all
from tasks.test import test

# Create namespace
ns = Collection()

# Add tasks
ns.add_task(format_code)
ns.add_task(check)
ns.add_task(test)
ns.add_task(run_all)
ns.add_task(precommit)
