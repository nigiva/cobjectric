from invoke import Collection, task


@task(name="serve")
def serve(ctx):
    """
    Serve MkDocs documentation locally.

    Starts a local development server that watches for changes
    and automatically rebuilds the documentation.
    """
    ctx.run("uv run mkdocs serve", pty=True)


@task(name="build")
def build(ctx):
    """
    Build MkDocs documentation.

    Generates static HTML files in the site/ directory.
    """
    ctx.run("uv run mkdocs build")


@task(name="deploy")
def deploy(ctx):
    """
    Deploy MkDocs documentation to GitHub Pages.

    Builds the documentation and deploys it to the gh-pages branch.
    """
    ctx.run("uv run mkdocs gh-deploy --force")


# Create namespace collection
docs = Collection("docs")
docs.add_task(serve)
docs.add_task(build)
docs.add_task(deploy)

