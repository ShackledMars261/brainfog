import typer

from .src.compile import app as compile_app
from .src.version import app as version_app
from .src.version import print_version_basic

app = typer.Typer(
    help="BrainFog: A custom high level language to make writing Brainfuck better.",
    invoke_without_command=True,
)

app.add_typer(compile_app)
app.add_typer(version_app)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Prints the version of the package.",
        is_eager=True,
    ),
):
    if version:
        print_version_basic()
        raise typer.Exit()

    if ctx.invoked_subcommand is None and not ctx.params["version"]:
        typer.echo(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":
    app()
