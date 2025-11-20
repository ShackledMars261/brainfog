import typer
from rich.console import Console
from typing_extensions import Annotated

from .bfcompiler.compiler import BrainFogCompiler

app: typer.Typer = typer.Typer()

console: Console = Console(soft_wrap=True)


@app.command()
def compile(
    filepath: Annotated[
        str, typer.Argument(help="The path to the input BrainFog file.")
    ],
    debug: Annotated[
        bool, typer.Option(help="Enables print statements used in debugging.")
    ] = False,
):
    """Compiles BrainFog code from a file."""
    compiler: BrainFogCompiler = BrainFogCompiler(debug=debug)
    output: str = compiler.compile_from_file(filepath)
    console.print(output)
