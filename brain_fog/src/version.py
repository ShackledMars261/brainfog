import importlib.metadata

import typer
from rich import print

app = typer.Typer()


@app.command(help="Prints the currently installed version of the package.")
def version():
    print(f"Brain Fog CLI Version {importlib.metadata.version('brainfog')}")


def print_version_basic():
    print(f"{importlib.metadata.version('brainfog')}")
