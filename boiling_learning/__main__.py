import typer

from boiling_learning.app.studies.validation import validate

app = typer.Typer()

app.command()(validate)


@app.command('all')
def run_all() -> None:
    import boiling_learning.app.main


if __name__ == '__main__':
    app()
