import typer
from pathlib import Path
from .config import load_cfg
from .pipeline import run
from . import discover
from .eval import describe_csv, evaluate

app = typer.Typer(add_completion=False)


@app.command()
def run_cfg(cfg: str):
    """Run with a YAML config."""
    discover()
    c = load_cfg(cfg)
    out = run(c)
    typer.echo(f"Saved: {out}")


@app.command()
def describe(path: str):
    """Describe a prediction CSV (stats + tail)."""
    info = describe_csv(Path(path))
    import json

    typer.echo(json.dumps(info, ensure_ascii=False, indent=2, default=str))


@app.command()
def eval(cfg: str, csv: str = typer.Argument(None)):
    """Evaluate predictions given a cfg (recompute labels) and optional CSV path."""
    metrics, path = evaluate(cfg, csv_path=csv)
    import json

    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
