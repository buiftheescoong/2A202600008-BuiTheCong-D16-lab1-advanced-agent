from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/real_run",
    reflexion_attempts: int = 3,
    mode: str = "real",  # "mock" hoặc "real"
) -> None:
    examples = load_dataset(dataset)
    react = ReActAgent(mode=mode)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, mode=mode)

    react_records = []
    reflexion_records = []

    for i, example in enumerate(examples):
        print(f"[cyan]({i+1}/{len(examples)})[/cyan] {example.qid}: {example.question[:60]}...")
        react_records.append(react.run(example))
        reflexion_records.append(reflexion.run(example))

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
