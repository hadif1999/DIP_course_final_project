from rich.console import Console
from rich.table import Table

from card.pipeline import run_demo
from card.types import DemoConfig


def main() -> None:
    result = run_demo(DemoConfig())
    table = Table(title="CARD Course Demo Metrics")
    table.add_column("Image")
    table.add_column("PSNR", justify="right")
    table.add_column("SSIM", justify="right")
    for name, metric in result.metrics.items():
        table.add_row(name, f"{metric.psnr:.2f}", f"{metric.ssim:.3f}")
    console = Console()
    console.print(table)
    console.print(f"Saved figure: {result.figure_path}")


if __name__ == "__main__":
    main()
