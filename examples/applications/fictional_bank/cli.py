#!/usr/bin/env python3
"""Command-line interface for Fictional Bank portfolio analysis.

This CLI provides easy access to all portfolio analysis capabilities:
- Portfolio loading and summary
- XVA calculations
- Reporting (HTML, CSV, Excel)
- Visualization generation
- Stress testing
- Sensitivity analysis
- Batch operations
"""
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Initialize Rich console for beautiful output
console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="Fictional Bank Portfolio CLI")
def cli():
    """
    Fictional Bank Portfolio Analysis CLI

    A comprehensive toolkit for portfolio management, risk analytics,
    and XVA calculations using Neutryx.
    """
    pass


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="snapshots",
    help="Directory to save portfolio snapshot",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def load(output_dir, verbose):
    """Load and display the fictional bank portfolio."""
    console.print(Panel.fit("Loading Fictional Bank Portfolio", style="bold blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading portfolio...", total=None)

        # Import and execute
        from neutryx.tests.fixtures.fictional_portfolio import (
            create_fictional_portfolio,
            get_portfolio_summary,
        )

        portfolio, book_hierarchy = create_fictional_portfolio()
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        progress.update(task, completed=True)

    # Display summary
    console.print(f"\n[bold green]✓ Portfolio loaded:[/bold green] {portfolio.name}\n")

    # Statistics table
    stats_table = Table(title="Portfolio Statistics", show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats = summary["statistics"]
    stats_table.add_row("Counterparties", str(stats["counterparties"]))
    stats_table.add_row("Netting Sets", str(stats["netting_sets"]))
    stats_table.add_row("Total Trades", str(stats["trades"]))
    stats_table.add_row("Active Trades", str(stats["active_trades"]))
    stats_table.add_row("Total MTM", f"${summary['total_mtm']:,.2f}")
    stats_table.add_row("Gross Notional", f"${summary['gross_notional']:,.2f}")

    console.print(stats_table)

    # Save snapshot if requested
    if output_dir:
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        snapshot_file = output_path / "portfolio_snapshot.json"
        portfolio_data = portfolio.model_dump(mode="json")

        with open(snapshot_file, "w") as f:
            json.dump(portfolio_data, f, indent=2, default=str)

        console.print(f"\n[green]✓ Snapshot saved:[/green] {snapshot_file}")


@cli.command()
@click.option(
    "--api-url",
    default="http://localhost:8000",
    help="Neutryx API URL",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="reports",
    help="Directory to save XVA results",
)
def xva(api_url, output_dir):
    """Compute XVA metrics via Neutryx API."""
    console.print(Panel.fit("Computing XVA Metrics", style="bold blue"))

    import requests

    # Check API health
    console.print(f"\nConnecting to API at {api_url}...")
    try:
        response = requests.get(f"{api_url}/docs", timeout=2)
        if response.status_code == 200:
            console.print("[green]✓ API is running[/green]")
        else:
            console.print("[red]✗ API returned unexpected status[/red]")
            return
    except requests.exceptions.RequestException:
        console.print("[red]✗ Cannot connect to API[/red]")
        console.print("\nPlease start the Neutryx API:")
        console.print("  [cyan]uvicorn neutryx.api.rest:create_app --factory --reload[/cyan]")
        return

    # Run XVA computation
    console.print("\n[yellow]Running XVA calculations...[/yellow]")
    console.print("[dim]This may take a moment...[/dim]\n")

    # Execute compute_xva script
    import subprocess

    script_path = Path(__file__).parent / "compute_xva.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)

    if result.returncode == 0:
        console.print("\n[bold green]✓ XVA calculations complete![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}/xva_results.json[/cyan]")
    else:
        console.print("\n[bold red]✗ XVA calculation failed[/bold red]")


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="reports",
    help="Directory to save reports",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "csv", "excel", "json", "all"]),
    default="all",
    help="Report format",
)
def report(output_dir, format):
    """Generate comprehensive portfolio reports."""
    console.print(Panel.fit("Generating Portfolio Reports", style="bold blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating reports...", total=None)

        import subprocess

        script_path = Path(__file__).parent / "portfolio_report.py"
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

        progress.update(task, completed=True)

    if result.returncode == 0:
        console.print("\n[bold green]✓ Reports generated successfully![/bold green]")
        console.print(f"Output directory: [cyan]{output_dir}/[/cyan]")
    else:
        console.print("\n[bold red]✗ Report generation failed[/bold red]")
        console.print(result.stderr)


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="sample_outputs/charts",
    help="Directory to save charts",
)
def visualize(output_dir):
    """Generate portfolio visualizations and charts."""
    console.print(Panel.fit("Generating Visualizations", style="bold blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating charts...", total=None)

        import subprocess

        script_path = Path(__file__).parent / "visualization.py"
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

        progress.update(task, completed=True)

    if result.returncode == 0:
        console.print("\n[bold green]✓ Visualizations generated![/bold green]")
        console.print(f"Charts saved to: [cyan]{output_dir}/[/cyan]")
    else:
        console.print("\n[bold red]✗ Visualization failed[/bold red]")
        console.print(result.stderr)


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="reports",
    help="Directory to save stress test results",
)
@click.option(
    "--category",
    "-c",
    type=click.Choice(["rates", "fx", "equity", "volatility", "credit", "combined", "all"]),
    default="all",
    help="Stress test category",
)
def stress(output_dir, category):
    """Run stress tests on the portfolio."""
    console.print(Panel.fit("Running Stress Tests", style="bold blue"))

    console.print(f"\nCategory: [cyan]{category}[/cyan]")
    console.print("[dim]Executing stress scenarios...[/dim]\n")

    import subprocess

    script_path = Path(__file__).parent / "stress_testing.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)

    if result.returncode == 0:
        console.print("\n[bold green]✓ Stress tests complete![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}/[/cyan]")
    else:
        console.print("\n[bold red]✗ Stress testing failed[/bold red]")


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="reports",
    help="Directory to save sensitivity results",
)
def sensitivity(output_dir):
    """Compute Greeks and risk sensitivities."""
    console.print(Panel.fit("Computing Sensitivities & Greeks", style="bold blue"))

    console.print("\n[dim]Calculating option Greeks and risk sensitivities...[/dim]\n")

    import subprocess

    script_path = Path(__file__).parent / "sensitivity_analysis.py"
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)

    if result.returncode == 0:
        console.print("\n[bold green]✓ Sensitivity analysis complete![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}/[/cyan]")
    else:
        console.print("\n[bold red]✗ Sensitivity analysis failed[/bold red]")


@cli.command()
def demo():
    """Run complete demo workflow (all analyses)."""
    console.print(
        Panel.fit("Running Complete Demo Workflow", style="bold magenta", padding=(1, 2))
    )

    steps = [
        ("load", "Loading Portfolio"),
        ("report", "Generating Reports"),
        ("visualize", "Creating Visualizations"),
        ("stress", "Running Stress Tests"),
        ("sensitivity", "Computing Sensitivities"),
    ]

    console.print("\n[bold]Demo Workflow Steps:[/bold]")
    for i, (cmd, desc) in enumerate(steps, 1):
        console.print(f"  {i}. {desc}")

    console.print("\n[yellow]Starting demo...[/yellow]\n")

    for i, (cmd, desc) in enumerate(steps, 1):
        console.rule(f"[bold blue]Step {i}/{len(steps)}: {desc}[/bold blue]")
        console.print()

        import subprocess

        if cmd == "load":
            script_path = Path(__file__).parent / "load_portfolio.py"
        elif cmd == "report":
            script_path = Path(__file__).parent / "portfolio_report.py"
        elif cmd == "visualize":
            script_path = Path(__file__).parent / "visualization.py"
        elif cmd == "stress":
            script_path = Path(__file__).parent / "stress_testing.py"
        elif cmd == "sensitivity":
            script_path = Path(__file__).parent / "sensitivity_analysis.py"

        result = subprocess.run([sys.executable, str(script_path)], capture_output=False)

        if result.returncode == 0:
            console.print(f"\n[green]✓ {desc} complete[/green]\n")
        else:
            console.print(f"\n[red]✗ {desc} failed[/red]\n")
            console.print("[yellow]Continuing with remaining steps...[/yellow]\n")

    console.rule("[bold green]Demo Complete[/bold green]")
    console.print("\n[bold]All analyses finished![/bold]")
    console.print("\nGenerated outputs can be found in:")
    console.print("  • [cyan]reports/[/cyan] - Reports and analysis results")
    console.print("  • [cyan]sample_outputs/charts/[/cyan] - Visualizations")
    console.print("  • [cyan]snapshots/[/cyan] - Portfolio snapshots")


@cli.command()
def info():
    """Display information about the fictional portfolio."""
    console.print(Panel.fit("Fictional Bank Portfolio Information", style="bold cyan"))

    info_table = Table(show_header=False, box=None)
    info_table.add_column("Key", style="bold cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Legal Entity", "Global Investment Bank Ltd")
    info_table.add_row("Business Unit", "Global Trading")
    info_table.add_row("Desks", "Interest Rates, Foreign Exchange, Equity Derivatives")
    info_table.add_row("Total Books", "7")
    info_table.add_row("Traders", "6")
    info_table.add_row("Counterparties", "6 (diverse credit ratings)")
    info_table.add_row("CSA Coverage", "4 counterparties with CSA agreements")
    info_table.add_row("Total Trades", "13 trades")
    info_table.add_row(
        "Products",
        "IRS, Swaptions, FX Options, Equity Options, Variance Swaps",
    )
    info_table.add_row("Total Notional", "~USD 152M")
    info_table.add_row("Base Currency", "USD")

    console.print()
    console.print(info_table)
    console.print()

    console.print("[bold]Available Commands:[/bold]")
    console.print("  • [cyan]cli.py load[/cyan]        - Load and display portfolio")
    console.print("  • [cyan]cli.py xva[/cyan]         - Compute XVA metrics")
    console.print("  • [cyan]cli.py report[/cyan]      - Generate reports")
    console.print("  • [cyan]cli.py visualize[/cyan]   - Create visualizations")
    console.print("  • [cyan]cli.py stress[/cyan]      - Run stress tests")
    console.print("  • [cyan]cli.py sensitivity[/cyan] - Compute Greeks")
    console.print("  • [cyan]cli.py demo[/cyan]        - Run complete workflow")
    console.print()


@cli.command()
@click.option("--check-api", is_flag=True, help="Check API connectivity")
@click.option("--check-deps", is_flag=True, help="Check Python dependencies")
def status(check_api, check_deps):
    """Check system status and dependencies."""
    console.print(Panel.fit("System Status Check", style="bold yellow"))

    # Check directory structure
    console.print("\n[bold]Directory Structure:[/bold]")
    dirs = ["reports", "snapshots", "data", "sample_outputs", "tests", "templates"]
    for dir_name in dirs:
        dir_path = Path(__file__).parent / dir_name
        if dir_path.exists():
            console.print(f"  [green]✓[/green] {dir_name}/")
        else:
            console.print(f"  [red]✗[/red] {dir_name}/ [dim](missing)[/dim]")

    # Check scripts
    console.print("\n[bold]Available Scripts:[/bold]")
    scripts = [
        "load_portfolio.py",
        "compute_xva.py",
        "portfolio_report.py",
        "visualization.py",
        "stress_testing.py",
        "sensitivity_analysis.py",
    ]
    for script in scripts:
        script_path = Path(__file__).parent / script
        if script_path.exists():
            console.print(f"  [green]✓[/green] {script}")
        else:
            console.print(f"  [red]✗[/red] {script} [dim](missing)[/dim]")

    # Check API
    if check_api:
        console.print("\n[bold]API Connectivity:[/bold]")
        import requests

        try:
            response = requests.get("http://localhost:8000/docs", timeout=2)
            if response.status_code == 200:
                console.print("  [green]✓[/green] Neutryx API is running")
            else:
                console.print(f"  [yellow]![/yellow] API returned status {response.status_code}")
        except requests.exceptions.RequestException:
            console.print("  [red]✗[/red] Cannot connect to API")
            console.print("    Start with: [cyan]uvicorn neutryx.api.rest:create_app --factory[/cyan]")

    # Check dependencies
    if check_deps:
        console.print("\n[bold]Python Dependencies:[/bold]")
        deps = [
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "plotly",
            "openpyxl",
            "click",
            "rich",
            "requests",
        ]
        for dep in deps:
            try:
                __import__(dep)
                console.print(f"  [green]✓[/green] {dep}")
            except ImportError:
                console.print(f"  [red]✗[/red] {dep} [dim](not installed)[/dim]")

    console.print()


if __name__ == "__main__":
    cli()
