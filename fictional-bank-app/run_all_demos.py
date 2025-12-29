#!/usr/bin/env python3
"""Master demo script - Execute all fictional bank demonstrations.

This script runs a complete workflow showcasing all Neutryx capabilities:
1. Portfolio loading and summary
2. XVA calculations (if API available)
3. Comprehensive reporting
4. Visualization generation
5. Stress testing
6. Sensitivity analysis

Perfect for demonstrations and testing the complete system.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("=" * 80)
    print(text.center(80))
    print("=" * 80)
    print(f"{Colors.ENDC}\n")


def print_step(step_num, total_steps, description):
    """Print a step header."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}")
    print(f"[Step {step_num}/{total_steps}] {description}")
    print("-" * 80)
    print(f"{Colors.ENDC}")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def run_script(script_name, description):
    """Run a Python script and return success status.

    Args:
        script_name: Name of the script file
        description: Human-readable description

    Returns:
        True if script executed successfully, False otherwise
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print_error(f"Script not found: {script_name}")
        return False

    print(f"  Executing: {script_name}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print()
            print_success(f"{description} completed in {elapsed_time:.2f}s")
            return True
        else:
            print()
            print_error(f"{description} failed (exit code: {result.returncode})")
            return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        print()
        print_error(f"{description} failed: {str(e)}")
        return False


def check_api_availability():
    """Check if Neutryx API is available.

    Returns:
        True if API is running, False otherwise
    """
    try:
        import requests
        response = requests.get("http://localhost:8000/docs", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    """Run the complete demo workflow."""
    start_time = datetime.now()

    print_header("FICTIONAL BANK PORTFOLIO - COMPLETE DEMO")

    print(f"{Colors.OKCYAN}Demo Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print()
    print("This demo will execute the following steps:")
    print("  1. Load Portfolio")
    print("  2. Generate Comprehensive Reports")
    print("  3. Create Visualizations")
    print("  4. Run Stress Tests")
    print("  5. Compute Sensitivities & Greeks")
    print("  6. (Optional) Compute XVA Metrics")
    print()

    # Check API availability
    api_available = check_api_availability()
    if api_available:
        print_success("Neutryx API is running - XVA calculations will be included")
    else:
        print_warning("Neutryx API not detected - XVA calculations will be skipped")
        print("  To enable XVA: uvicorn neutryx.api.rest:create_app --factory --reload")

    input(f"\n{Colors.BOLD}Press Enter to begin the demo...{Colors.ENDC}")

    # Track results
    results = {}
    total_steps = 6 if api_available else 5

    # Step 1: Load Portfolio
    print_step(1, total_steps, "Load Portfolio")
    results["load"] = run_script("load_portfolio.py", "Portfolio loading")

    # Step 2: Generate Reports
    print_step(2, total_steps, "Generate Comprehensive Reports")
    results["report"] = run_script("portfolio_report.py", "Report generation")

    # Step 3: Create Visualizations
    print_step(3, total_steps, "Create Visualizations")
    results["visualize"] = run_script("visualization.py", "Visualization generation")

    # Step 4: Stress Testing
    print_step(4, total_steps, "Run Stress Tests")
    results["stress"] = run_script("stress_testing.py", "Stress testing")

    # Step 5: Sensitivity Analysis
    print_step(5, total_steps, "Compute Sensitivities & Greeks")
    results["sensitivity"] = run_script("sensitivity_analysis.py", "Sensitivity analysis")

    # Step 6: XVA Calculation (if API available)
    if api_available:
        print_step(6, total_steps, "Compute XVA Metrics")
        results["xva"] = run_script("compute_xva.py", "XVA calculation")

    # Summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print_header("DEMO SUMMARY")

    print(f"Start Time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print()

    # Results table
    print(f"{Colors.BOLD}Execution Results:{Colors.ENDC}")
    print("-" * 80)

    success_count = 0
    for step, success in results.items():
        status = f"{Colors.OKGREEN}SUCCESS{Colors.ENDC}" if success else f"{Colors.FAIL}FAILED{Colors.ENDC}"
        print(f"  {step.ljust(20)} : {status}")
        if success:
            success_count += 1

    print("-" * 80)
    print(f"Success Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print()

    # Output locations
    print(f"{Colors.BOLD}Generated Outputs:{Colors.ENDC}")
    print("-" * 80)

    output_dirs = [
        ("reports/", "Analysis reports and results"),
        ("sample_outputs/charts/", "Visualization charts"),
        ("snapshots/", "Portfolio snapshots"),
    ]

    for dir_path, description in output_dirs:
        full_path = Path(__file__).parent / dir_path
        if full_path.exists():
            file_count = len(list(full_path.glob("*")))
            print(f"  {dir_path.ljust(30)} : {file_count} files - {description}")
        else:
            print(f"  {dir_path.ljust(30)} : {Colors.WARNING}Not found{Colors.ENDC}")

    print()

    # Recommendations
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("-" * 80)
    print("  1. Review HTML reports in reports/ directory")
    print("  2. View visualization charts in sample_outputs/charts/")
    print("  3. Examine Excel reports for detailed analysis")
    print("  4. Use the CLI for interactive analysis:")
    print(f"     {Colors.OKCYAN}./cli.py --help{Colors.ENDC}")
    print()

    if not api_available:
        print(f"{Colors.WARNING}Note:{Colors.ENDC} XVA calculations were skipped.")
        print("  Start the API and run compute_xva.py for XVA metrics:")
        print(f"    {Colors.OKCYAN}uvicorn neutryx.api.rest:create_app --factory --reload{Colors.ENDC}")
        print(f"    {Colors.OKCYAN}./compute_xva.py{Colors.ENDC}")
        print()

    print_header("DEMO COMPLETE")

    if success_count == len(results):
        print(f"{Colors.OKGREEN}{Colors.BOLD}All steps completed successfully!{Colors.ENDC}")
    elif success_count > 0:
        print(f"{Colors.WARNING}{Colors.BOLD}Demo completed with some warnings.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}Demo encountered errors.{Colors.ENDC}")

    print()
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
