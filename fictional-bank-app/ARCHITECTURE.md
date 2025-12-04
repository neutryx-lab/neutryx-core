# System Architecture

Technical architecture documentation for the Fictional Bank portfolio example.

## Table of Contents

1. [Overview](#overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Module Structure](#module-structure)
5. [Integration Points](#integration-points)
6. [Extension Points](#extension-points)
7. [Design Patterns](#design-patterns)
8. [Performance Considerations](#performance-considerations)

## Overview

The Fictional Bank example is designed as a modular, production-quality demonstration of Neutryx's capabilities. It follows a layered architecture with clear separation of concerns.

### Architecture Principles

1. **Modularity**: Each script is self-contained and can run independently
2. **Reusability**: Common functionality in shared utilities
3. **Extensibility**: Easy to add new analytics or scenarios
4. **Testability**: Clear interfaces for unit testing
5. **Documentation**: Comprehensive inline and external docs

### Technology Stack

```
┌─────────────────────────────────────────┐
│          User Interface Layer           │
│  (CLI, Scripts, Jupyter Notebooks)     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Analytics Layer                 │
│  (Reporting, Visualization, Analysis)   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Business Logic Layer            │
│  (Portfolio Management, Calculations)   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Neutryx API Layer              │
│     (XVA, Pricing, Risk Metrics)        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           Data Layer                    │
│  (Config, Snapshots, Market Data)       │
└─────────────────────────────────────────┘
```

## Component Architecture

### Core Components

#### 1. Portfolio Management

**File**: `fictional_portfolio.py` (in fixtures)

**Responsibilities**:
- Create comprehensive test portfolio
- Define book hierarchy
- Manage counterparties and netting sets
- Generate portfolio summaries

**Key Classes**:
- `Portfolio`: Main portfolio container
- `BookHierarchy`: Organizational structure
- `Trade`: Individual trade representation
- `Counterparty`: Counterparty entities
- `NettingSet`: Netting set definitions

#### 2. CLI Interface

**File**: `cli.py`

**Responsibilities**:
- Provide user-friendly command interface
- Orchestrate workflow execution
- Display formatted output
- Handle user input

**Technology**:
- Click: Command-line argument parsing
- Rich: Terminal formatting and progress bars

**Command Structure**:
```
cli.py
├── info        # Display portfolio information
├── load        # Load and display portfolio
├── report      # Generate reports
├── visualize   # Create charts
├── stress      # Run stress tests
├── sensitivity # Compute Greeks
├── xva         # Calculate XVA
├── demo        # Complete workflow
└── status      # System health check
```

#### 3. Reporting Engine

**File**: `portfolio_report.py`

**Responsibilities**:
- Generate multi-format reports
- Aggregate portfolio data
- Create formatted outputs

**Output Formats**:
- HTML: Styled reports with tables
- Excel: Multi-sheet workbooks
- CSV: Raw data exports
- JSON: Structured data

**Class Structure**:
```python
class PortfolioReporter:
    def __init__(output_dir)
    def generate_all_reports() -> Dict[str, Path]
    def generate_json_report() -> Path
    def generate_csv_reports() -> Dict[str, Path]
    def generate_excel_report() -> Path
    def generate_html_report() -> Path
```

#### 4. Visualization System

**File**: `visualization.py`

**Responsibilities**:
- Generate professional charts
- Create multiple chart types
- Save high-resolution outputs

**Technologies**:
- Matplotlib: Core plotting
- Seaborn: Statistical visualizations
- Plotly: Interactive charts (future)

**Chart Types**:
- Bar charts (exposure, comparisons)
- Pie charts (breakdowns)
- Histograms (distributions)
- Box plots (statistics)
- Stacked charts (composition)
- Waterfall charts (XVA components)

#### 5. Stress Testing Framework

**File**: `stress_testing.py`

**Responsibilities**:
- Define stress scenarios
- Execute stress tests
- Aggregate results
- Generate stress reports

**Scenario Categories**:
- Interest Rates
- FX
- Equity
- Volatility
- Credit
- Combined (multi-factor)

**Class Structure**:
```python
@dataclass
class StressScenario:
    name: str
    description: str
    category: str
    shocks: Dict[str, Any]
    severity: str

class StressTester:
    def __init__(output_dir)
    def _define_scenarios() -> List[StressScenario]
    def run_all_stress_tests() -> Dict[str, Dict]
    def generate_stress_report() -> Path
```

#### 6. Sensitivity Analysis

**File**: `sensitivity_analysis.py`

**Responsibilities**:
- Compute option Greeks
- Calculate risk sensitivities
- Bucketed risk ladders
- Generate sensitivity reports

**Analytics**:
- Greeks: Delta, Gamma, Vega, Theta, Rho
- IR: PV01, DV01, CS01
- FX: Delta exposures
- EQ: Delta, Gamma
- Bucketed: Tenor-based sensitivities

#### 7. XVA Calculator

**File**: `compute_xva.py`

**Responsibilities**:
- Interface with Neutryx API
- Calculate portfolio XVA
- Compute netting set XVA
- Analyze CSA impact

**XVA Components**:
- CVA: Credit Valuation Adjustment
- DVA: Debit Valuation Adjustment
- FVA: Funding Valuation Adjustment
- MVA: Margin Valuation Adjustment

#### 8. Demo Orchestrator

**File**: `run_all_demos.py`

**Responsibilities**:
- Execute complete workflow
- Track execution status
- Display progress
- Summarize results

## Data Flow

### Portfolio Loading Flow

```
┌──────────────────┐
│ Start            │
└────────┬─────────┘
         ↓
┌──────────────────────────────┐
│ create_fictional_portfolio() │
│ - Create trades              │
│ - Set up counterparties      │
│ - Define netting sets        │
└────────┬─────────────────────┘
         ↓
┌──────────────────────────┐
│ get_portfolio_summary()  │
│ - Aggregate statistics   │
│ - Calculate MTM          │
│ - Count trades           │
└────────┬─────────────────┘
         ↓
┌──────────────────┐
│ Display Summary  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Save Snapshot    │
└──────────────────┘
```

### XVA Calculation Flow

```
┌───────────────────┐
│ Load Portfolio    │
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Check API Health  │
└────────┬──────────┘
         ↓
┌──────────────────────────┐
│ Register Portfolio       │
│ POST /portfolio/register │
└────────┬─────────────────┘
         ↓
┌───────────────────────────┐
│ Get Netting Sets          │
│ GET /portfolio/.../       │
│     netting-sets          │
└────────┬──────────────────┘
         ↓
┌───────────────────────────┐
│ Calculate Portfolio XVA   │
│ POST /portfolio/xva       │
└────────┬──────────────────┘
         ↓
┌───────────────────────────┐
│ For each netting set:     │
│  - Calculate XVA          │
│  - Aggregate results      │
└────────┬──────────────────┘
         ↓
┌───────────────────────────┐
│ Analyze CSA Impact        │
│ Generate Insights         │
└────────┬──────────────────┘
         ↓
┌───────────────────────────┐
│ Save Results to JSON      │
└───────────────────────────┘
```

### Report Generation Flow

```
┌────────────────────┐
│ Load Portfolio     │
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Load XVA Results   │
│ (if available)     │
└──────┬─────────────┘
       ↓
┌──────────────────────────┐
│ Generate Reports:        │
│ ┌──────────────────────┐ │
│ │ 1. JSON Report       │ │
│ │ 2. CSV Reports       │ │
│ │ 3. Excel Workbook    │ │
│ │ 4. HTML Report       │ │
│ └──────────────────────┘ │
└──────┬───────────────────┘
       ↓
┌────────────────────┐
│ Save All Files     │
└────────────────────┘
```

## Module Structure

### File Organization

```
fictional_bank/
├── Core Scripts (executable)
│   ├── load_portfolio.py          # Entry point for loading
│   ├── compute_xva.py             # XVA workflow
│   ├── portfolio_report.py        # Reporting engine
│   ├── visualization.py           # Chart generation
│   ├── stress_testing.py          # Stress framework
│   └── sensitivity_analysis.py    # Greeks calculator
│
├── CLI & Automation
│   ├── cli.py                     # Main CLI interface
│   └── run_all_demos.py           # Workflow orchestrator
│
├── Configuration
│   ├── config.yaml                # Market data & parameters
│   └── requirements.txt           # Python dependencies
│
├── Documentation
│   ├── README.md                  # Overview
│   ├── USER_GUIDE.md              # Usage guide
│   ├── API_EXAMPLES.md            # API integration
│   ├── ARCHITECTURE.md            # This file
│   ├── TROUBLESHOOTING.md         # Common issues
│   └── CONTRIBUTING.md            # Contribution guide
│
└── Outputs (generated)
    ├── reports/                   # Analysis reports
    ├── snapshots/                 # Portfolio snapshots
    ├── sample_outputs/charts/     # Visualizations
    └── data/scenarios/            # Scenario definitions
```

### Dependencies

**Core Dependencies**:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `pydantic`: Data validation (via Neutryx)

**Visualization**:
- `matplotlib`: Plotting
- `seaborn`: Statistical graphics
- `plotly`: Interactive charts

**Reporting**:
- `jinja2`: HTML templating
- `openpyxl`: Excel file creation
- `xlsxwriter`: Excel formatting

**CLI**:
- `click`: Command-line interface
- `rich`: Terminal formatting

**API**:
- `requests`: HTTP client
- `uvicorn`: ASGI server (for API)

## Integration Points

### 1. Neutryx API Integration

**Protocol**: REST API over HTTP
**Format**: JSON
**Authentication**: None (development)

**Key Endpoints**:
- `POST /portfolio/register` - Register portfolio
- `GET /portfolio/{id}/summary` - Get summary
- `GET /portfolio/{id}/netting-sets` - List netting sets
- `POST /portfolio/xva` - Calculate XVA

### 2. File System Integration

**Read**:
- `config.yaml` - Market data configuration
- `fictional_portfolio.py` - Portfolio fixture

**Write**:
- `reports/` - Analysis results
- `snapshots/` - Portfolio data
- `sample_outputs/charts/` - Visualizations

### 3. External Systems

**Potential Integrations**:
- Market data feeds (Bloomberg, Reuters)
- Risk management systems
- Reporting platforms
- Data warehouses
- Monitoring systems

## Extension Points

### Adding New Analytics

1. **Create new script** following existing pattern:
   ```python
   #!/usr/bin/env python3
   """New analysis description."""

   class NewAnalyzer:
       def __init__(self, output_dir):
           self.output_dir = output_dir

       def analyze(self, portfolio):
           # Implementation
           pass
   ```

2. **Add CLI command** in `cli.py`:
   ```python
   @cli.command()
   def new_analysis():
       """Description."""
       # Call new script
       pass
   ```

3. **Integrate in demo** workflow (`run_all_demos.py`)

### Adding New Scenarios

Edit `stress_testing.py`:
```python
scenarios.append(
    StressScenario(
        name="New_Scenario",
        description="...",
        category="combined",
        shocks={...},
        severity="severe",
    )
)
```

### Adding New Visualizations

Edit `visualization.py`:
```python
def plot_new_chart(self, data):
    fig, ax = plt.subplots(figsize=(12, 8))
    # Chart implementation
    output_file = self.output_dir / "new_chart.png"
    plt.savefig(output_file, dpi=300)
    return output_file
```

## Design Patterns

### 1. Strategy Pattern

Used for different report formats:
```python
class PortfolioReporter:
    def generate_json_report()
    def generate_csv_reports()
    def generate_excel_report()
    def generate_html_report()
```

### 2. Factory Pattern

Portfolio creation:
```python
def create_fictional_portfolio() -> Tuple[Portfolio, BookHierarchy]:
    # Complex object construction
    pass
```

### 3. Builder Pattern

Report building:
```python
html = self._build_html_report(summary, xva_results)
```

### 4. Template Method

Script execution pattern:
```python
def main():
    print_header()
    load_data()
    process()
    save_results()
    print_summary()
```

## Performance Considerations

### Optimization Strategies

1. **Caching**:
   - Portfolio summaries
   - API responses
   - Expensive calculations

2. **Parallel Processing**:
   - Netting set XVA calculations
   - Chart generation
   - Report creation

3. **Lazy Loading**:
   - Market data
   - XVA results
   - Large datasets

4. **Memory Management**:
   - Stream large files
   - Use generators
   - Clear unused data

### Scalability

**Current Design**:
- Single-threaded execution
- In-memory data processing
- Serial API calls

**Production Enhancements**:
- Async/await for API calls
- Multiprocessing for analytics
- Database for large portfolios
- Message queues for distributed processing

### Bottlenecks

1. **API Calls**: Sequential netting set XVA calculations
2. **Chart Generation**: matplotlib rendering
3. **Excel Writing**: Large workbook creation
4. **Portfolio Loading**: Complex object construction

**Mitigation**:
- Implement async API calls
- Use batch API endpoints
- Generate charts on demand
- Cache intermediate results

## Security Considerations

### Current Implementation

- No authentication (development only)
- Local file system access
- HTTP (not HTTPS)
- No input sanitization

### Production Requirements

1. **Authentication**: JWT/OAuth2
2. **Authorization**: Role-based access
3. **Encryption**: HTTPS/TLS
4. **Input Validation**: Sanitize all inputs
5. **Audit Logging**: Track all operations
6. **Secrets Management**: Vault for credentials

## Future Enhancements

### Planned Features

1. **Real-time Analytics**: Streaming calculations
2. **Machine Learning**: Predictive models
3. **Advanced Visualizations**: Interactive dashboards
4. **Multi-portfolio**: Cross-portfolio analytics
5. **Historical Analysis**: Time-series analytics
6. **Backtesting**: Strategy validation

### Technical Debt

- Add comprehensive unit tests
- Implement integration tests
- Add type hints throughout
- Improve error handling
- Add logging framework
- Create Docker container
- CI/CD pipeline

---

For implementation details, see the source code. For usage information, see [USER_GUIDE.md](USER_GUIDE.md).
