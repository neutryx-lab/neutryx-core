# Troubleshooting Guide

Common issues and their solutions for the Fictional Bank portfolio example.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [API Connection Problems](#api-connection-problems)
3. [Script Execution Errors](#script-execution-errors)
4. [Output Generation Issues](#output-generation-issues)
5. [Performance Problems](#performance-problems)
6. [Data Validation Errors](#data-validation-errors)

## Installation Issues

### Problem: pip install fails with dependency conflicts

**Symptoms:**
```
ERROR: Cannot install -r requirements.txt (from line X) because these package versions have conflicting dependencies
```

**Solutions:**

1. **Use a fresh virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Install dependencies one by one:**
   ```bash
   pip install pandas numpy matplotlib seaborn
   pip install plotly openpyxl xlsxwriter
   pip install click rich requests pyyaml
   ```

3. **Use conda instead:**
   ```bash
   conda create -n neutryx python=3.10
   conda activate neutryx
   pip install -r requirements.txt
   ```

### Problem: ImportError for neutryx modules

**Symptoms:**
```python
ModuleNotFoundError: No module named 'neutryx'
```

**Solutions:**

1. **Verify PYTHONPATH:**
   ```bash
   export PYTHONPATH=/workspaces/neutryx-core/src:$PYTHONPATH
   ```

2. **Check project structure:**
   ```bash
   ls /workspaces/neutryx-core/src/neutryx
   ```

3. **Run from correct directory:**
   ```bash
   cd examples/applications/fictional_bank
   ./script.py
   ```

### Problem: Permission denied when running scripts

**Symptoms:**
```
bash: ./cli.py: Permission denied
```

**Solution:**
```bash
chmod +x *.py
# Or run with python directly
python cli.py
```

## API Connection Problems

### Problem: Cannot connect to Neutryx API

**Symptoms:**
```
✗ Cannot connect to API
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solutions:**

1. **Check if API is running:**
   ```bash
   curl http://localhost:8000/docs
   ```

2. **Start the API:**
   ```bash
   uvicorn neutryx.api.rest:create_app --factory --reload
   ```

3. **Check port availability:**
   ```bash
   lsof -i :8000  # On Linux/Mac
   netstat -ano | findstr :8000  # On Windows
   ```

4. **Try different port:**
   ```bash
   uvicorn neutryx.api.rest:create_app --factory --port 8001
   # Then update API_BASE_URL in scripts
   ```

### Problem: API returns 404 errors

**Symptoms:**
```
HTTP Error 404: Not Found
```

**Solutions:**

1. **Verify API is running correctly:**
   ```bash
   curl http://localhost:8000/docs  # Should return HTML
   ```

2. **Check endpoint URL:**
   ```python
   # Correct
   response = requests.get("http://localhost:8000/portfolio/summary")

   # Incorrect
   response = requests.get("http://localhost:8000/api/portfolio/summary")
   ```

3. **Review API logs for errors**

### Problem: API timeout during XVA calculation

**Symptoms:**
```
requests.exceptions.Timeout: HTTPConnectionPool(host='localhost', port=8000): Read timed out
```

**Solutions:**

1. **Increase timeout:**
   ```python
   response = requests.post(
       "http://localhost:8000/portfolio/xva",
       json=xva_request,
       timeout=300  # 5 minutes
   )
   ```

2. **Reduce portfolio size for testing:**
   - Start with fewer trades
   - Test on single netting set

3. **Check API server resources:**
   - CPU usage
   - Memory availability

## Script Execution Errors

### Problem: FileNotFoundError for portfolio fixture

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../fictional_portfolio.py'
```

**Solutions:**

1. **Verify fixture file exists:**
   ```bash
   ls src/neutryx/tests/fixtures/fictional_portfolio.py
   ```

2. **Check import path:**
   ```python
   # In your script
   import sys
   from pathlib import Path
   project_root = Path(__file__).parent.parent.parent
   sys.path.insert(0, str(project_root / "src"))
   ```

### Problem: YAML configuration not loading

**Symptoms:**
```
FileNotFoundError: config.yaml not found
yaml.scanner.ScannerError: while scanning...
```

**Solutions:**

1. **Verify file exists:**
   ```bash
   ls config.yaml
   pwd  # Ensure you're in the right directory
   ```

2. **Check YAML syntax:**
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

3. **Fix path issues:**
   ```python
   config_path = Path(__file__).parent / "config.yaml"
   with open(config_path) as f:
       config = yaml.safe_load(f)
   ```

### Problem: Output directory creation fails

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'reports'
```

**Solutions:**

1. **Check permissions:**
   ```bash
   ls -la
   chmod 755 .
   ```

2. **Create directories manually:**
   ```bash
   mkdir -p reports snapshots data sample_outputs/charts tests templates
   ```

3. **Use writable location:**
   ```python
   output_dir = Path.home() / "neutryx_reports"
   output_dir.mkdir(parents=True, exist_ok=True)
   ```

## Output Generation Issues

### Problem: HTML reports not displaying correctly

**Symptoms:**
- Broken formatting
- Missing styles
- Charts not showing

**Solutions:**

1. **Check HTML file integrity:**
   ```bash
   wc -l reports/portfolio_report_*.html  # Should have substantial line count
   ```

2. **Open in different browser:**
   - Some browsers have security restrictions
   - Try Chrome, Firefox, or Safari

3. **Verify file encoding:**
   ```python
   with open(html_file, "w", encoding="utf-8") as f:
       f.write(html_content)
   ```

### Problem: Excel files corrupted or won't open

**Symptoms:**
```
Excel cannot open the file because the format or extension is not valid
```

**Solutions:**

1. **Check openpyxl version:**
   ```bash
   pip install --upgrade openpyxl
   ```

2. **Try different Excel writer:**
   ```python
   # Instead of openpyxl
   df.to_excel(output_file, engine="xlsxwriter")
   ```

3. **Verify file is complete:**
   ```bash
   ls -lh reports/*.xlsx
   # File should have reasonable size, not 0 bytes
   ```

### Problem: Matplotlib charts not saving

**Symptoms:**
```
RuntimeError: Invalid DISPLAY variable
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Solutions:**

1. **Set backend explicitly:**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Non-interactive backend
   import matplotlib.pyplot as plt
   ```

2. **Install required libraries:**
   ```bash
   # On Linux
   sudo apt-get install python3-tk

   # On Mac
   brew install python-tk
   ```

3. **Check output directory is writable:**
   ```bash
   touch sample_outputs/charts/test.png
   rm sample_outputs/charts/test.png
   ```

### Problem: Charts have no data or are blank

**Symptoms:**
- Charts save but show no data
- Empty plots

**Solutions:**

1. **Verify data is present:**
   ```python
   print(f"Data points: {len(data)}")
   print(f"Data values: {data}")
   ```

2. **Check data types:**
   ```python
   # Ensure numeric data
   data = [float(x) for x in data]
   ```

3. **Add debug output:**
   ```python
   plt.savefig(output_file)
   print(f"Chart saved: {output_file}")
   print(f"File size: {output_file.stat().st_size} bytes")
   ```

## Performance Problems

### Problem: Scripts running very slowly

**Symptoms:**
- Takes minutes instead of seconds
- High CPU/memory usage

**Solutions:**

1. **Profile the code:**
   ```python
   import cProfile
   cProfile.run('main()', sort='cumtime')
   ```

2. **Reduce portfolio size for testing:**
   - Comment out some trades
   - Test with fewer scenarios

3. **Use efficient data structures:**
   ```python
   # Use pandas for large datasets
   import pandas as pd
   df = pd.DataFrame(data)
   df.to_csv(output_file)  # Faster than manual writing
   ```

4. **Enable caching:**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=128)
   def expensive_calculation(param):
       # Your code here
       pass
   ```

### Problem: Out of memory errors

**Symptoms:**
```
MemoryError
killed
```

**Solutions:**

1. **Process data in chunks:**
   ```python
   for chunk in pd.read_csv(large_file, chunksize=1000):
       process(chunk)
   ```

2. **Free memory explicitly:**
   ```python
   import gc
   del large_object
   gc.collect()
   ```

3. **Use generators instead of lists:**
   ```python
   # Instead of
   results = [expensive_calc(x) for x in large_list]

   # Use
   results = (expensive_calc(x) for x in large_list)
   ```

## Data Validation Errors

### Problem: Portfolio validation fails

**Symptoms:**
```
ValidationError: invalid portfolio structure
pydantic.error_wrappers.ValidationError
```

**Solutions:**

1. **Check portfolio structure:**
   ```python
   # Ensure all required fields are present
   portfolio_data = portfolio.model_dump(mode="json")
   print(json.dumps(portfolio_data, indent=2)[:500])
   ```

2. **Validate against schema:**
   ```python
   from pydantic import ValidationError
   try:
       portfolio = Portfolio(**data)
   except ValidationError as e:
       print(e.errors())
   ```

3. **Check date formats:**
   ```python
   # Use ISO format
   "valuation_date": "2024-01-15"  # YYYY-MM-DD
   ```

### Problem: Market data validation errors

**Symptoms:**
```
ValueError: Invalid rate curve
KeyError: 'USD'
```

**Solutions:**

1. **Verify config.yaml structure:**
   ```yaml
   market_data:
     rates:
       USD:
         curve_date: "2024-01-15"
         spot_rates:
           - tenor: "1Y"
             rate: 0.0485
   ```

2. **Check data types:**
   ```python
   # Rates should be floats, not strings
   rate: 0.0485  # Correct
   rate: "4.85%"  # Incorrect
   ```

3. **Validate YAML syntax:**
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
   ```

## Getting Help

If you're still experiencing issues:

1. **Check logs:**
   ```bash
   # API logs
   tail -f neutryx_api.log

   # Script output
   ./script.py 2>&1 | tee script.log
   ```

2. **Enable debug mode:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Simplify the problem:**
   - Test with minimal portfolio
   - Run one component at a time
   - Use CLI status check: `./cli.py status --check-deps --check-api`

4. **Review documentation:**
   - [README.md](README.md) - Overview
   - [USER_GUIDE.md](USER_GUIDE.md) - Detailed usage
   - [API_EXAMPLES.md](API_EXAMPLES.md) - API integration

5. **Check Python version:**
   ```bash
   python --version  # Should be 3.10+
   ```

6. **Verify package versions:**
   ```bash
   pip list | grep -E "(pandas|numpy|matplotlib|openpyxl)"
   ```

## Common Error Messages Quick Reference

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `ModuleNotFoundError` | Missing dependency or wrong PYTHONPATH | Install package or fix path |
| `ConnectionRefusedError` | API not running | Start Neutryx API |
| `FileNotFoundError` | Wrong directory or missing file | Check working directory |
| `PermissionError` | No write permissions | Check directory permissions |
| `ValidationError` | Invalid data structure | Verify data format |
| `TimeoutError` | API slow or not responding | Increase timeout |
| `MemoryError` | Insufficient RAM | Process data in chunks |
| `KeyError` | Missing configuration key | Check config.yaml |

---

Still need help? Review the inline comments in the source code or consult the main Neutryx documentation.
