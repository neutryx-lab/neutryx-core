# Code Duplication Analysis - Complete Index

## Overview

This analysis examines the `/workspaces/neutryx-core/src/neutryx/models/` directory for code duplication patterns that can be consolidated into reusable utility functions.

**Analysis Date:** November 3, 2025
**Scope:** 12 Python model files
**Total Duplication:** 50-60 lines of identical/near-identical code
**Consolidation Potential:** 20-25 lines in utilities

---

## Document Guide

### 1. CODE_DUPLICATION_SUMMARY.txt
**Best for:** Quick overview and impact assessment

Contains:
- Top 5 duplication hotspots with impact ratings
- Medium priority duplications
- Concrete code block examples
- Impact assessment table with LOC counts
- Recommended extraction order

**Use this to:** Understand duplication severity and prioritization

---

### 2. CODE_DUPLICATION_ANALYSIS.md
**Best for:** Comprehensive detailed analysis with context

Contains:
- 10 detailed duplication patterns
- Complete code snippets for each instance
- Specific file locations and line numbers
- Recommended utility functions
- Severity scoring by pattern type
- 3-phase refactoring approach

**Sections:**
1. Parameter Validation Patterns
2. Simulation Path Structure
3. Discount Factor Calculations
4. Option Payoff Calculations
5. Array Initialization Patterns
6. Bond Pricing Patterns
7. Yield Curve Computation
8. Random Number Generation
9. Numerical Stability Patterns
10. Implied Vol / Calibration Patterns

**Use this to:** Understand pattern context and propose utility designs

---

### 3. CODE_DUPLICATION_DETAILED_REFERENCE.md
**Best for:** Exact line-by-line implementation details

Contains:
- Instance-by-instance duplication catalog
- Exact file paths and line numbers
- Status classification (IDENTICAL, SIMILAR, PARTIAL, REPETITIVE)
- Summary table with all instances
- Critical issues (duplicate function definitions)

**Categories:**
1. Option Payoff Computation (4 instances)
2. Discount Factor & Present Value (4 instances)
3. Simulate Paths Generic Pattern (3 instances)
4. Log-normal Path Construction (3 instances)
5. Initial Value Path Prepending (3 instances)
6. Parameter Validation Patterns (3 instances)
7. Yield Curve Computation (2 instances)
8. Bond Price Calculations (3 models)
9. Safe Square Root with Clipping (4 instances)
10. Duplicate Function Definitions (1 critical)

**Use this to:** Implement refactoring with exact references

---

## Quick Navigation Guide

### Looking for...

**Quick overview of all issues?**
→ Start with `CODE_DUPLICATION_SUMMARY.txt`

**Specific line numbers for refactoring?**
→ Go to `CODE_DUPLICATION_DETAILED_REFERENCE.md`

**Understanding why this matters?**
→ Read `CODE_DUPLICATION_ANALYSIS.md` sections 1-5

**Implementation examples?**
→ Check concrete code blocks in `CODE_DUPLICATION_ANALYSIS.md`

**Prioritization guide?**
→ See Impact Assessment Table in `CODE_DUPLICATION_SUMMARY.txt`

---

## Key Findings Summary

### Highest Priority Issues (Quick Win)

1. **Option Payoff Calculation**
   - 8+ occurrences across 4+ files
   - File: kou.py:179-184, variance_gamma.py:154-159, etc.
   - Extraction: `compute_payoff(spot, strike, kind="call")`
   - Effort: LOW | Impact: HIGH

2. **Discount & Present Value**
   - 4+ occurrences across 4+ files
   - File: variance_gamma.py:161-162, kou.py:186-187, etc.
   - Extraction: `discount_payoff(payoffs, rate, maturity)`
   - Effort: LOW | Impact: HIGH

3. **Simulate Paths via vmap**
   - 3 identical implementations
   - File: vasicek.py:200-235, cir.py:249-289, hull_white.py:216-251
   - Extraction: `generic_vmap_simulate_paths(fn, key, n_paths)`
   - Effort: LOW | Impact: MEDIUM

4. **Log-Path Construction**
   - 3 nearly identical blocks
   - File: variance_gamma.py:94-100, kou.py:113-120
   - Extraction: `build_log_paths(S0, increments, dtype)`
   - Effort: LOW | Impact: MEDIUM

5. **Initial Value Prepending**
   - 3 identical lines
   - File: vasicek.py:197, cir.py:246, hull_white.py:213
   - Extraction: `prepend_initial_value(initial, path)`
   - Effort: MINIMAL | Impact: CLARITY

### Critical Issues

- **Duplicate Function Definition:** rough_vol.py has `price_european_call_mc()` defined twice (lines 189-205 and 208-242)
  - Second definition overwrites first
  - Requires immediate resolution

---

## Files Referenced

### Model Files Analyzed
- `/workspaces/neutryx-core/src/neutryx/models/bs.py`
- `/workspaces/neutryx-core/src/neutryx/models/cir.py`
- `/workspaces/neutryx-core/src/neutryx/models/heston_cf.py`
- `/workspaces/neutryx-core/src/neutryx/models/hull_white.py`
- `/workspaces/neutryx-core/src/neutryx/models/jump_diffusion.py`
- `/workspaces/neutryx-core/src/neutryx/models/kou.py`
- `/workspaces/neutryx-core/src/neutryx/models/rough_vol.py`
- `/workspaces/neutryx-core/src/neutryx/models/sde.py`
- `/workspaces/neutryx-core/src/neutryx/models/variance_gamma.py`
- `/workspaces/neutryx-core/src/neutryx/models/vasicek.py`
- `/workspaces/neutryx-core/src/neutryx/models/workflows.py`
- `/workspaces/neutryx-core/src/neutryx/models/__init__.py`

### Analysis Documents Generated
- `/workspaces/neutryx-core/CODE_DUPLICATION_SUMMARY.txt` (this index)
- `/workspaces/neutryx-core/CODE_DUPLICATION_ANALYSIS.md`
- `/workspaces/neutryx-core/CODE_DUPLICATION_DETAILED_REFERENCE.md`

---

## Recommended Utility Module Structure

```
neutryx/core/math/
├── payoffs.py          # compute_payoff(), discount_payoff()
├── numerics.py         # safe_sqrt(), safe_parameter()
└── bonds.py            # bond pricing utilities

neutryx/core/engine/
├── simulation.py       # build_log_paths(), prepend_initial_value()
│                      # generic_vmap_simulate_paths()
└── yield_curves.py     # generic_yield_curve()
```

---

## Implementation Checklist

- [ ] Extract `compute_payoff()` utility
- [ ] Extract `discount_payoff()` utility
- [ ] Extract `build_log_paths()` utility
- [ ] Extract `prepend_initial_value()` utility
- [ ] Extract `safe_sqrt()` utility
- [ ] Extract `generic_vmap_simulate_paths()` utility
- [ ] Extract `validate_positive_param()` utility
- [ ] Extract `generic_yield_curve()` utility
- [ ] Resolve duplicate `price_european_call_mc()` in rough_vol.py
- [ ] Update imports in all model files
- [ ] Add comprehensive docstrings with examples
- [ ] Create unit tests for new utilities
- [ ] Verify backward compatibility
- [ ] Update __all__ exports

---

## Success Criteria

- All duplication patterns identified with exact references
- Extraction utilities designed with clear signatures
- Consolidation potential demonstrated with LOC savings
- Implementation guide provided for each utility
- No breaking changes to public API
- Improved maintainability and consistency

---

## Contact / Questions

For questions about specific instances, refer to:
1. `CODE_DUPLICATION_DETAILED_REFERENCE.md` for exact locations
2. `CODE_DUPLICATION_ANALYSIS.md` for pattern context
3. Source files for implementation details

---

**Analysis Complete**
