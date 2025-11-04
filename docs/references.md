# Bibliography and References

## Foundational Textbooks

### General Quantitative Finance

1. **Hull, J. C.** (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
   - Standard textbook covering fundamental derivatives pricing theory
   - Black-Scholes-Merton framework, Greeks, basic numerical methods

2. **Shreve, S. E.** (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer.
   - Rigorous mathematical foundation for continuous-time finance
   - Risk-neutral pricing, martingale measures, change of numeraire

3. **Björk, T.** (2009). *Arbitrage Theory in Continuous Time* (3rd ed.). Oxford University Press.
   - Complete arbitrage theory framework
   - Term structure models, martingale methods

### Advanced Topics

4. **Gatheral, J.** (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
   - Volatility smile modeling, SABR model, local volatility
   - Market microstructure of volatility

5. **Glasserman, P.** (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
   - Comprehensive treatment of Monte Carlo methods
   - Variance reduction, quasi-Monte Carlo, American options

6. **Cont, R., & Tankov, P.** (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.
   - Lévy processes, jump-diffusion models
   - Variance gamma, Merton, Kou models

7. **Brigo, D., & Mercurio, F.** (2006). *Interest Rate Models - Theory and Practice* (2nd ed.). Springer.
   - Comprehensive interest rate modeling
   - Short rate models (Vasicek, CIR, Hull-White), HJM framework

8. **Gregory, J.** (2015). *The xVA Challenge: Counterparty Credit Risk, Funding, Collateral, and Capital* (3rd ed.). Wiley.
   - CVA, DVA, FVA, MVA, KVA calculations
   - Exposure modeling, wrong-way risk

## Seminal Research Papers

### Black-Scholes Framework

9. **Black, F., & Scholes, M.** (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.
   - Original Black-Scholes formula derivation
   - Foundation of modern option pricing theory

10. **Merton, R. C.** (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141-183.
    - Extended Black-Scholes framework
    - Rigorous continuous-time derivation

### Stochastic Volatility

11. **Heston, S. L.** (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies*, 6(2), 327-343.
    - Heston model with semi-analytical solution
    - Characteristic function approach
    - Implementation: `src/neutryx/models/heston.py`

12. **Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E.** (2002). "Managing Smile Risk." *Wilmott Magazine*, September, 84-108.
    - SABR (Stochastic Alpha Beta Rho) model
    - Asymptotic expansion for implied volatility
    - Implementation: `src/neutryx/models/sabr.py`

13. **Bayer, C., Friz, P., & Gatheral, J.** (2016). "Pricing under rough volatility." *Quantitative Finance*, 16(6), 887-904.
    - Rough Bergomi model with fractional Brownian motion
    - Empirical evidence for rough volatility (H ≈ 0.1)
    - Implementation: `src/neutryx/models/rough_vol.py`

### Local Volatility

14. **Dupire, B.** (1994). "Pricing with a Smile." *Risk*, 7(1), 18-20.
    - Dupire's local volatility formula
    - Forward Kolmogorov equation approach
    - Implementation: `src/neutryx/models/dupire.py`

15. **Derman, E., & Kani, I.** (1994). "Riding on a Smile." *Risk*, 7(2), 32-39.
    - Alternative derivation of local volatility
    - Implied binomial tree construction

### Jump-Diffusion Models

16. **Merton, R. C.** (1976). "Option Pricing when Underlying Stock Returns are Discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.
    - Merton jump-diffusion model
    - Poisson jump process with lognormal jump sizes
    - Implementation: `src/neutryx/models/jump_diffusion.py`

17. **Kou, S. G.** (2002). "A Jump-Diffusion Model for Option Pricing." *Management Science*, 48(8), 1086-1101.
    - Double exponential jump-diffusion model
    - Asymmetric jumps with exponential distributions
    - Implementation: `src/neutryx/models/kou.py`

18. **Madan, D. B., Carr, P., & Chang, E. C.** (1998). "The Variance Gamma Process and Option Pricing." *European Finance Review*, 2(1), 79-105.
    - Variance gamma model as pure jump Lévy process
    - Gamma time change representation
    - Implementation: `src/neutryx/models/variance_gamma.py`

### Interest Rate Models

19. **Vasicek, O.** (1977). "An Equilibrium Characterization of the Term Structure." *Journal of Financial Economics*, 5(2), 177-188.
    - Mean-reverting Gaussian short rate model
    - Analytical bond pricing formulas
    - Implementation: `src/neutryx/models/vasicek.py`

20. **Cox, J. C., Ingersoll, J. E., & Ross, S. A.** (1985). "A Theory of the Term Structure of Interest Rates." *Econometrica*, 53(2), 385-407.
    - CIR model with square-root diffusion
    - Non-negative interest rates, Feller condition
    - Implementation: `src/neutryx/models/cir.py`

21. **Hull, J., & White, A.** (1990). "Pricing Interest-Rate-Derivative Securities." *Review of Financial Studies*, 3(4), 573-592.
    - Extended Vasicek with time-dependent parameters
    - Calibration to initial term structure
    - Implementation: `src/neutryx/models/hull_white.py`

### Numerical Methods

22. **Carr, P., & Madan, D. B.** (1999). "Option Valuation using the Fast Fourier Transform." *Journal of Computational Finance*, 2(4), 61-73.
    - FFT-based option pricing using characteristic functions
    - Fast computation of option prices across strikes
    - Implementation: `src/neutryx/engines/fourier.py`

23. **Fang, F., & Oosterlee, C. W.** (2008). "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions." *SIAM Journal on Scientific Computing*, 31(2), 826-848.
    - COS method for Fourier pricing
    - Superior stability and accuracy
    - Implementation: `src/neutryx/engines/fourier.py`

24. **Longstaff, F. A., & Schwartz, E. S.** (2001). "Valuing American Options by Simulation: A Simple Least-Squares Approach." *Review of Financial Studies*, 14(1), 113-147.
    - LSM algorithm for American option pricing
    - Regression-based early exercise boundary
    - Implementation: `src/neutryx/engines/longstaff_schwartz.py`

25. **Andersen, L.** (2008). "Simple and Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance*, 11(3), 1-42.
    - QE (Quadratic-Exponential) scheme for Heston
    - Avoids negative variance, maintains moments
    - Implementation: `src/neutryx/models/heston.py:simulate()`

26. **Giles, M. B.** (2008). "Multilevel Monte Carlo Path Simulation." *Operations Research*, 56(3), 607-617.
    - MLMC for variance reduction
    - Optimal allocation across refinement levels
    - Implementation: `src/neutryx/engines/qmc.py`

27. **Broadie, M., & Glasserman, P.** (1996). "Estimating Security Price Derivatives Using Simulation." *Management Science*, 42(2), 269-285.
    - Pathwise derivative method for Greeks
    - Likelihood ratio method
    - Implementation: `src/neutryx/engines/pathwise.py`

### Variance Reduction

28. **Glasserman, P., Heidelberger, P., & Shahabuddin, P.** (1999). "Asymptotically Optimal Importance Sampling and Stratification for Pricing Path-Dependent Options." *Mathematical Finance*, 9(2), 117-152.
    - Optimal importance sampling for rare events
    - Stratified sampling strategies
    - Implementation: `src/neutryx/engines/variance_reduction.py`

29. **Clewlow, L., & Carverhill, A.** (1994). "On the Simulation of Contingent Claims." *Journal of Derivatives*, 2(2), 66-74.
    - Control variate techniques
    - Moment matching methods
    - Implementation: `src/neutryx/engines/variance_reduction.py`

### Calibration

30. **Cont, R., & Tankov, P.** (2009). "Constant Proportion Portfolio Insurance in the Presence of Jumps in Asset Prices." *Mathematical Finance*, 19(3), 379-401.
    - Calibration of jump-diffusion models to market data
    - Regularization techniques for ill-posed problems

31. **Guyon, J., & Henry-Labordère, P.** (2014). *Nonlinear Option Pricing*. Chapman & Hall/CRC.
    - Advanced calibration methods
    - Particle method for SLV models
    - Regularization and stability

### PDE Methods

32. **Wilmott, P., Howison, S., & Dewynne, J.** (1995). *The Mathematics of Financial Derivatives: A Student Introduction*. Cambridge University Press.
    - PDE formulation of option pricing
    - Finite difference schemes
    - Implementation: `src/neutryx/models/pde.py`

33. **Tavella, D., & Randall, C.** (2000). *Pricing Financial Instruments: The Finite Difference Method*. Wiley.
    - Crank-Nicolson and theta schemes
    - American options with constraint
    - Implementation: `src/neutryx/models/pde.py`

### Risk Management

34. **Artzner, P., Delbaen, F., Eber, J. M., & Heath, D.** (1999). "Coherent Measures of Risk." *Mathematical Finance*, 9(3), 203-228.
    - Axiomatic foundation of risk measures
    - CVaR (Expected Shortfall) as coherent measure

35. **Rockafellar, R. T., & Uryasev, S.** (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3), 21-41.
    - CVaR optimization techniques
    - Portfolio risk management
    - Implementation: `src/neutryx/valuations/risk_metrics.py`

36. **Kupiec, P. H.** (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives*, 3(2), 73-84.
    - VaR backtesting methodology
    - Statistical tests for model validation
    - Implementation: `src/neutryx/valuations/risk_metrics.py:backtest_var()`

### XVA and Counterparty Risk

37. **Brigo, D., Morini, M., & Pallavicini, A.** (2013). *Counterparty Credit Risk, Collateral and Funding: With Pricing Cases for All Asset Classes*. Wiley.
    - CVA/DVA calculation methodology
    - Collateral modeling, CSA agreements
    - Implementation: `src/neutryx/valuations/xva/`

38. **Pykhtin, M., & Zhu, S.** (2007). "A Guide to Modeling Counterparty Credit Risk." *GARP Risk Review*, July/August.
    - EPE, PFE calculation
    - Wrong-way risk modeling
    - Implementation: `src/neutryx/valuations/exposure.py`

39. **ISDA SIMM Methodology** (2021). "ISDA SIMM Methodology Version 2.4."
    - Standard Initial Margin Model
    - Risk weight calibration
    - Implementation: `src/neutryx/valuations/margin/simm/`

40. **Albanese, C., & Andersen, L.** (2014). "Accounting for OTC Derivatives: Funding Adjustments and the Re-Hypothecation Option." *Risk Magazine*, January.
    - FVA (Funding Valuation Adjustment) theory
    - Asymmetric funding costs
    - Implementation: `src/neutryx/valuations/xva/fva.py`

## Review Articles and Surveys

41. **Gatheral, J., Jaisson, T., & Rosenbaum, M.** (2018). "Volatility is Rough." *Quantitative Finance*, 18(6), 933-949.
    - Empirical evidence for rough volatility
    - Fractional stochastic volatility models

42. **Andersen, L., & Piterbarg, V.** (2010). *Interest Rate Modeling* (3 volumes). Atlantic Financial Press.
    - Comprehensive reference for rates derivatives
    - LMM, cross-currency models, hybrid models

43. **Rebonato, R.** (2004). *Volatility and Correlation: The Perfect Hedger and the Fox* (2nd ed.). Wiley.
    - Market models for interest rates
    - Correlation modeling, copulas

## Computational Methods

44. **Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q.** (2018). "JAX: Composable transformations of Python+NumPy programs." http://github.com/google/jax
    - Automatic differentiation framework
    - JIT compilation, GPU/TPU support
    - Core technology: All Neutryx implementations

45. **Sobol, I. M.** (1967). "On the Distribution of Points in a Cube and the Approximate Evaluation of Integrals." *USSR Computational Mathematics and Mathematical Physics*, 7(4), 86-112.
    - Sobol sequences for quasi-Monte Carlo
    - Implementation: `src/neutryx/engines/qmc.py`

## Industry Standards

46. **FpML (Financial products Markup Language)** Specification Version 5.12. http://www.fpml.org/
    - Industry standard for derivatives representation
    - Implementation: `src/neutryx/integrations/fpml/`

47. **Basel Committee on Banking Supervision** (2019). "Minimum Capital Requirements for Market Risk."
    - Regulatory framework for market risk
    - Internal models approach, standardized approach

## Additional References by Topic

### Greeks and Sensitivities

48. **Haug, E. G.** (2007). *The Complete Guide to Option Pricing Formulas* (2nd ed.). McGraw-Hill.
    - Comprehensive collection of analytical formulas
    - Greeks for various option types
    - Implementation reference: `src/neutryx/valuations/greeks/`

### Path-Dependent Options

49. **Curran, M.** (1994). "Valuing Asian and Portfolio Options by Conditioning on the Geometric Mean Price." *Management Science*, 40(12), 1705-1711.
    - Moment matching for Asian options
    - Analytical approximations

50. **Broadie, M., Glasserman, P., & Kou, S. G.** (1997). "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance*, 7(4), 325-349.
    - Discrete monitoring bias correction
    - Barrier option adjustments

### Exotic Options

51. **Lipton, A.** (2001). *Mathematical Methods for Foreign Exchange: A Financial Engineer's Approach*. World Scientific.
    - FX options, quanto products
    - Green's function approach

52. **Zhang, P. G.** (1998). *Exotic Options: A Guide to Second Generation Options* (2nd ed.). World Scientific.
    - Comprehensive catalog of exotic products
    - Pricing methodologies

## Software and Tools References

53. **QuantLib** - Open-source library for quantitative finance. https://www.quantlib.org/
    - Reference implementation for many models
    - Industry standard C++ library

54. **Weights & Biases** - MLOps platform for experiment tracking. https://wandb.ai/
    - Integration: Calibration tracking in `src/neutryx/calibration/`

55. **MLflow** - Open-source platform for ML lifecycle. https://mlflow.org/
    - Integration: Model versioning and tracking

---

## Citation Style

Throughout this documentation, we reference papers using the format:

> [Author(s), Year] - Brief description
>
> Implementation: `file/path.py:function_name()`

This links theoretical foundations to practical implementations in the Neutryx codebase.

---

## Notes on Theoretical Rigor

The Neutryx implementation prioritizes:

1. **Mathematical Correctness**: All models follow rigorous derivations from the cited literature
2. **Numerical Stability**: Methods are chosen for robustness (e.g., QE scheme for Heston)
3. **Performance**: JAX enables GPU acceleration while maintaining clarity
4. **Validation**: Extensive test coverage against analytical solutions and benchmarks

See individual model documentation for detailed mathematical specifications and derivations.
