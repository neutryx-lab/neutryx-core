# Security Design Description (SDD)
## ADV_TDS: TOE Design Specification

**Document Classification:** Internal - Security Architecture
**Target of Evaluation:** Neutryx Financial Engineering Library
**Version:** 1.0
**Date:** 2025-12-30

---

## 1. Executive Summary

This document presents the TOE Design for the Neutryx financial engineering library, structured in accordance with Common Criteria ADV_TDS requirements. The architecture has been decomposed into five core subsystems based on architectural recovery from implementation artefacts. Each subsystem is described with its constituent modules and security-relevant properties.

---

## 2. Subsystem Architecture Overview

The Neutryx TOE comprises five logically distinct subsystems, each encapsulating cohesive functional responsibilities and security concerns:

| Subsystem | Primary Function | Security Relevance |
|-----------|------------------|-------------------|
| **NumericKernel** | Mathematical foundations and numerical computation | FDP_SDI.1, FPT_PHP.1 |
| **ModelEngine** | Stochastic modelling and calibration | FDP_ACC.1, FMT_MSA.1 |
| **ProductLogic** | Financial instrument definitions and lifecycle | FDP_ITC.1, FDP_ETC.1 |
| **RiskFramework** | Risk metrics, XVA, and regulatory reporting | FDP_ACC.1, FAU_GEN.1 |
| **Infrastructure** | Data connectivity, persistence, and observability | FIA_UAU.1, FPT_STM.1, FMT_SMR.1 |

---

## 3. Subsystem Definitions

### 3.1 NumericKernel

**Purpose:**
The NumericKernel subsystem provides foundational mathematical primitives for numerical computation, ensuring stability, precision, and integrity of floating-point operations across the TOE.

**Key Modules:**

- **Interpolation & Curve Mathematics:**
  - `CurveBuilder`, `MultiCurveBuilder`, `BootstrappedCurve`, `FlatCurve`, `Curve`, `CurveBumper`, `CurveBump`

- **Mesh & Grid Management:**
  - `MeshConfig`

- **Parallel Computation:**
  - `ParallelConfig`, `ChunkedSimConfig`

- **Variance Reduction Techniques:**
  - `VarianceReductionEngine`, `VarianceReductionConfig`

**Security Relevance:**

The NumericKernel enforces numerical integrity (FDP_SDI.1) by implementing controlled propagation of NaN and infinite values, singularity detection, and tolerance-based convergence criteria (`ToleranceConfig`). Deterministic floating-point behaviour is critical to ensure reproducibility of risk calculations and regulatory capital figures, thereby supporting audit trail integrity (FAU_GEN.1). The subsystem also implements physical protection against side-channel leakage via timing-constant implementations where numerically sensitive operations occur (FPT_PHP.1).

---

### 3.2 ModelEngine

**Purpose:**
The ModelEngine subsystem encapsulates all stochastic models, volatility surfaces, calibration engines, and model risk assessment capabilities. It is responsible for transforming market data into calibrated model parameters.

**Key Modules:**

- **Interest Rate Models:**
  - `InterestRateModel`, `G2PPInterestRateModel`, `QuasiGaussianInterestRateModel`, `AlmgrenChrissModel`

- **Credit Models:**
  - `CreditMetricsParams`, `MertonModel`, `MertonModelParams`, `KMVModel`, `ECLModel`, `ReducedFormModel`, `JarrowTurnbullModel`, `DuffieSingletonModel`, `BlackCoxModel`

- **Equity & FX Models:**
  - `FXHestonModel`, `FXBatesModel`, `FXSABRModel`, `TwoFactorFXModel`, `CharacteristicFunctionModel`, `BlackScholesCharacteristicModel`, `BinomialModel`

- **Volatility Surfaces:**
  - `VolatilitySurface`, `VolatilitySurfaceCalibration`, `VolSurfaceBuilder`, `FXVolatilitySurface`, `FXVolatilitySurfaceBuilder`, `ImpliedVolSurface`, `LocalVolSurface`, `SABRSurface`, `CapletVolSurface`, `SmileModel`, `SpreadModel`, `Surface`, `SurfaceBumper`

- **Calibration & Model Risk:**
  - `ModelFit`, `ModelComparison`, `ModelWeights`, `BayesianModelAveraging`, `IdentifiabilityMetrics`, `ModelRiskCalculator`, `ModelRiskMetrics`, `ModellabilityTester`, `ModellabilityTestResult`, `ModellabilityStatus`, `ModelWorkflow`

- **Curve Builders:**
  - `YieldCurve`, `YieldCurveOption`, `CreditCurve`, `CreditCurveType`, `CreditSpreadCurve`, `SovereignCreditCurve`, `CDSCurve`, `HazardRateCurve`, `SurvivalProbabilityCurve`, `DiscountCurve`, `ForwardRateCurve`, `FXForwardCurve`, `DividendYieldCurve`, `DividendCurveBuilder`, `EONIACurveBuilder`, `ESTRCurveBuilder`, `SOFRCurveBuilder`, `SONIACurveBuilder`, `TONARCurveBuilder`

**Security Relevance:**

The ModelEngine implements access control (FDP_ACC.1) to ensure that only authorised roles may modify model parameters or override calibration results. Model versioning and lineage tracking support auditability (FAU_GEN.1), enabling reconstruction of historical valuations for regulatory inquiries. The subsystem enforces security attribute management (FMT_MSA.1) to tag models with approval status, validator identity, and effective date ranges. Modellability tests (`ModellabilityTester`) align with FRTB requirements and constitute a control boundary for capital calculations.

---

### 3.3 ProductLogic

**Purpose:**
The ProductLogic subsystem defines financial instruments, their payoffs, lifecycle events, and trade repositories. It is the semantic core of the TOE, translating financial contracts into computational artefacts.

**Key Modules:**

- **Trade Management:**
  - `TradeRepository`, `InMemoryTradeRepository`, `PostgresTradeRepository`, `BookRepository`, `InMemoryBookRepository`

- **Counterparty Management:**
  - `CounterpartyRepository`, `InMemoryCounterpartyRepository`, `PostgresCounterpartyRepository`, `CounterpartyCodeConfig`

- **Lifecycle & Workflow:**
  - `LifecycleManager`, `LifecycleSettlementConfig`, `PostTradeProcessingService`, `SettlementWorkflowConfig`, `ConfirmationManager`, `ConfirmationReconciliationEngine`

- **CSA & Collateral:**
  - `CSAManager`, `CSARepository`, `PostgresCSARepository`

- **RFQ & Execution:**
  - `RFQManager`, `RFQWorkflowService`, `RFQCCPIntegrationService`, `RFQCCPIntegrationConfig`, `TradeExecutionService`, `ExecutionModel`, `SimpleExecutionModel`, `ExecutionConfig`

- **Pricing Adapters:**
  - `FpMLAdapter`, `FpMLPricingAdapter`

**Security Relevance:**

The ProductLogic subsystem enforces information flow control (FDP_ITC.1, FDP_ETC.1) to ensure that trade data imported from external systems undergoes validation and sanitisation before acceptance into the TOE. Repository implementations abstract data persistence, enabling enforcement of separation of duties (FMT_SMR.1) whereby trade capture, confirmation, and settlement are performed by distinct roles. The lifecycle manager ensures tamper-evident state transitions (FPT_STM.1), logging each event for subsequent audit (FAU_GEN.1). CSA management enforces contractual security attributes (collateral thresholds, haircuts) that directly impact credit exposure calculations.

---

### 3.4 RiskFramework

**Purpose:**
The RiskFramework subsystem computes all risk metrics, performs XVA calculations, generates regulatory capital figures, and produces compliance reports mandated by Basel III, FRTB, EMIR, and MiFID II.

**Key Modules:**

- **XVA Calculation:**
  - `XVAEngine`, `CVACalculator`, `DVACalculator`, `FVACalculator`, `KVACalculator`, `MVACalculator`

- **Exposure & PFE:**
  - `ExposureMetric`, `WWREngine`

- **Initial Margin:**
  - `InitialMarginModel`, `SIMMCalculator`, `MarginAggregationService`, `MarginAggregatorConfig`, `MarginTrackerConfig`, `AggregatedMarginReport`

- **Regulatory Capital:**
  - `RegulatoryCapitalEngine`, `CapitalCalculator`, `BaselCapitalCalculator`, `SACCRCalculator`, `NMRFCapitalCalculator`

- **Regulatory Reporting:**
  - `RegulatoryReportEngine`, `RegulatoryReport`, `ReportValidator`, `ReportSubmission`, `ReportStatus`, `ReportType`

- **Basel III Reports:**
  - `BaselCapitalReport`, `BaselCapitalReporter`, `BaselCapitalRegulatoryReport`, `BaselCVARegulatoryReport`, `BaselFRTBRegulatoryReport`, `BaselLeverageRegulatoryReport`, `CVACapitalReport`, `FRTBCapitalReport`, `LeverageRatioReport`, `OperationalRiskReport`

- **EMIR Reports:**
  - `EMIRTradeReport`, `EMIRTradeReporter`, `EMIRTradeRegulatoryReport`, `EMIRLifecycleRegulatoryReport`, `EMIRValuationReport`, `EMIRValuationRegulatoryReport`

- **MiFID II Reports:**
  - `MiFIDTransactionReport`, `MiFIDTransactionReporter`, `MiFIDTransactionRegulatoryReport`, `MiFIDReferenceDataReport`, `MiFIDReferenceDataRegulatoryReport`

- **Compliance & Best Execution:**
  - `ComplianceReport`, `ComplianceReporter`, `BestExecutionReport`

- **P&L Attribution:**
  - `PnLAttributionEngine`, `GreekPLCalculator`, `RiskFactorPLCalculator`

- **Scenario & Stress Testing:**
  - `ScenarioEngine`, `ScenarioReport`

- **Backtesting:**
  - `BacktestEngine`, `BacktestConfig`

- **Concentration Risk:**
  - `ConcentrationRiskMetrics`, `ConcentrationMetric`

- **Portfolio Loss:**
  - `PortfolioLossMetrics`

- **Quality Assurance:**
  - `QualityReport`, `QualityMetrics`

- **Performance & PLA:**
  - `PerformanceMetrics`, `PLAMetrics`

**Security Relevance:**

The RiskFramework is the primary enforcement point for access control policies (FDP_ACC.1) governing risk metrics and regulatory reports. Report generation timestamps are tamper-evident (FPT_STM.1), and all capital calculations are audited (FAU_GEN.1) to enable regulatory reconstruction. The subsystem enforces role-based access control (FMT_SMR.1) such that only authorised risk managers and compliance officers may approve and submit reports. Cryptographic integrity (FCS_COP.1) may be applied to report artefacts to ensure non-repudiation. The XVA engine integrates with the ModelEngine to ensure consistent application of credit, funding, and capital valuation adjustments, thereby preventing arbitrage and ensuring coherent risk aggregation.

---

### 3.5 Infrastructure

**Purpose:**
The Infrastructure subsystem provides horizontal services for data ingestion, persistence, observability, alerting, distributed computation, identity management, and system governance.

**Key Modules:**

- **Market Data Adapters:**
  - `BaseMarketDataAdapter`, `BloombergAdapter`, `BloombergConfig`, `BloombergDataAdapter`, `RefinitivAdapter`, `RefinitivConfig`, `RefinitivDataAdapter`, `ICEDataServicesAdapter`, `ICEDataServicesConfig`, `CMEMarketDataAdapter`, `CMEMarketDataConfig`, `SimulatedAdapter`, `SimulatedConfig`, `AdapterConfig`

- **Feed Management:**
  - `FeedManager`, `FeedConfig`, `FeedMetrics`

- **Curve Market Data:**
  - `CurveMarketData`, `CurveDefinition`, `CurveConfig`, `CurveType`, `CurveMonitor`, `MultiCurveEnvironment`

- **Database Connectivity:**
  - `DatabaseConnector`, `DatabaseConnectorError`, `DatabaseConfig`, `PostgresConnector`, `PostgreSQLConfig`, `MongoConnector`, `MongoDBConfig`, `TimescaleConnector`, `TimescaleDBConfig`, `InMemoryConnector`, `RepositoryFactory`

- **CCP & Clearing Connectivity:**
  - `CCPConnector`, `CCPConfig`, `CCPMetrics`, `CCPReconciliationEngine`, `LCHSwapClearConnector`, `LCHSwapClearConfig`, `ICEClearConnector`, `ICEClearConfig`, `ICEClearService`, `EurexClearingConnector`, `EurexClearingConfig`, `CMEClearingConnector`, `CMEClearingConfig`

- **Settlement & Messaging:**
  - `SettlementManager`, `SettlementRoutingStrategy`, `AutomaticSettlementService`, `CLSConnector`, `CLSSettlementService`, `EuroclearConnector`, `EuroclearSettlementService`, `SwiftConfig`, `BankConnectionManager`

- **Observability & Monitoring:**
  - `ObservabilityConfig`, `PrometheusConfig`, `TracingConfig`, `TrackingConfig`, `MetricsRecorder`, `ProfilingConfig`

- **Alerting:**
  - `AlertManager`, `BaseAlertManager`, `NullAlertManager`, `AlertingConfig`

- **Configuration & Validation:**
  - `AppConfig`, `ConfigValidationError`

- **Computation & Engine Settings:**
  - `RiskEngine`, `EngineSettings`, `AggregationEngine`, `LSMConfig`, `MCConfig`, `SensitivityConfig`, `OptimizationConfig`

- **Checkpointing & Storage:**
  - `CheckpointManager`, `StorageConfig`

- **Distributed Compute:**
  - `ClusterConfig`

- **Identity & Access:**
  - `RBACManager`, `TenantManager`, `GovernanceService`, `LimitManager`, `IDGeneratorConfig`

- **Security Master:**
  - `SecurityMasterDBConfig`

- **Reconciliation:**
  - `ReconciliationConfig`

- **Business Logic:**
  - `BusinessModel`

**Security Relevance:**

The Infrastructure subsystem enforces user authentication and authorisation (FIA_UAU.1) via the `RBACManager` and `TenantManager`, ensuring multi-tenancy isolation and role-based permissions. Secure time stamping (FPT_STM.1) is provided for audit events and trade timestamps. The subsystem implements secure communication channels (FTP_ITC.1) for market data feeds and CCP connectivity, ensuring confidentiality and integrity of data in transit. Observability components (`TracingConfig`, `PrometheusConfig`) support security event logging (FAU_GEN.1) and anomaly detection. The `GovernanceService` enforces policy-based controls over system configuration changes, supporting separation of duties (FMT_SMR.1). Database connectors abstract persistence, enabling enforcement of least privilege and data compartmentalisation. The `CheckpointManager` ensures resilience and recovery, supporting availability objectives.

---

## 4. Subsystem Interaction Model

The subsystems interact in a layered architecture with clear dependency constraints:

```
┌─────────────────────────────────────────────────────────────┐
│                      RiskFramework                          │
│  (XVA, Capital, Regulatory Reports, Compliance)             │
└────────────┬────────────────────────────────────────────────┘
             │
             │ depends on
             ▼
┌────────────────────────────────────────────────────────────┐
│                      ProductLogic                          │
│  (Trades, Instruments, Lifecycle, Repositories)            │
└────────────┬───────────────────────────────────────────────┘
             │
             │ depends on
             ▼
┌────────────────────────────────────────────────────────────┐
│                      ModelEngine                           │
│  (Models, Surfaces, Calibration, Curves)                   │
└────────────┬───────────────────────────────────────────────┘
             │
             │ depends on
             ▼
┌────────────────────────────────────────────────────────────┐
│                      NumericKernel                         │
│  (Interpolation, PDE, Meshing, Parallel Compute)           │
└────────────────────────────────────────────────────────────┘

     All subsystems depend on Infrastructure (horizontal)
┌────────────────────────────────────────────────────────────┐
│                      Infrastructure                         │
│  (Data, DB, Observability, RBAC, CCP, Settlement)          │
└────────────────────────────────────────────────────────────┘
```

**Security Properties of Subsystem Boundaries:**

- **NumericKernel → ModelEngine:** Ensures numerical stability propagates upward; no untrusted floating-point data may bypass kernel validation.
- **ModelEngine → ProductLogic:** Calibrated models are immutable once approved; product definitions reference models via secure identifiers.
- **ProductLogic → RiskFramework:** Trade data flows are subject to integrity checks; risk calculations operate on validated, timestamped trade snapshots.
- **Infrastructure → All:** Authentication, authorisation, and audit trail enforcement are uniform across all subsystems via centralised RBAC and governance services.

---

## 5. Security Functional Requirements Mapping

| SFR | Subsystem(s) | Implementation |
|-----|--------------|----------------|
| FDP_ACC.1 (Access Control) | ModelEngine, RiskFramework, Infrastructure | `RBACManager`, `TenantManager` |
| FDP_SDI.1 (Stored Data Integrity) | NumericKernel, ModelEngine | NaN propagation, tolerance checks |
| FDP_ITC.1 (Import) | ProductLogic, Infrastructure | Data validation on ingestion |
| FDP_ETC.1 (Export) | ProductLogic, RiskFramework | Report validation, format checks |
| FIA_UAU.1 (Authentication) | Infrastructure | `RBACManager` |
| FMT_MSA.1 (Management of Security Attributes) | ModelEngine, RiskFramework | Model approval, report status |
| FMT_SMR.1 (Security Roles) | Infrastructure, ProductLogic | RBAC, separation of duties |
| FAU_GEN.1 (Audit Generation) | All subsystems | Centralised logging via `TracingConfig` |
| FPT_STM.1 (Reliable Time Stamps) | Infrastructure, RiskFramework | Trade timestamps, report generation time |
| FPT_PHP.1 (Passive Detection) | NumericKernel | Timing-constant arithmetic |
| FCS_COP.1 (Cryptographic Operations) | Infrastructure, RiskFramework | Report integrity (optional extension) |

---

## 6. Assurance Continuity Statement

This architecture description satisfies ADV_TDS.2 requirements by identifying subsystems, describing their security functions, and mapping them to SFRs. All modules listed are traceable to implementation artefacts in the Neutryx codebase. Future updates to this document shall maintain synchronisation with the implementation via automated tooling ([audit_sdd_gap.py](audit_sdd_gap.py)) to detect architectural drift.

---

## 7. Compliance & Regulatory Alignment

The RiskFramework subsystem implements:

- **Basel III / CRR II:** Capital adequacy (SA-CCR, FRTB, CVA capital)
- **EMIR:** Trade reporting, lifecycle events, valuation reporting
- **MiFID II:** Transaction reporting, reference data, best execution
- **BCBS 239:** Risk data aggregation and reporting (via `RegulatoryReportEngine`)

These implementations constitute security-relevant functions insofar as they enforce regulatory policy constraints on risk-taking and ensure transparency to supervisory authorities.

---

## 8. Unimplemented / Future Subsystems

The following architectural elements are identified for future extension but are not yet implemented:

- **CryptographicServices:** Dedicated subsystem for key management, digital signatures, and encryption of sensitive trade data.
- **AuditSubsystem:** Segregated audit log repository with write-once semantics and cryptographic integrity.

---

## 9. Document Control

**Author:** Lead Security Architect
**Approver:** Chief Information Security Officer
**Review Cycle:** Quarterly, or upon material architectural change
**Change Log:**

| Version | Date | Change Summary |
|---------|------|----------------|
| 1.0 | 2025-12-30 | Initial ADV_TDS subsystem architecture |

---

**End of Document**
