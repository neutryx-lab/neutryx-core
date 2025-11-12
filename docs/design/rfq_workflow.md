---
title: RFQ Workflow and Post-Trade Automation
status: draft
last_updated: 2024-05-12
---

# RFQ Workflow and Auction Design

This document captures the current state of the RFQ implementation under
`src/neutryx/integrations/clearing/rfq.py` and proposes concrete workflow,
auction, and API semantics for orchestrating cleared RFQs end-to-end. The
workflow covers the RFQ lifecycle, quote collection, auction execution,
trade booking, confirmation/affirmation, and generation of settlement
instructions.

## 1. Existing Capabilities

### RFQ Core Models

The RFQ module already provides rich primitives:

* **`RFQ`** – contains requester, product specification, auction settings,
  timing, and execution results.
* **`Quote`** – captures quotes submitted by liquidity providers with
  quantity/price, constraints, and execution status.
* **`AuctionEngine` hierarchy** – implements single-price, multi-price,
  dutch, vickrey, and continuous auctions. Each engine enforces deadline
  checks and maintains competition metrics.
* **`RFQManager`** – orchestrates RFQ creation, quote submission, auction
  execution, and maintains in-memory stores of RFQs/quotes/results.

These classes already use common building blocks such as `Party`, `Trade`,
and `TradeEconomics` from `base.py`, making them compatible with the rest of
the clearing ecosystem.

### Confirmation and Settlement Models

The clearing package also contains reusable models for post-trade steps:

* **`confirmation.py`** provides `Confirmation`, `ConfirmationDetails`, and
  matching/affirmation enums.
* **`settlement_instructions.py`** exposes `SettlementInstruction`,
  `CashFlow`, and `SecuritiesMovement` for settlement directives.

The new workflow service will compose these models rather than introducing
duplicate representations.

## 2. RFQ Workflow State Machine

The proposed orchestrator manages RFQ state transitions using a dedicated
enumeration. Each transition validates prerequisites enforced by the
`RFQManager` and downstream services.

```text
 ┌─────────────────┐  create_rfq   ┌───────────────────────┐
 │   RFQ_CREATED   │──────────────▶│   RFQ_SUBMITTED       │
 └─────────────────┘               └───────────────────────┘
          │ submit_quote(s)                    │ execute_auction
          ▼                                    ▼
 ┌─────────────────┐                ┌─────────────────────────┐
 │ QUOTE_COLLECTION│──────────────▶│   AUCTION_EXECUTED       │
 └─────────────────┘                └─────────────────────────┘
                                              │ book_trade
                                              ▼
                                 ┌─────────────────────────┐
                                 │      TRADE_BOOKED       │
                                 └─────────────────────────┘
                                              │ attach_confirmations
                                              ▼
                                 ┌─────────────────────────┐
                                 │  CONFIRMATION_MATCHED   │
                                 └─────────────────────────┘
                                              │ affirm_confirmations
                                              ▼
                                 ┌─────────────────────────┐
                                 │   AFFIRMATION_COMPLETE  │
                                 └─────────────────────────┘
                                              │ create_settlement_instruction
                                              ▼
                                 ┌─────────────────────────┐
                                 │SETTLEMENT_INSTRUCTED    │
                                 └─────────────────────────┘
                                              │ mark_settled
                                              ▼
                                 ┌─────────────────────────┐
                                 │      SETTLED            │
                                 └─────────────────────────┘
```

Transitions are idempotent where possible and raise explicit errors when
preconditions fail (e.g., attempting to book a trade prior to executing an
auction).

## 3. Auction Rules

All auction engines keep the existing rules. The orchestrator layers on
workflow semantics:

* Quotes can be submitted until `quote_deadline`. A new `force` flag on the
  workflow allows tests or administrators to bypass the deadline check for
  replay scenarios.
* Auctions reuse the pricing/allocation logic from each engine. The
  workflow simply captures the winning quotes and associates the result to
  the RFQ state.
* Competition metrics and allocations from `AuctionResult` are propagated to
  downstream services so confirmations and settlements can reference the
  execution price and filled quantity.

## 4. REST API Endpoints

The workflow service can be exposed via HTTP using the following resource
contract. Routes use `/api/v1/clearing` as the prefix.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/rfqs` | Create an RFQ. Body mirrors `RFQSpecification`, `side`, `auction_type`, and optional metadata. |
| `POST` | `/rfqs/{rfq_id}/submit` | Transition RFQ from draft to open. |
| `POST` | `/rfqs/{rfq_id}/quotes` | Submit a quote. Payload accepts the `Quote` fields with implicit `rfq_id`. |
| `POST` | `/rfqs/{rfq_id}/auction` | Execute the configured auction. Optional query parameter `force=true` bypasses the deadline check. |
| `POST` | `/rfqs/{rfq_id}/trade` | Book the cleared trade, providing `trade_id`, `buyer`, `seller`, and settlement preferences. |
| `POST` | `/rfqs/{rfq_id}/confirmations` | Generate bilateral confirmations and perform matching. |
| `POST` | `/rfqs/{rfq_id}/affirm` | Affirm matched confirmations. |
| `POST` | `/rfqs/{rfq_id}/settlement-instructions` | Create settlement instructions referencing confirmations. |
| `POST` | `/rfqs/{rfq_id}/settle` | Mark settlement complete / update status. |
| `GET`  | `/rfqs/{rfq_id}` | Retrieve RFQ details, quotes, auction result, and workflow status snapshot. |

Each endpoint returns a resource envelope containing the RFQ state, the data
artifact produced, and any relevant metrics (e.g., auction competition score).

## 5. Post-Trade Service / Job

To reuse existing confirmation and settlement models, we introduce a
`PostTradeProcessingService` responsible for:

1. **Confirmation generation** – builds bilateral `Confirmation` instances
   from the booked `Trade`, applying tolerance configuration and referencing
   execution prices.
2. **Matching & affirmation** – compares key fields, sets `match_score`, and
   moves confirmations through `SENT → RECEIVED → MATCHED → AFFIRMED`.
3. **Settlement instruction creation** – constructs `SettlementInstruction`
   using `CashFlow` or `SecuritiesMovement` definitions, generating payment
   references and linking back to confirmations.
4. **Instruction monitoring** – updates settlement status (e.g., instructed,
   affirmed, settled) and records fail reasons when applicable.

This service is stateless and can be run inside a background job that polls
for auctions marked as executed. The workflow orchestrator coordinates state
transitions while the service performs deterministic transformations and
status updates.

## 6. Integration Testing Strategy

An end-to-end integration test under `tests/integration/test_rfq_workflow.py`
validates the workflow:

1. Create an RFQ, submit quotes, and run a single-price auction.
2. Book a trade using the auction result and verify workflow state changes.
3. Generate confirmations, perform matching, and mark them affirmed.
4. Produce settlement instructions and mark the workflow as settled.

Assertions ensure each state transition occurs, the winning quote price is
propagated into trade economics, confirmations contain expected values, and
settlement cash flows tie back to the trade notional. This test serves as a
template for exercising additional auction types or CCP integrations.

## 7. Future Extensions

* Persist workflow state to a database or message bus for fault tolerance.
* Add SLA timers for each state to drive notifications/escalations.
* Support partial fills and multi-allocation bookings by extending the trade
  booking helper to create multiple `Trade` objects per auction result.
* Integrate with SWIFT/CLS connectors for real settlement message emission.

---

The combination of the RFQ workflow orchestrator, post-trade service, and
integration tests delivers a complete automated path from RFQ initiation to
settlement instruction, ready for incremental production hardening.

