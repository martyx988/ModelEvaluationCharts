# Task Contract
- Goal: Přidat nový notebook `testing_msp.ipynb` pro rychlé spuštění MSP reportu a kontrolu výsledného HTML.
- Acceptance Criteria:
  - Notebook existuje v `notebooks/testing_msp.ipynb`.
  - Notebook umí vygenerovat `outputs/model_evaluation_report_msp.html` přes `EvaluateModel_msp`.
  - Notebook obsahuje krok pro otevření/náhled výsledného HTML.
- Non-Goals:
  - Přepis produkční logiky reportu.

# Strategic Plan
1. Vytvořit minimální notebook s reproducibilním flow.
2. Ověřit validitu notebook JSON.
3. Commitnout změnu.

# Tactical Plan
- [x] Přidat import buňky (`EvaluateModel_msp`, `create_simulated_tables`).
- [x] Přidat buňku pro vytvoření `campaign_clients` + generování MSP reportu.
- [x] Přidat buňky s cestou na HTML a IFrame náhledem.
- [x] Ověřit JSON validitu notebooku.

# Test Plan
- `python -m json.tool notebooks/testing_msp.ipynb`

# Progress Log
- 2026-02-25: Přidán nový notebook `notebooks/testing_msp.ipynb`.
- 2026-02-25: Ověřena JSON validita notebooku.

# Final Summary
- Notebook pro MSP smoke-test je připraven a generuje výsledné HTML pro vizuální kontrolu.
