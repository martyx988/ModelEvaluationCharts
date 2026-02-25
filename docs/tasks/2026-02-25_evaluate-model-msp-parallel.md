# Task Contract
- Goal: Upravit MSP report tak, aby první dva finální grafy byly vedle sebe s tooltip nápovědou, a doplnit campaign-clients sekci se 3 novými grafy napojenými na horní slidery.
- Acceptance Criteria:
  - `EvaluateModel_msp` generuje horní 2 grafy vedle sebe.
  - Textové vysvětlivky jsou přesunuty do tooltip ikon u grafů.
  - Při `include_campaign_selection=True` jsou vykresleny 3 nové campaign grafy (1 široký + 2 spodní vedle sebe).
  - Dolní campaign gain/success grafy používají pouze campaign subset a očekávané úspěchy dle `score`.
  - Horní cutoff a desired-rate slidery aktualizují i nové campaign grafy.
- Non-Goals:
  - Přepis původní `EvaluateModel` report větve.
- Assumptions:
  - Očekávané úspěchy pro campaign subset jsou aproximovány jako `expected_success = score`.

# Strategic Plan
1. Přepsat MSP větev reportu na vlastní render flow.
2. Přidat nové campaign metriky/grafy pro omezenou bázi.
3. Napojit všechny grafy na stávající interaktivní slidery.
4. Ověřit automaticky + screenshot.

# Tactical Plan
- [x] Napsat MSP helpery pro top gain + success grafy ve 2 sloupcích.
- [x] Přidat tooltip ikonky místo pravého textového panelu.
- [x] Přidat campaign distribution graf a 2 campaign expected-performance grafy.
- [x] Doplnit JS synchronizaci sliderů pro všechny nové grafy.
- [x] Upravit/rozšířit testy pro nové MSP chování.
- [x] Spustit testy a vygenerovat screenshot reportu.
- [x] Přesunout MSP implementaci do samostatného `evaluate_model_msp.py` a notebook napojit na nový modul.

# Architecture Notes
- `EvaluateModel_msp` je samostatný report renderer v odděleném souboru `evaluate_model_msp.py`, aby se MSP vývoj nemíchal s původním `evaluate_model.py`.
- Campaign expected metriky jsou počítány nad `latest_model_score` omezeným na klíče z `campaign_clients`.
- Interaktivita: jeden pár sliderů řídí cutoff ve všech MSP grafech přes JS `Plotly.relayout`.

# Test Plan
- Automated:
  - `python -m pytest -q`
  - nový test layoutu MSP top grafů vedle sebe
  - aktualizovaný MSP test přítomnosti 3 campaign grafů a expected-score textu
- Manual:
  - otevřít `outputs/model_evaluation_report_msp.html`
  - ověřit posuvníky a synchronní aktualizaci horních i campaign grafů

# Progress Log
- 2026-02-25: Přepsána MSP report větev na vlastní HTML+Plotly layout.
- 2026-02-25: Přidány helpery pro campaign distribution + expected gain/success grafy.
- 2026-02-25: Slidery napojeny i na nové campaign grafy.
- 2026-02-25: Upraveny testy, všechny testy zelené.
- 2026-02-25: Vygenerován screenshot nové MSP UI.
- 2026-02-25: MSP implementace přesunuta do samostatného `evaluate_model_msp.py`; `testing_msp.ipynb` importuje nový modul.

# Final Summary
- MSP report nyní odpovídá cílovému wireframu (2 horní grafy vedle sebe + 3 campaign grafy pod nimi).
- MSP kód je oddělen od původního reportu v samostatném python souboru.
- Campaign grafy používají jen campaign subset a expected outcomes dle score.
- Top slidery ovládají všechny grafy konzistentně.
