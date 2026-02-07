# Databusters 2026 — NUS x NTU Datathon

**Anatomy of a Run: Terra-Luna 2022 vs Reserve Primary Fund 2008**

Comparative analysis of financial run dynamics across traditional and crypto finance.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the following in the project root:
- `ERC20-stablecoins-001/` — Token transfer data and price CSVs
- `gfc data/` — 2008 GFC market data (VIX, S&P500, TED spread, bank stocks)

## Run

```bash
python Databusters26_Code.py
```

## Output

- `figures/` — 8 publication-quality analysis figures (PNG + PDF)
- `figures/summary_statistics.csv` — Key metrics
- `Databusters26_Report.pdf` — 10-slide presentation deck

## Analysis

**Section A — Run Dynamics**
- Q1: When the Peg Breaks (onset and spread comparison)
- Q2: Where Does the Money Go (flight-to-safety patterns)
- Q3: Who Bears the Losses (institutional design and recovery)

**Section B — Designing for Confidence**
- Hybrid stability framework: redemption gates + over-collateralization

**Novel Contribution:** Panic Index — composite on-chain metric for real-time crisis detection
