# Databusters 2026 - NUS x NTU

Analysis of financial runs: Terra-Luna 2022 vs Reserve Primary Fund 2008.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the following in the project root:
- `ERC20-stablecoins-001/` - Token transfer data and price CSVs
- `gfc data/` - 2008 GFC market data (VIX, S&P500, TED spread)

## Run

```bash
python Databusters26_FINAL_Code.py
```

Output figures are saved to `./figures/`
