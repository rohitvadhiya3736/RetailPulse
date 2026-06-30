
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.loader import get_settings

np.random.seed(42)


def generate(n_rows: int = 50000) -> pd.DataFrame:
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2021-12-31")
    dates = pd.date_range(start, end, freq="h")
    date_samples = np.random.choice(dates, n_rows)

    countries = ["United Kingdom", "Germany", "France", "Spain", "Netherlands", "Belgium", "Italy"]
    country_weights = [0.45, 0.12, 0.1, 0.08, 0.08, 0.07, 0.1]

    n_customers = 4000
    customer_ids = [f"C{i:05d}" for i in range(n_customers)]
    # Power-law: loyal vs churned cohorts for clear ML signal
    loyal = np.random.choice(customer_ids, size=int(n_customers * 0.35), replace=False)
    churned = np.random.choice(
        [c for c in customer_ids if c not in loyal],
        size=int(n_customers * 0.25),
        replace=False,
    )

    rows = []
    stock_codes = [f"S{i:05d}" for i in range(500)]
    descriptions = [
        "HOME DECOR LAMP", "GIFT BOX SET", "RETRO MUG", "PARTY BAG",
        "SCARF RED", "KITCHEN PLATE", "TOY GAME", "NOTEBOOK A4",
    ]

    for i in range(n_rows):
        roll = np.random.random()
        if roll < 0.55:
            cid = np.random.choice(loyal)
            dt = date_samples[i]
        elif roll < 0.75:
            cid = np.random.choice(churned)
            # Churned customers: purchases only in early period
            dt = pd.Timestamp(np.random.choice(dates[: len(dates) // 2]))
        else:
            cid = np.random.choice(customer_ids)
            dt = date_samples[i]
        qty = max(1, int(np.random.lognormal(0.5, 0.8)))
        price = round(np.random.lognormal(2.0, 0.6), 2)
        rows.append({
            "Invoice": f"INV{np.random.randint(100000, 999999)}",
            "StockCode": np.random.choice(stock_codes),
            "Description": np.random.choice(descriptions) + f" {np.random.randint(1,99)}",
            "Quantity": qty,
            "InvoiceDate": dt,
            "Price": price,
            "Customer ID": cid,
            "Country": np.random.choice(countries, p=country_weights),
        })

    return pd.DataFrame(rows)


def main() -> None:
    settings = get_settings()
    out = settings.path("paths", "raw_data")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
