# core.py — pure Python logica voor Mini_TUG (geen Streamlit)

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

# Zelfde path als in app.py
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "mini_tug.db"


def load_data():
    """
    Lees invoices en bank_tx uit mini_tug.db.
    Geen Streamlit, alleen pure pandas.
    """
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    con = sqlite3.connect(DB_PATH)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )["name"].tolist()

        inv = (
            pd.read_sql_query("SELECT * FROM invoices", con)
            if "invoices" in tables
            else pd.DataFrame()
        )
        bank = (
            pd.read_sql_query("SELECT * FROM bank_tx", con)
            if "bank_tx" in tables
            else pd.DataFrame()
        )
    finally:
        con.close()

    return inv, bank


def get_kpis():
    """
    Basis-KPIs voor de Next.js frontend.

    Geeft een dict terug met:
    - invoices_count
    - bank_count
    - total_revenue
    - collection_rate
    """
    inv, bank = load_data()

    # Aantallen rijen
    invoices_count = int(len(inv))
    bank_count = int(len(bank))

    # Totale omzet = som van amount voor revenue-invoices
    if not inv.empty and "type" in inv.columns and "amount" in inv.columns:
        revenue_rows = inv[inv["type"] == "revenue"].copy()
        total_revenue = float(revenue_rows["amount"].sum())
    else:
        revenue_rows = pd.DataFrame()
        total_revenue = 0.0

    # Collection rate = matched revenue / totale revenue
    if (
        not revenue_rows.empty
        and "match_id" in revenue_rows.columns
        and "amount" in revenue_rows.columns
    ):
        matched_rows = revenue_rows[revenue_rows["match_id"].notna()]
        matched_amt = float(matched_rows["amount"].sum()) if not matched_rows.empty else 0.0
        total_rev_all = float(revenue_rows["amount"].sum())
        collection_rate = matched_amt / total_rev_all if total_rev_all > 0 else 0.0
    else:
        collection_rate = 0.0

    return {
        "invoices_count": invoices_count,
        "bank_count": bank_count,
        "total_revenue": float(total_revenue),
        "collection_rate": float(collection_rate),
    }
