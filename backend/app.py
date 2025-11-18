from pathlib import Path
import sqlite3
import io, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from google.cloud import documentai as docai
import json
import mimetypes
from google.oauth2 import service_account



st.set_page_config(
    page_title="Welcome to Mini_TUG, your AI-Based ERP for your startup!",
    layout="wide",
)


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DB_PATH = BASE / "mini_tug.db"

# ---------- Document AI (Google) ----------
PROJECT_ID   = "361271679946"
LOCATION     = "eu"
PROCESSOR_ID = "2ee67d07894fd7f1"

DOC_PROCESSOR_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

# Client op EU-endpoint + expliciete service-account credentials
KEY_PATH = BASE / "tug-docai-key.json"

_doc_credentials = service_account.Credentials.from_service_account_file(
    str(KEY_PATH)
)

_doc_client = docai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-documentai.googleapis.com"},
    credentials=_doc_credentials,
)

# -------------------------------
# Helpers: DB / sample / reset / load
# -------------------------------
def db_has_tables() -> bool:
    """Check of er überhaupt een invoices/bank_tx table is."""
    if not DB_PATH.exists():
        return False
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('invoices','bank_tx')"
        )
        return len(cur.fetchall()) > 0
    finally:
        con.close()


def _guess_mime_type(filename: str) -> str:
    mt, _ = mimetypes.guess_type(filename)
    if mt is None:
        if filename.lower().endswith(".pdf"):
            return "application/pdf"
        return "image/png"
    return mt


def call_external_ocr(file_bytes: bytes, filename: str) -> dict:
    """
    Stuurt één bestand naar Google Document AI Invoice Parser
    en geeft het DocumentAI 'document' object terug als dict.
    """
    mime_type = _guess_mime_type(filename)

    raw_document = docai.RawDocument(
        content=file_bytes,
        mime_type=mime_type,
    )

    request = {
        "name": DOC_PROCESSOR_NAME,
        "raw_document": raw_document,
    }

    result = _doc_client.process_document(request=request)
    # result.document is een protobuf; maak er een dict van
    document = docai.Document.to_dict(result.document)
    return document


def _to_float(val, default: float = 0.0) -> float:
    """
    Maak van een bedrag-string zoals 'EUR 1,210.00' een float 1210.00.
    Werkt ook voor '1.210,00' en varianten.
    """
    s = str(val or "").strip()
    if not s:
        return default

    # strip valuta-dingen
    for token in ["EUR", "eur", "€"]:
        s = s.replace(token, "")

    # spaties weg
    s = s.replace(" ", "")

    # case 1: zowel komma als punt -> neem aan: komma = duizendtallen
    if "," in s and "." in s:
        s = s.replace(",", "")      # '1,210.00' -> '1210.00'
    else:
        # case 2: alleen komma -> neem aan: komma = decimaal
        s = s.replace(",", ".")     # '1210,00' -> '1210.00'

    try:
        return float(s)
    except ValueError:
        return default


def parse_invoice_image_to_rows(ocr_doc: dict) -> list[dict]:
    """
    Converteert het DocumentAI resultaat naar rijen voor de 'invoices' tabel.
    Voor nu: 1 invoice per document.
    """
    entities = {e.get("type_"): e for e in ocr_doc.get("entities", [])}

    def _val(key, default=""):
        ent = entities.get(key)
        if not ent:
            return default
        return ent.get("mention_text", default)

    # Typische entity keys van Invoice Parser
    invoice_date = _val("invoice_date", None)
    due_date = _val("due_date", None)
    currency = _val("currency_code", "EUR")
    supplier_name = _val("supplier_name", "")
    invoice_no = _val("invoice_id", "")
    net_amount = _to_float(_val("subtotal", "0"))
    vat_amount = _to_float(_val("total_tax_amount", "0"))
    gross_amount = _to_float(_val("total_amount", "0"))

    row = {
        "date": pd.to_datetime(invoice_date) if invoice_date else pd.NaT,
        "due_date": pd.to_datetime(due_date) if due_date else pd.NaT,
        "amount": gross_amount,          # gross als 'amount'
        "net_amount": net_amount,
        "vat_amount": vat_amount,
        "currency": currency,
        "partner": supplier_name,
        "invoice_no": invoice_no,
        "type": "expense",               # voorlopig: kostenfacturen
        "entity": "TUG_NL",              # hard-coded; later dropdown maken
        "source": "ocr",
        "raw_ocr": json.dumps(ocr_doc),
    }
    return [row]


def load_sample_into_db():
    """Load sample CSVs from /data into SQLite."""
    inv_path = DATA / "invoices.csv"
    bank_path = DATA / "bank_tx.csv"
    if not (inv_path.exists() and bank_path.exists()):
        st.error("Sample CSVs not found in /data.")
        return
    con = sqlite3.connect(DB_PATH)
    try:
        inv_df = pd.read_csv(inv_path, parse_dates=["date"])
        bank_df = pd.read_csv(bank_path, parse_dates=["date"])

        # safety cols voor invoices
        for c in ["match_id", "status", "invoice_no"]:
            if c not in inv_df.columns:
                inv_df[c] = pd.NA

        # safety cols voor bank
        for c in ["match_id", "status"]:
            if c not in bank_df.columns:
                bank_df[c] = pd.NA
        for c in ["partner", "memo"]:
            if c not in bank_df.columns:
                bank_df[c] = pd.NA


        inv_df["month"] = inv_df["date"].dt.to_period("M").dt.to_timestamp()
        bank_df["month"] = bank_df["date"].dt.to_period("M").dt.to_timestamp()

        inv_df.to_sql("invoices", con, if_exists="replace", index=False)
        bank_df.to_sql("bank_tx", con, if_exists="replace", index=False)
    finally:
        con.close()


def reset_db():
    """Delete SQLite DB file."""
    if DB_PATH.exists():
        DB_PATH.unlink()


def load_data():
    """Load invoices/bank ONLY from SQLite if present, else empty."""
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    con = sqlite3.connect(DB_PATH)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )["name"].tolist()

        inv = pd.read_sql_query("SELECT * FROM invoices", con) if "invoices" in tables else pd.DataFrame()
        bank = pd.read_sql_query("SELECT * FROM bank_tx", con) if "bank_tx" in tables else pd.DataFrame()
    finally:
        con.close()

    # normalize dates/month
    for df in (inv, bank):
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if "month" not in df.columns:
                df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return inv, bank


# -------------------------------
# Sidebar: upload + import to DB
# -------------------------------
st.sidebar.markdown("### 1. Data uploads")
up_inv = st.sidebar.file_uploader(
    "Invoices (CSV export)",
    type="csv",
    key="inv_up",
)
up_bank = st.sidebar.file_uploader(
    "Bank transactions (CSV export)",
    type="csv",
    key="bank_up",
)

# -------------------------------
# Sidebar: OCR invoice upload (PDF/JPG/PNG)
# -------------------------------
st.sidebar.markdown("### 2. Scan invoice PDFs (DocAI)")
inv_imgs = st.sidebar.file_uploader(
    "Upload invoice PDF/JPG/PNG",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="inv_imgs",
)

if st.sidebar.button("Scan invoices to SQLite"):
    if not inv_imgs:
        st.sidebar.warning("Upload at least één invoice-bestand.")
    else:
        all_rows = []
        for f in inv_imgs:
            file_bytes = f.read()
            ocr_doc = call_external_ocr(file_bytes, f.name)
            rows = parse_invoice_image_to_rows(ocr_doc)
            all_rows.extend(rows)

        if all_rows:
            new_df = pd.DataFrame(all_rows)
            # Zorg dat 'month' bestaat zoals de rest van de app verwacht
            if "date" in new_df.columns:
                new_df["month"] = new_df["date"].dt.to_period("M").dt.to_timestamp()

            with sqlite3.connect(DB_PATH) as con:
                new_df.to_sql("invoices", con, if_exists="append", index=False)

            st.sidebar.success(f"OCR import succesvol: {len(all_rows)} invoices toegevoegd.")
            st.rerun()
        else:
            st.sidebar.error("OCR gaf geen bruikbare invoice-data terug.")


if st.sidebar.button("Import CSVs to SQLite"):
    if not (up_inv or up_bank):
        st.sidebar.warning("Upload at least one CSV first.")
    else:
        con = sqlite3.connect(DB_PATH)
        try:
            if up_inv:
                inv_df = pd.read_csv(up_inv, parse_dates=["date"])
                inv_df["month"] = inv_df["date"].dt.to_period("M").dt.to_timestamp()
                for c in ["match_id", "status", "invoice_no"]:
                    if c not in inv_df.columns:
                        inv_df[c] = pd.NA
                inv_df.to_sql("invoices", con, if_exists="replace", index=False)


            if up_bank:
                bank_df = pd.read_csv(up_bank, parse_dates=["date"])
                for c in ["partner", "memo"]:
                    if c not in bank_df.columns:
                        bank_df[c] = pd.NA
                bank_df["month"] = bank_df["date"].dt.to_period("M").dt.to_timestamp()
                for c in ["match_id", "status"]:
                    if c not in bank_df.columns:
                        bank_df[c] = pd.NA
                bank_df.to_sql("bank_tx", con, if_exists="replace", index=False)

            st.sidebar.success("Imported to mini_tug.db.")
        finally:
            con.close()

# Sidebar: hard reset DB
if st.sidebar.button("Reset SQLite DB (clear all data)"):
    reset_db()
    st.sidebar.success("Database cleared. No data loaded.")
    st.rerun()

# -------------------------------
# Load data
# -------------------------------
inv, bank = load_data()

# -------------------------------
# Empty state (écht leeg: beide leeg)
# -------------------------------
if inv.empty and bank.empty:
    st.title("Welcome to Mini_TUG, your AI-Based ERP for your startup!")
    st.info("Upload invoices/bank CSVs in the sidebar, scan invoices via OCR, of laad sample data.")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Load sample data from /data"):
            load_sample_into_db()
            st.rerun()

    with c2:
        if st.button("Reset DB"):
            reset_db()
            st.rerun()

    st.stop()

# Ensure reconciliation + text cols exist
for c in ["match_id", "status", "invoice_no"]:
    if c not in inv.columns:
        inv[c] = pd.NA

for c in ["match_id", "status"]:
    if c not in bank.columns:
        bank[c] = pd.NA
for c in ["partner", "memo"]:
    if c not in bank.columns:
        bank[c] = pd.NA

st.sidebar.caption(
    f"Source: {'SQLite mini_tug.db' if DB_PATH.exists() else 'No DB file yet'}"
)

# ---------------------------------
# Main title (data is present)
# ---------------------------------
st.title("Welcome to Mini_TUG (v2.0), your (almost) AI-Based ERP!")

# ---------------------------------
# Filters & recon parameters (sliders + auto)
# ---------------------------------
if not inv.empty and "entity" in inv.columns:
    entities = ["ALL"] + sorted(inv["entity"].astype(str).unique().tolist())
else:
    entities = ["ALL"]

st.sidebar.markdown("### 3. View filters")
ent = st.sidebar.selectbox("Reporting entity", entities, index=0)

st.sidebar.markdown("### 4. Matching rules")
# Reconciliation controls (sliders)
DATE_WINDOW_DAYS = st.sidebar.slider("Date window for match (± days)", min_value=0, max_value=14, value=3, step=1)
AMOUNT_TOL = st.sidebar.slider("Amount tolerance (€)", min_value=0.00, max_value=10.00, value=0.50, step=0.10)

# PSP fee assumptions (for Rule 2)
PSP_MAX_FEE_ABS = st.sidebar.slider("Max PSP fee (absolute)", min_value=0.00, max_value=100.00, value=50.00, step=1.00)
_psp_pct = st.sidebar.slider("Max PSP fee (% of gross)", min_value=0.0, max_value=10.0, value=4.0, step=0.5)
PSP_MAX_FEE_PCT = _psp_pct / 100.0
ONLY_PSP_NAMES = st.sidebar.checkbox("Only treat Stripe/Adyen/etc. as PSP", value=True)

AUTO_RECON = st.sidebar.toggle("Auto-reconcile on change", value=True)
PERSIST_TO_DB = st.sidebar.checkbox("Persist matches to SQLite", value=False)
st.sidebar.caption(f"Auto-recon: {'ON' if AUTO_RECON else 'OFF'} • Persist: {'ON' if PERSIST_TO_DB else 'OFF'}")

# ---------------------------------
# Sidebar Insights navigation
# ---------------------------------
st.sidebar.markdown("### Insights")
view_mode = st.sidebar.radio(
    "Choose view",
    ["Reporting", "Metrics", "Raw data"],
    index=0,
)

# ---------------------------------
# Reconciliation helpers
# ---------------------------------
def fee_ok(gross, net, fee_abs_max, fee_pct_max):
    """Return (True, fee) if net ~= gross - plausible_fee within caps."""
    gross = float(gross)
    net = float(net)
    fee = round(gross - net, 2)
    if fee <= 0:
        return False, 0.0
    if fee <= fee_abs_max and (gross > 0) and (fee / gross) <= fee_pct_max:
        return True, fee
    return False, 0.0


def greedy_many_to_one(open_rows, target_net, tol, fee_abs_max, fee_pct_max):
    """
    Greedy picker: accumulate invoices by date order until
    (sum ≈ target within tol) OR (sum - fee ≈ target with plausible fee).
    Returns (picked_ids, gross_sum, fee, ok).
    """
    rows = open_rows.sort_values("date")
    picked, gross_sum = [], 0.0
    for idx, r in rows.iterrows():
        if pd.notna(r.get("match_id")) and str(r.get("match_id")) != "":
            continue
        amt = float(r["amount"])
        if gross_sum + amt <= target_net + fee_abs_max:
            picked.append(int(idx))
            gross_sum += amt
        if abs(gross_sum - target_net) <= tol:
            return picked, gross_sum, 0.0, True
        ok, fee = fee_ok(gross_sum, target_net, fee_abs_max, fee_pct_max)
        if ok:
            return picked, gross_sum, fee, True
    return [], 0.0, 0.0, False

# ---------------------------------
# Reconciliation (Rules 1–3)
# ---------------------------------
if view_mode == "Reporting":
    st.markdown("## Reconciliation")
    run_recon = AUTO_RECON or st.button("Run reconciliation")
else:
    # Op andere tabs alleen auto-recon, geen knop
    run_recon = AUTO_RECON

total_rule1 = total_rule2 = total_rule3 = 0
recent = []

if run_recon and (not inv.empty) and (not bank.empty):
    # Rule 1: exact 1–1
    inv_u = inv[(inv.get("type") == "revenue") & (inv["match_id"].isna())].copy()
    bank_u = bank[(bank.get("direction") == "in") & (bank["match_id"].isna())].copy()

    matches = []
    for i_idx, irow in inv_u.iterrows():
        cands = bank_u[
            (bank_u["entity"] == irow["entity"])
            & (
                bank_u["amount"].round(2).between(
                    round(irow["amount"] - AMOUNT_TOL, 2),
                    round(irow["amount"] + AMOUNT_TOL, 2),
                )
            )
            & ((bank_u["date"] - irow["date"]).abs() <= pd.Timedelta(days=DATE_WINDOW_DAYS))
        ]
        if len(cands) == 1:
            b_idx = cands.index[0]
            mid = f"M{i_idx}-{b_idx}"
            matches.append((i_idx, b_idx, mid))

    for i_idx, b_idx, mid in matches:
        inv.loc[i_idx, ["match_id", "status"]] = [mid, "Matched"]
        bank.loc[b_idx, ["match_id", "status"]] = [mid, "Matched"]
        recent.append(dict(rule="R1 exact", inv_id=i_idx, bank_id=b_idx, match_id=mid))
    total_rule1 = len(matches)

    # Rule 2: PSP fee-adjusted 1–1
    inv_u2 = inv[(inv.get("type") == "revenue") & (inv["match_id"].isna())].copy()
    bank_u2 = bank[(bank.get("direction") == "in") & (bank["match_id"].isna())].copy()

    if ONLY_PSP_NAMES and ("partner" in bank_u2.columns or "memo" in bank_u2.columns):
        txtcol = "partner" if "partner" in bank_u2.columns else "memo"
        bank_u2 = bank_u2[
            bank_u2[txtcol]
            .fillna("")
            .str.contains(r"stripe|adyen|mollie|paypal|checkout\.com|braintree", case=False, regex=True)
        ]

    psp_matches = []
    for i_idx, irow in inv_u2.iterrows():
        cands = bank_u2[
            (bank_u2["entity"] == irow["entity"])
            & ((bank_u2["date"] - irow["date"]).abs() <= pd.Timedelta(days=DATE_WINDOW_DAYS))
        ]
        for b_idx, brow in cands.iterrows():
            ok, fee = fee_ok(irow["amount"], brow["amount"], PSP_MAX_FEE_ABS, PSP_MAX_FEE_PCT)
            if ok:
                mid = f"F{i_idx}-{b_idx}"
                psp_matches.append((i_idx, b_idx, mid))
                break

    for i_idx, b_idx, mid in psp_matches:
        inv.loc[i_idx, ["match_id", "status"]] = [mid, "Matched"]
        bank.loc[b_idx, ["match_id", "status"]] = [mid, "Matched (fee)"]
        recent.append(dict(rule="R2 fee", inv_id=i_idx, bank_id=b_idx, match_id=mid))
    total_rule2 = len(psp_matches)

    # Rule 3: many-to-one deposit (batch)
    inv_u3 = inv[(inv.get("type") == "revenue") & (inv["match_id"].isna())].copy()
    bank_u3 = bank[(bank.get("direction") == "in") & (bank["match_id"].isna())].copy()

    batch_matches = []
    for b_idx, brow in bank_u3.iterrows():
        cands = inv_u3[
            (inv_u3["entity"] == brow["entity"])
            & ((inv_u3["date"] - brow["date"]).abs() <= pd.Timedelta(days=DATE_WINDOW_DAYS))
        ]
        if cands.empty:
            continue
        ids, gross_sum, fee, ok = greedy_many_to_one(
            cands, float(brow["amount"]), AMOUNT_TOL, PSP_MAX_FEE_ABS, PSP_MAX_FEE_PCT
        )
        if ok and ids:
            mid = f"B{b_idx}-" + ",".join(map(str, ids))
            batch_matches.append((ids, b_idx, mid))

    for ids, b_idx, mid in batch_matches:
        inv.loc[ids, ["match_id", "status"]] = [mid, "Matched"]
        bank.loc[b_idx, ["match_id", "status"]] = [mid, "Matched (batch)"]
        recent.append(
            dict(rule="R3 batch", inv_ids=",".join(map(str, ids)), bank_id=b_idx, match_id=mid)
        )
    total_rule3 = len(batch_matches)

    if view_mode == "Reporting":
        st.success(f"Matched — Rule1: {total_rule1} | Rule2: {total_rule2} | Rule3: {total_rule3}")

        if recent:
            st.markdown("#### Recent matches")
            st.dataframe(pd.DataFrame(recent))

    if PERSIST_TO_DB and DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as con:
            inv.to_sql("invoices", con, if_exists="replace", index=False)
            bank.to_sql("bank_tx", con, if_exists="replace", index=False)

# ---------------------------------
# Post-recon KPIs + Charts (reactive to sliders)
# ---------------------------------
# ---------------------------------
# Post-recon aggregates (used by all views)
# ---------------------------------
revexp = pd.DataFrame()
cash = pd.DataFrame()
re_ent = pd.DataFrame()
cash_ent = pd.DataFrame()

if not inv.empty and "entity" in inv.columns:
    # Accrual P&L by entity/month
    if {"type", "amount", "month"}.issubset(inv.columns):
        revexp = (
            inv.assign(
                revenue=np.where(inv["type"].eq("revenue"), inv["amount"], 0.0),
                expense=np.where(inv["type"].eq("expense"), inv["amount"], 0.0),
            )
            .groupby(["entity", "month"], as_index=False)[["revenue", "expense"]]
            .sum()
        )
        revexp_all = (
            revexp.groupby("month", as_index=False)[["revenue", "expense"]]
            .sum()
            .assign(entity="ALL")
        )
        revexp = pd.concat([revexp, revexp_all], ignore_index=True)

    # Cash from bank
    if {"direction", "amount", "month"}.issubset(bank.columns):
        bank2 = bank.copy()
        bank2["inflow"] = np.where(bank2["direction"].eq("in"), bank2["amount"], 0.0)
        bank2["outflow"] = np.where(bank2["direction"].eq("out"), bank2["amount"], 0.0)
        cash = (
            bank2.groupby(["entity", "month"], as_index=False)[["inflow", "outflow"]]
            .sum()
            .assign(net_cash=lambda d: d["inflow"] - d["outflow"])
        )
        cash_all = (
            cash.groupby("month", as_index=False)[["inflow", "outflow", "net_cash"]]
            .sum()
            .assign(entity="ALL")
        )
        cash = pd.concat([cash, cash_all], ignore_index=True)

    matched_inv = inv[(inv.get("type") == "revenue") & (inv["match_id"].notna())].copy()
    unmatched_inv = inv[(inv.get("type") == "revenue") & (inv["match_id"].isna())].copy()

    if not revexp.empty:
        re_ent = revexp[revexp["entity"].eq(ent)].sort_values("month")
    if not cash.empty:
        cash_ent = cash[cash["entity"].eq(ent)].sort_values("month")

    # VAT + currency info
    if "vat_amount" in inv.columns:
        vat_total = float(inv.get("vat_amount", 0).sum())
    else:
        vat_total = 0.0
    currencies = ", ".join(
        sorted(
            inv.get("currency", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    # Classic KPIs
    last_rev = re_ent["revenue"].iloc[-1] if not re_ent.empty else 0.0
    last_exp = re_ent["expense"].iloc[-1] if not re_ent.empty else 0.0
    gross_prof = last_rev - last_exp

    cash_balance = cash_ent["net_cash"].cumsum().iloc[-1] if not cash_ent.empty else 0.0
    prev_burn = (
        abs(cash_ent["net_cash"].shift(1).iloc[-1])
        if len(cash_ent) > 1 and cash_ent["net_cash"].shift(1).iloc[-1] < 0
        else 0.0
    )
    runway_months = (cash_balance / max(1.0, prev_burn)) if prev_burn > 0 else np.nan

    # Matched/unmatched KPIs (slider-sensitive)
    matched_amt = float(matched_inv["amount"].sum()) if not matched_inv.empty else 0.0
    unmatched_amt = float(unmatched_inv["amount"].sum()) if not unmatched_inv.empty else 0.0
    matched_cnt = int(matched_inv.shape[0])
    unmatched_cnt = int(unmatched_inv.shape[0])

    # Overall collection rate over the whole horizon
    total_revenue = float(re_ent["revenue"].sum()) if not re_ent.empty else 0.0
    collection_rate = matched_amt / total_revenue if total_revenue > 0 else 0.0
else:
    matched_inv = pd.DataFrame()
    unmatched_inv = pd.DataFrame()
    vat_total = 0.0
    currencies = ""
    last_rev = last_exp = gross_prof = 0.0
    cash_balance = prev_burn = 0.0
    runway_months = np.nan
    matched_amt = unmatched_amt = 0.0
    matched_cnt = unmatched_cnt = 0
    collection_rate = 0.0

# ---------------------------------
# Exception datasets (used in Raw data)
# ---------------------------------
if {"type", "match_id"}.issubset(inv.columns):
    q_inv_unmatched = inv.query("type=='revenue' and match_id.isna()")
else:
    q_inv_unmatched = pd.DataFrame()

if {"direction", "match_id"}.issubset(bank.columns):
    q_bank_unmatched = bank.query("direction=='in' and match_id.isna()")
else:
    q_bank_unmatched = pd.DataFrame()

if "status" in bank.columns:
    q_partial = bank[bank["status"].fillna("").str.contains("fee|batch|Partial", case=False)]
else:
    q_partial = pd.DataFrame()


# ---------------------------------
# INSIGHTS: REPORTING
# ---------------------------------
if view_mode == "Reporting":
    # Top KPI band
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matched invoices (#)", f"{matched_cnt:,}")
    k2.metric("Matched €", f"{matched_amt:,.0f}")
    k3.metric("Unmatched € (AR)", f"{unmatched_amt:,.0f}")
    k4.metric("Runway (months)", "-" if np.isnan(runway_months) else f"{runway_months:.1f}")

    k5, k6 = st.columns(2)
    k5.metric("Total VAT in dataset", f"{vat_total:,.0f} EUR")
    k6.caption(f"Currencies in data: {currencies or '–'}")

    # Accrued vs collected revenue
    st.markdown("### Accrued vs Collected Revenue (monthly)")

    accr_m = re_ent[["month", "revenue"]].set_index("month") if not re_ent.empty else pd.DataFrame()

    if not matched_inv.empty:
        if "month" not in matched_inv.columns:
            matched_inv = matched_inv.assign(
                month=pd.to_datetime(matched_inv["date"]).dt.to_period("M").dt.to_timestamp()
            )
        matched_m = (
            matched_inv.groupby("month", as_index=True)["amount"]
            .sum()
            .to_frame("matched_revenue")
        )
    else:
        matched_m = pd.DataFrame(columns=["matched_revenue"])

    both = accr_m.join(matched_m, how="outer").fillna(0.0)

    if not both.empty:
        df_chart = both.reset_index()
        df_chart = df_chart.melt(
            id_vars=["month"],
            value_vars=["revenue", "matched_revenue"],
            var_name="metric",
            value_name="amount",
        )

        hover = alt.selection_single(
            fields=["month"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        base = alt.Chart(df_chart).encode(
            x=alt.X("month:T", axis=alt.Axis(title="Month")),
            y=alt.Y("amount:Q", axis=alt.Axis(title="EUR")),
            color=alt.Color(
                "metric:N",
                title="",
                scale=alt.Scale(scheme="tableau10"),
                legend=alt.Legend(title=""),
            ),
        )

        area = base.mark_area(opacity=0.4)

        points = base.mark_circle(size=50).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0))
        ).add_selection(hover)

        tooltips = base.mark_rule(color="gray").encode(
            tooltip=[
                alt.Tooltip("month:T", title="Month"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("amount:Q", title="Amount", format=",.0f"),
            ]
        ).transform_filter(hover)

        chart1 = alt.layer(area, points, tooltips).properties(height=260)

        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info("No revenue data for selected entity.")

    # Net revenue & VAT
    st.markdown("### Net Revenue & VAT (monthly)")
    if {"month", "net_amount", "vat_amount"}.issubset(inv.columns):
        rev_vat = (
            inv.groupby("month", as_index=False)[["net_amount", "vat_amount"]]
            .sum()
            .rename(columns={"net_amount": "Net revenue", "vat_amount": "VAT"})
        )

        df_vat = rev_vat.melt(
            id_vars=["month"],
            value_vars=["Net revenue", "VAT"],
            var_name="metric",
            value_name="amount",
        )

        hover2 = alt.selection_single(
            fields=["month"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        base2 = alt.Chart(df_vat).encode(
            x=alt.X("month:T", axis=alt.Axis(title="Month")),
            y=alt.Y("amount:Q", axis=alt.Axis(title="EUR")),
            color=alt.Color(
                "metric:N",
                title="",
                scale=alt.Scale(scheme="set2"),
                legend=alt.Legend(title=""),
            ),
        )

        area2 = base2.mark_area(opacity=0.4)

        points2 = base2.mark_circle(size=50).encode(
            opacity=alt.condition(hover2, alt.value(1), alt.value(0))
        ).add_selection(hover2)

        tooltips2 = base2.mark_rule(color="gray").encode(
            tooltip=[
                alt.Tooltip("month:T", title="Month"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("amount:Q", title="Amount", format=",.0f"),
            ]
        ).transform_filter(hover2)

        chart2 = alt.layer(area2, points2, tooltips2).properties(height=260)

        st.altair_chart(chart2, use_container_width=True)
    # Geen blauwe warning meer als de kolommen ontbreken

# ---------------------------------
# INSIGHTS: METRICS
# ---------------------------------
elif view_mode == "Metrics":
    st.subheader("Revenue & expense by month")
    if not re_ent.empty:
        st.dataframe(re_ent)
    else:
        st.info("No revenue/expense data for this entity.")

    st.subheader("Cash movement by month")
    if not cash_ent.empty:
        st.dataframe(cash_ent)
    else:
        st.info("No cash data for this entity.")

    st.subheader("Top 5 open AR invoices")
    if not unmatched_inv.empty:
        top_ar = unmatched_inv.sort_values("amount", ascending=False).head(5)
        st.dataframe(top_ar[["date", "partner", "amount", "invoice_no"]])
    else:
        st.caption("No unmatched revenue invoices — nice.")

# ---------------------------------
# INSIGHTS: RAW DATA
# ---------------------------------
else:
    # Small KPIs at the top of raw view
    c1, c2, c3 = st.columns(3)
    c1.metric("Unmatched invoices", len(q_inv_unmatched))
    c2.metric("Unmatched bank entries", len(q_bank_unmatched))
    c3.metric("PSP / batch matches", len(q_partial))

    st.markdown("### Exception detail")
    view = st.selectbox(
        "Exception view",
        ["Unmatched invoices", "Unmatched bank", "PSP / batch matches", "All raw data"],
        index=0,
    )

    if view == "Unmatched invoices":
        st.dataframe(q_inv_unmatched.sort_values("date"))
    elif view == "Unmatched bank":
        st.dataframe(q_bank_unmatched.sort_values("date"))
    elif view == "PSP / batch matches":
        st.dataframe(q_partial.sort_values("date"))
    else:
        st.dataframe(inv)

    # Journal preview & Board Pack
    st.markdown("### Journal preview")

    COA = {
        "Revenue": "4000-Revenue",
        "Cash": "1000-Cash",
        "AR": "1200-Accounts Receivable",
        "PSP Fees": "6060-Payment Processing Fees",
    }

    journal = []

    def matched_bank_amount_and_fee(inv_row):
        mid = inv_row["match_id"]
        if pd.isna(mid):
            return None, 0.0
        b = bank.loc[bank["match_id"] == mid]
        if b.empty:
            return None, 0.0
        bank_amt = float(b.iloc[0]["amount"])
        is_fee_context = "fee" in str(b.iloc[0]["status"]).lower() or "fee" in str(inv_row["status"]).lower()
        fee = max(0.0, float(inv_row["amount"]) - bank_amt) if is_fee_context else 0.0
        return bank_amt, fee

    # Matched invoices -> Dr Cash + (optioneel) Dr Fees, Cr Revenue
    for _, r in inv.dropna(subset=["match_id"]).iterrows():
        bank_amt, fee = matched_bank_amount_and_fee(r)
        if bank_amt is None:
            journal += [
                dict(
                    date=r["date"],
                    entity=r.get("entity", ""),
                    account=COA["AR"],
                    debit=float(r["amount"]),
                    credit=0.0,
                    ref="UNRESOLVED",
                ),
                dict(
                    date=r["date"],
                    entity=r.get("entity", ""),
                    account=COA["Revenue"],
                    debit=0.0,
                    credit=float(r["amount"]),
                    ref="UNRESOLVED",
                ),
            ]
            continue
        journal += [
            dict(
                date=r["date"],
                entity=r.get("entity", ""),
                account=COA["Cash"],
                debit=float(bank_amt),
                credit=0.0,
                ref=r["match_id"],
            ),
            dict(
                date=r["date"],
                entity=r.get("entity", ""),
                account=COA["Revenue"],
                debit=0.0,
                credit=float(r["amount"]),
                ref=r["match_id"],
            ),
        ]
        if fee > 0.0001:
            journal += [
                dict(
                    date=r["date"],
                    entity=r.get("entity", ""),
                    account=COA["PSP Fees"],
                    debit=float(fee),
                    credit=0.0,
                    ref=r["match_id"],
                )
            ]

    # Unmatched invoices -> Dr AR, Cr Revenue
    for _, r in inv[(inv.get("type") == "revenue") & (inv["match_id"].isna())].iterrows():
        journal += [
            dict(
                date=r["date"],
                entity=r.get("entity", ""),
                account=COA["AR"],
                debit=float(r["amount"]),
                credit=0.0,
                ref="UNMATCHED",
            ),
            dict(
                date=r["date"],
                entity=r.get("entity", ""),
                account=COA["Revenue"],
                debit=0.0,
                credit=float(r["amount"]),
                ref="UNMATCHED",
            ),
        ]

    jdf = pd.DataFrame(journal)
    st.dataframe(jdf)

    # Build Board Pack ZIP (journal + P&L + cash + raw tables)
    csv_journal = jdf.to_csv(index=False).encode()
    if not revexp.empty:
        pnl_df = revexp.groupby("month", as_index=False)[["revenue", "expense"]].sum()
        csv_pl = pnl_df.to_csv(index=False).encode()
    else:
        csv_pl = b""

    if not cash.empty:
        cash_df = cash.groupby("month", as_index=False)[["net_cash"]].sum()
        csv_cash = cash_df.to_csv(index=False).encode()
    else:
        csv_cash = b""

    left_df = inv
    right_df = bank

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("journal.csv", csv_journal)
        z.writestr("pl_monthly.csv", csv_pl)
        z.writestr("cash_monthly.csv", csv_cash)
        z.writestr("invoices_raw.csv", left_df.to_csv(index=False))
        z.writestr("bank_raw.csv", right_df.to_csv(index=False))

    st.download_button(
        "Download Board Pack (ZIP)",
        data=buf.getvalue(),
        file_name="board_pack.zip",
        mime="application/zip",
    )

    # Raw tables at the very bottom
    st.markdown("### Raw data (for debugging)")
    tab1, tab2 = st.tabs(["Invoices (raw)", "Bank (raw)"])
    with tab1:
        st.dataframe(inv if (ent == "ALL" or inv.empty or "entity" not in inv.columns) else inv[inv["entity"].eq(ent)])
    with tab2:
        st.dataframe(bank if (ent == "ALL" or bank.empty or "entity" not in bank.columns) else bank[bank["entity"].eq(ent)])
