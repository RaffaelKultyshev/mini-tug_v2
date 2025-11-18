# api.py — FastAPI backend for Mini-TUG

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core import get_kpis  # gebruikt dezelfde mini_tug.db als streamlit / core

app = FastAPI(
    title="Mini-TUG API",
    version="0.1.0",
    description="Backend API for Mini-TUG prototype",
)

# Tijdens dev / eerste deploy: we staan requests toe vanaf alle origins
# Later kun je dit beperken tot je Vercel-domein.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # voor nu: alles toestaan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/kpi")
def read_kpis():
    """
    Simple KPI endpoint voor de frontend.
    Haalt basisstatistieken op uit mini_tug.db via core.get_kpis().
    """
    return get_kpis()
