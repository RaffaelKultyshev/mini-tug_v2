from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core import get_kpis  # gebruikt dezelfde DB als streamlit

app = FastAPI()

origins = [
    "http://localhost:3000",  # Next.js dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/kpi")
def read_kpis():
    """
    Endpoint dat de basis-KPIs teruggeeft voor de Next.js frontend.
    """
    return get_kpis()
