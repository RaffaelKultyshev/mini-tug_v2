// src/app/page.tsx

type KpiResponse = {
  invoice_count: number;
  bank_tx_count: number;
  total_revenue: number;
  matched_revenue: number;
  collection_rate: number;
};

async function getKpi(): Promise<KpiResponse> {
  const res = await fetch("http://localhost:8000/kpi", {
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error("Failed to fetch KPI from backend");
  }

  return res.json();
}

export default async function Home() {
  const kpi = await getKpi();

  return (
    <main className="min-h-screen bg-slate-50 flex flex-col items-center py-10">
      <div className="w-full max-w-5xl px-4">
        <h1 className="text-3xl font-semibold text-slate-900 mb-6">
          Mini_TUG (v2.0) – Finance Overview
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="rounded-2xl bg-white shadow p-4">
            <p className="text-xs text-slate-500">Invoices</p>
            <p className="text-2xl font-semibold">{kpi.invoice_count}</p>
          </div>
          <div className="rounded-2xl bg-white shadow p-4">
            <p className="text-xs text-slate-500">Bank transactions</p>
            <p className="text-2xl font-semibold">{kpi.bank_tx_count}</p>
          </div>
          <div className="rounded-2xl bg-white shadow p-4">
            <p className="text-xs text-slate-500">Total revenue (€)</p>
            <p className="text-2xl font-semibold">
              {kpi.total_revenue.toLocaleString("en-US", {
                maximumFractionDigits: 0,
              })}
            </p>
          </div>
          <div className="rounded-2xl bg-white shadow p-4">
            <p className="text-xs text-slate-500">Collection rate</p>
            <p className="text-2xl font-semibold">
              {(kpi.collection_rate * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <p className="text-sm text-slate-500">
          Backend: <code>http://localhost:8000/kpi</code> — rendered
          server-side in Next.js.
        </p>
      </div>
    </main>
  );
}
