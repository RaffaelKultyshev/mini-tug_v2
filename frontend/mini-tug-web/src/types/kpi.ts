import { KpiResponse } from "@/types/kpi";

async function getKpis(): Promise<KpiResponse> {
  const res = await fetch("http://localhost:8000/kpi", {
    next: { revalidate: 0 },
  });

  if (!res.ok) {
    throw new Error("Failed to fetch KPIs");
  }

  return res.json();
}

export default async function HomePage() {
  const data = await getKpis();

  return (
    <main className="mx-auto max-w-5xl py-10">
      <h1 className="text-3xl font-semibold mb-6">
        Mini_TUG (v2.0) – Finance Overview
      </h1>

      <div className="grid grid-cols-4 gap-4">
        <KpiCard label="Invoices" value={data.invoices_count} />
        <KpiCard label="Bank transactions" value={data.bank_count} />
        <KpiCard
          label="Total revenue (€)"
          value={data.total_revenue.toFixed(0)}
        />
        <KpiCard
          label="Collection rate"
          value={`${data.collection_rate.toFixed(1)}%`}
        />
      </div>

      <p className="mt-6 text-sm text-gray-500">
        Backend: http://localhost:8000/kpi — rendered server-side in Next.js.
      </p>
    </main>
  );
}

type KpiCardProps = {
  label: string;
  value: string | number;
};

function KpiCard({ label, value }: KpiCardProps) {
  return (
    <div className="rounded-xl border px-6 py-4 shadow-sm">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-2xl font-semibold mt-2">{value}</div>
    </div>
  );
}
