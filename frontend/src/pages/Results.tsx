import { useState } from "react";

import { api, type MetricRow, type MetricsResponse } from "../api";
import CiBarChart, { type CiValue } from "../components/CiBarChart";
import { METRIC_NAMES, MODEL_NAMES, MODEL_ORDER, SERIES_COLORS, type MetricKey, pct } from "../format";
import { useAsync, useIsDark } from "../hooks";

const TOPICS = ["abuse", "anxiety", "money", "ptsd", "relationships"];

function ci(row: MetricRow, metric: MetricKey): CiValue {
  return { value: row[metric], low: row[`${metric}_low`], high: row[`${metric}_high`], n: row.n };
}

function find(rows: MetricRow[], model: string, setting: string, testSet: string) {
  return rows.find((r) => r.model === model && r.setting === setting && r.test_set === testSet);
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="card p-5">
      <p className="text-sm text-ink-3">{label}</p>
      <p className="mt-1 text-3xl font-semibold tabular-nums">{value}</p>
      {note && <p className="mt-1 text-xs text-ink-3">{note}</p>}
    </div>
  );
}

function ModelComparison({ rows, colors }: { rows: MetricRow[]; colors: string[] }) {
  const models = MODEL_ORDER.filter((m) => find(rows, m, "standard", "all"));
  const metrics = Object.keys(METRIC_NAMES) as MetricKey[];
  const values = Object.fromEntries(
    metrics.map((metric) => [
      metric,
      Object.fromEntries(models.map((m) => [m, ci(find(rows, m, "standard", "all")!, metric)])),
    ]),
  );
  return (
    <CiBarChart
      title="Model comparison"
      description="All 715 test posts. Bars show the score; whiskers show the 95% bootstrap confidence interval (1,000 resamples)."
      categories={metrics.map((m) => ({ key: m, name: METRIC_NAMES[m] }))}
      series={models.map((m) => ({ key: m, name: MODEL_NAMES[m], color: colors[MODEL_ORDER.indexOf(m)] }))}
      values={values}
    />
  );
}

function NewTopic({ rows, colors }: { rows: MetricRow[]; colors: string[] }) {
  const models = MODEL_ORDER.filter((m) => m !== "majority" && rows.some((r) => r.model === m && r.setting === "new_topic"));
  const [model, setModel] = useState(models[models.length - 1]);
  const values = Object.fromEntries(
    TOPICS.map((t) => {
      const seen = find(rows, model, "in_topic", t);
      const held = find(rows, model, "new_topic", t);
      return [t, { in_topic: seen && ci(seen, "f1"), new_topic: held && ci(held, "f1") }];
    }),
  );
  return (
    <div className="space-y-3">
      {models.length > 1 && (
        <div className="flex gap-2" role="group" aria-label="Choose model">
          {models.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setModel(m)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium ${
                m === model ? "bg-accent text-accent-ink" : "border border-border text-ink-2 hover:bg-surface-2"
              }`}
            >
              {MODEL_NAMES[m]}
            </button>
          ))}
        </div>
      )}
      <CiBarChart
        title={`New-topic test · ${MODEL_NAMES[model]}`}
        description="Stress F1 on each topic's test posts: when the topic was seen in training (standard model) vs. held out entirely (trained on the other 4 topics)."
        categories={TOPICS.map((t) => ({ key: t, name: t }))}
        series={[
          { key: "in_topic", name: "Seen topic", color: colors[0] },
          { key: "new_topic", name: "New topic (held out)", color: colors[1] },
        ]}
        values={values}
      />
    </div>
  );
}

function AppChecks({ data }: { data: MetricsResponse }) {
  const { api_check, latency } = data;
  if (!api_check && !latency) return null;
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">App checks</h2>
      <div className="grid gap-4 sm:grid-cols-3">
        {api_check && (
          <Stat
            label="API matches offline model"
            value={pct(api_check.label_match_pct / 100, 1)}
            note={`${api_check.n_label_match} of ${api_check.n_posts} test posts · ${MODEL_NAMES[api_check.model_name] ?? api_check.model_name}`}
          />
        )}
        {latency && (
          <>
            <Stat label="Mean prediction time" value={`${latency.server_mean_ms.toFixed(1)} ms`}
                  note={`Inside the API, ${latency.n_requests} requests (goal: under 1,000 ms)`} />
            <Stat label="95th percentile" value={`${latency.server_p95_ms.toFixed(1)} ms`}
                  note={`Full round trip: mean ${latency.client_mean_ms.toFixed(1)} ms, p95 ${latency.client_p95_ms.toFixed(1)} ms`} />
          </>
        )}
      </div>
    </section>
  );
}

function Errors({ data }: { data: MetricsResponse }) {
  const [open, setOpen] = useState(false);
  if (!data.errors?.length) return null;
  const shown = open ? data.errors : data.errors.slice(0, 5);
  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Where the model is wrong</h2>
        <p className="mt-1 text-sm text-ink-3">
          {data.errors.length} misclassified test posts (random sample, seed 42), saved in results/errors.csv for manual
          grouping.
        </p>
      </div>
      <div className="card divide-y divide-border">
        {shown.map((e) => (
          <article key={e.id} className="space-y-2 p-4 text-sm">
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="rounded-full bg-surface-2 px-2 py-0.5">{e.topic}</span>
              <span className="rounded-full bg-stress-soft px-2 py-0.5 text-stress">{e.error_type}</span>
              <span className="text-ink-3">
                true: {e.true_label} · predicted: {e.predicted_label} ({pct(e.confidence)})
              </span>
              {e.group && <span className="rounded-full bg-accent-soft px-2 py-0.5 text-accent">{e.group}</span>}
            </div>
            <p className="leading-relaxed text-ink-2">{e.text}</p>
          </article>
        ))}
      </div>
      {data.errors.length > 5 && (
        <button type="button" onClick={() => setOpen((o) => !o)} className="text-sm font-medium text-accent">
          {open ? "Show fewer" : `Show all ${data.errors.length}`}
        </button>
      )}
    </section>
  );
}

export default function Results() {
  const state = useAsync(api.metrics);
  const colors = SERIES_COLORS[useIsDark() ? "dark" : "light"];

  if (state.status === "loading") return <p className="text-ink-3">Loading results…</p>;
  if (state.status === "error") return <p className="text-stress">{state.error}</p>;
  const { metrics } = state.data;
  if (!metrics.length) return <p className="text-ink-3">No results yet. Run the evaluation scripts first.</p>;

  const hasRoberta = metrics.some((r) => r.model === "roberta");
  return (
    <div className="space-y-10">
      <header className="max-w-2xl">
        <h1 className="text-3xl font-semibold tracking-tight">Results</h1>
        <p className="mt-3 text-ink-2">
          Every number on this page was produced by the evaluation code and loaded from <code>results/</code>.
          F1, precision and recall are for the <em>stress</em> class.
        </p>
        {!hasRoberta && (
          <p className="mt-3 rounded-lg bg-surface-2 px-3 py-2 text-sm text-ink-3">
            RoBERTa results will appear here after the Colab notebook has been run and evaluated.
          </p>
        )}
      </header>
      <ModelComparison rows={metrics} colors={colors} />
      <NewTopic rows={metrics} colors={colors} />
      <AppChecks data={state.data} />
      <Errors data={state.data} />
    </div>
  );
}
