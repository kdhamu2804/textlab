/** Result sections built from scripts.analyze output: significance, confusion matrices, calibration. */
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { AgreementRow, CalibrationBin, CalibrationSummary, ConfusionRow, SignificanceRow, Temperature } from "../api";
import { MODEL_NAMES, MODEL_ORDER, pct, points, pValue, score } from "../format";
import { useIsDark } from "../hooks";

export function SectionHeading({ id, title, children }: { id: string; title: string; children?: React.ReactNode }) {
  return (
    <div id={id} className="scroll-mt-24">
      <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      {children && <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-ink-3">{children}</p>}
    </div>
  );
}

/** Paired bootstrap + McNemar for each (baseline, challenger) pair. */
export function Significance({ rows }: { rows: SignificanceRow[] }) {
  return (
    <div className="card overflow-x-auto">
      <table className="w-full min-w-[40rem] text-sm">
        <thead>
          <tr className="border-b border-border text-left text-ink-3">
            <th className="px-5 py-3 font-medium">Comparison</th>
            <th className="px-5 py-3 font-medium">Δ F1 (stress)</th>
            <th className="px-5 py-3 font-medium">95% CI of Δ</th>
            <th className="px-5 py-3 font-medium">Paired bootstrap</th>
            <th className="px-5 py-3 font-medium">McNemar</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.model_a}-${r.model_b}`} className="border-b border-border last:border-0">
              <td className="px-5 py-3 font-medium">
                {MODEL_NAMES[r.model_b]} <span className="text-ink-3">vs</span> {MODEL_NAMES[r.model_a]}
              </td>
              <td className="px-5 py-3 font-semibold tabular-nums text-calm">{points(r.difference)} pts</td>
              <td className="px-5 py-3 tabular-nums text-ink-2">
                [{points(r.difference_low)}, {points(r.difference_high)}]
              </td>
              <td className="px-5 py-3 tabular-nums">{pValue(r.p_value)}</td>
              <td className="px-5 py-3 tabular-nums">
                {pValue(r.mcnemar_p)}
                <span className="block text-xs text-ink-3">
                  only {MODEL_NAMES[r.model_b]} right: {r.only_b_correct} · only {MODEL_NAMES[r.model_a]} right:{" "}
                  {r.only_a_correct}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One 2×2 confusion matrix per model, cells shaded by row share. */
export function ConfusionMatrices({ rows }: { rows: ConfusionRow[] }) {
  const ordered = MODEL_ORDER.map((m) => rows.find((r) => r.model === m)).filter(Boolean) as ConfusionRow[];
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {ordered.map((r) => {
        const cells = [
          [{ n: r.tn, label: "True negative", ok: true }, { n: r.fp, label: "False positive", ok: false }],
          [{ n: r.fn, label: "False negative", ok: false }, { n: r.tp, label: "True positive", ok: true }],
        ];
        return (
          <figure key={r.model} className="card p-5">
            <figcaption className="mb-3 font-semibold">{MODEL_NAMES[r.model]}</figcaption>
            <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-xs">
              <span />
              <span className="pb-1 text-center text-ink-3">pred. no stress</span>
              <span className="pb-1 text-center text-ink-3">pred. stress</span>
              {cells.map((row, i) => (
                <div key={i} className="contents">
                  <span className="flex items-center pr-1 text-ink-3">{i === 0 ? "no stress" : "stress"}</span>
                  {row.map((c) => {
                    const share = c.n / (row[0].n + row[1].n || 1);
                    const color = c.ok ? "var(--calm)" : "var(--stress)";
                    return (
                      <div
                        key={c.label}
                        className="rounded-lg px-2 py-3 text-center"
                        style={{ background: `color-mix(in srgb, ${color} ${Math.round(8 + share * 45)}%, transparent)` }}
                        title={c.label}
                      >
                        <span className="block text-lg font-semibold tabular-nums text-ink">{c.n}</span>
                        <span className="text-ink-2">{pct(share)}</span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-ink-3">Rows: true label. Percentages are shares of each row.</p>
          </figure>
        );
      })}
    </div>
  );
}

const CALIBRATION_COLORS: Record<string, { light: string; dark: string }> = {
  tfidf: { light: "#eb6834", dark: "#d95926" },
  roberta: { light: "#1baf7a", dark: "#199e70" },
};

/** Reliability diagram plus ECE / Brier table, and the temperature-scaling effect. */
export function Calibration({
  bins,
  summary,
  temperature,
}: {
  bins: CalibrationBin[];
  summary: CalibrationSummary[];
  temperature: Temperature | null;
}) {
  const dark = useIsDark();
  const ink = dark ? "#b9c0bb" : "#4d5550";
  const grid = dark ? "#2f3632" : "#e3e1d9";
  const models = MODEL_ORDER.filter((m) => summary.some((s) => s.model === m));
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <figure className="card p-5 sm:p-6">
        <figcaption className="mb-1 font-semibold">Reliability diagram</figcaption>
        <p className="mb-4 text-sm text-ink-3">
          Posts are grouped by predicted probability. A perfectly calibrated model lies on the dashed diagonal.
        </p>
        <div className="h-72 w-full">
          <ResponsiveContainer>
            <LineChart margin={{ top: 8, right: 12, left: -12, bottom: 8 }}>
              <CartesianGrid stroke={grid} />
              <XAxis type="number" dataKey="mean_predicted" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]}
                     tick={{ fill: ink, fontSize: 12 }} tickLine={false} axisLine={{ stroke: grid }}
                     label={{ value: "Predicted P(stress)", position: "insideBottom", offset: -4, fill: ink, fontSize: 12 }} />
              <YAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fill: ink, fontSize: 12 }}
                     tickLine={false} axisLine={false} />
              <Tooltip
                formatter={(v) => (typeof v === "number" ? score(v) : String(v))}
                labelFormatter={(v) => (typeof v === "number" ? `predicted ${score(v)}` : "")}
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
              <Line data={[{ mean_predicted: 0, observed_rate: 0 }, { mean_predicted: 1, observed_rate: 1 }]}
                    dataKey="observed_rate" name="Perfect calibration" stroke={ink} strokeDasharray="5 5"
                    dot={false} isAnimationActive={false} />
              {models.map((m) => (
                <Line key={m} data={bins.filter((b) => b.model === m)} dataKey="observed_rate" name={MODEL_NAMES[m]}
                      stroke={CALIBRATION_COLORS[m]?.[dark ? "dark" : "light"] ?? ink} strokeWidth={2}
                      dot={{ r: 3 }} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </figure>
      <div className="space-y-4">
        <div className="card p-5">
          <p className="mb-3 font-semibold">Calibration error (lower is better)</p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-ink-3">
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 font-medium">ECE</th>
                <th className="pb-2 font-medium">Brier</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const s = summary.find((x) => x.model === m)!;
                return (
                  <tr key={m} className="border-t border-border">
                    <td className="py-2 font-medium">{MODEL_NAMES[m]}</td>
                    <td className="py-2 tabular-nums">{score(s.ece)}</td>
                    <td className="py-2 tabular-nums">{score(s.brier)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {temperature && (
          <div className="card space-y-2 p-5 text-sm">
            <p className="font-semibold">Temperature scaling (RoBERTa)</p>
            <p className="text-ink-2">
              Raw RoBERTa probabilities were over-confident. A single temperature{" "}
              <strong className="tabular-nums">T = {temperature.temperature.toFixed(2)}</strong>, fitted on the{" "}
              {temperature.n_val} validation posts only, softens them without changing any label.
            </p>
            <p className="tabular-nums text-ink-2">
              Test ECE: {score(temperature.test_ece_before)} → <strong className="text-calm">{score(temperature.test_ece_after)}</strong>
              <span className="text-ink-3"> · validation NLL {score(temperature.val_nll_before)} → {score(temperature.val_nll_after)}</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

const RATERS: Record<string, string> = {
  dataset_label: "Dataset label",
  dhamodhar: "Dhamodhar",
  kushvanth: "Kushvanth",
  uday: "Uday",
};

/** Agreement of each team member with the dataset labels and with each other. */
export function LabelAgreement({ rows }: { rows: AgreementRow[] | null }) {
  if (!rows?.length) {
    return (
      <div className="card flex flex-wrap items-center justify-between gap-3 p-5 text-sm text-ink-3">
        <span>Pending: each team member labels the same 50 training posts. This section fills in as they do.</span>
        <a href="/label" className="rounded-lg bg-accent px-3 py-1.5 font-medium text-accent-ink hover:opacity-90">
          Open the labelling page
        </a>
      </div>
    );
  }
  return (
    <div className="card overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-ink-3">
            <th className="px-5 py-3 font-medium">Pair</th>
            <th className="px-5 py-3 font-medium">Posts</th>
            <th className="px-5 py-3 font-medium">Agreement</th>
            <th className="px-5 py-3 font-medium">Cohen's κ</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.rater_a}-${r.rater_b}`} className="border-b border-border last:border-0">
              <td className="px-5 py-3">
                {RATERS[String(r.rater_a)] ?? r.rater_a} <span className="text-ink-3">vs</span>{" "}
                {RATERS[String(r.rater_b)] ?? r.rater_b}
              </td>
              <td className="px-5 py-3 tabular-nums">{r.n}</td>
              <td className="px-5 py-3 tabular-nums">
                {r.agreement_pct === null ? "—" : `${Number(r.agreement_pct).toFixed(0)}%`}
              </td>
              <td className="px-5 py-3 tabular-nums">{r.cohen_kappa === null ? "—" : Number(r.cohen_kappa).toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
