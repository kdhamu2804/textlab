import { CloudSun, Gauge, Timer, Waves } from "lucide-react";

import type { PredictResponse } from "../api";
import { MODEL_NAMES, pct } from "../format";
import CrisisNote from "./CrisisNote";

/** Horizontal 0–100% scale of P(stress) with the 50% decision threshold marked. */
function ProbabilityScale({ p }: { p: number }) {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between text-sm">
        <span className="text-ink-2">Probability of stress</span>
        <span className="text-lg font-semibold tabular-nums">{pct(p, 1)}</span>
      </div>
      <div
        className="relative h-3 rounded-full bg-gradient-to-r from-calm/70 via-surface-2 to-stress/80"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(p * 100)}
        aria-label="Probability of stress"
      >
        <span className="absolute inset-y-[-3px] left-1/2 w-px bg-ink-3" aria-hidden />
        <span
          className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-surface bg-ink shadow-md transition-[left] duration-500"
          style={{ left: `${p * 100}%` }}
          aria-hidden
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] text-ink-3">
        <span>no stress</span>
        <span>threshold 50%</span>
        <span>stress</span>
      </div>
    </div>
  );
}

/** Predicted label, calibrated probability, model and response time. */
export default function ResultCard({ result }: { result: PredictResponse }) {
  const stress = result.label === "stress";
  const Icon = stress ? Waves : CloudSun;
  const tone = stress ? "text-stress bg-stress-soft" : "text-calm bg-calm-soft";

  return (
    <section className="card flex flex-col gap-6 p-6" aria-live="polite">
      <div className="flex items-center gap-4">
        <span className={`grid h-14 w-14 shrink-0 place-items-center rounded-2xl ${tone}`}>
          <Icon size={28} aria-hidden />
        </span>
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-ink-3">Prediction</p>
          <p className={`text-3xl font-semibold capitalize ${stress ? "text-stress" : "text-calm"}`}>{result.label}</p>
          <p className="text-sm text-ink-3">{pct(result.confidence)} confident</p>
        </div>
      </div>

      <ProbabilityScale p={result.stress_probability} />

      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-xl bg-surface-2 px-3 py-2.5">
          <dt className="flex items-center gap-1.5 text-ink-3"><Gauge size={14} aria-hidden /> Model</dt>
          <dd className="mt-0.5 font-medium">{MODEL_NAMES[result.model_name] ?? result.model_name}</dd>
        </div>
        <div className="rounded-xl bg-surface-2 px-3 py-2.5">
          <dt className="flex items-center gap-1.5 text-ink-3"><Timer size={14} aria-hidden /> Response time</dt>
          <dd className="mt-0.5 font-medium tabular-nums">{result.latency_ms.toFixed(0)} ms</dd>
        </div>
      </dl>

      <CrisisNote />
    </section>
  );
}
