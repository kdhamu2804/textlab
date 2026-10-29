import { CloudSun, Waves } from "lucide-react";

import type { PredictResponse } from "../api";
import { MODEL_NAMES, pct } from "../format";
import CrisisNote from "./CrisisNote";

/** Shows the predicted label, a confidence bar and the model that was used. */
export default function ResultCard({ result }: { result: PredictResponse }) {
  const stress = result.label === "stress";
  const Icon = stress ? Waves : CloudSun;
  const tone = stress ? "text-stress bg-stress-soft" : "text-calm bg-calm-soft";
  const bar = stress ? "bg-stress" : "bg-calm";

  return (
    <section className="card space-y-5 p-6" aria-live="polite">
      <div className="flex items-center gap-4">
        <span className={`grid h-12 w-12 place-items-center rounded-xl ${tone}`}>
          <Icon size={24} aria-hidden />
        </span>
        <div>
          <p className="text-sm text-ink-3">Prediction</p>
          <p className="text-2xl font-semibold capitalize">{result.label}</p>
        </div>
      </div>

      <div>
        <div className="mb-1.5 flex justify-between text-sm">
          <span className="text-ink-2">Confidence</span>
          <span className="font-medium tabular-nums">{pct(result.confidence)}</span>
        </div>
        <div
          className="h-2.5 overflow-hidden rounded-full bg-surface-2"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(result.confidence * 100)}
          aria-label="Model confidence"
        >
          <div className={`h-full rounded-full ${bar} transition-[width] duration-500`}
               style={{ width: `${result.confidence * 100}%` }} />
        </div>
        <p className="mt-2 text-xs text-ink-3">
          Probability of stress: {pct(result.stress_probability)}
        </p>
      </div>

      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-lg bg-surface-2 px-3 py-2">
          <dt className="text-ink-3">Model</dt>
          <dd className="font-medium">{MODEL_NAMES[result.model_name] ?? result.model_name}</dd>
        </div>
        <div className="rounded-lg bg-surface-2 px-3 py-2">
          <dt className="text-ink-3">Response time</dt>
          <dd className="font-medium tabular-nums">{result.latency_ms.toFixed(1)} ms</dd>
        </div>
      </dl>

      <CrisisNote />
    </section>
  );
}
