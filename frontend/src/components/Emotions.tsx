import { Info, Smile } from "lucide-react";

import type { EmotionLabelRow, EmotionMetricRow, EmotionScore, EmotionsResponse, SignificanceRow } from "../api";
import { pct, points, pValue, score } from "../format";

const SHOWN = 6;

function EmotionBar({ e, strong }: { e: EmotionScore; strong: boolean }) {
  return (
    <li className="grid grid-cols-[7.5rem_1fr_3rem] items-center gap-3 text-sm">
      <span className={`capitalize ${strong ? "font-medium" : "text-ink-3"}`}>{e.emotion}</span>
      <span className="relative h-2.5 overflow-hidden rounded-full bg-surface-2">
        <span
          className={`absolute inset-y-0 left-0 rounded-full ${strong ? "bg-accent" : "bg-ink-3/40"}`}
          style={{ width: `${e.probability * 100}%` }}
        />
        <span className="absolute inset-y-0 w-px bg-ink-2" style={{ left: `${e.threshold * 100}%` }}
              title={`Threshold ${pct(e.threshold)}`} />
      </span>
      <span className={`text-right tabular-nums ${strong ? "" : "text-ink-3"}`}>{pct(e.probability)}</span>
    </li>
  );
}

/** "Emotions in this post" panel on the Analyze page (GoEmotions extension). */
export function EmotionPanel({ result }: { result: EmotionsResponse | null }) {
  if (!result) return null;
  if (!result.available) {
    return (
      <section className="card flex gap-3 p-5 text-sm text-ink-3">
        <Info size={18} className="mt-0.5 shrink-0" aria-hidden />
        <p>
          Emotion detection (27 emotions + neutral) runs in the local version of the app. The free online server does
          not have enough memory for a second model.
        </p>
      </section>
    );
  }
  const detected = result.emotions.filter((e) => e.detected);
  const others = result.emotions.filter((e) => !e.detected).slice(0, Math.max(0, SHOWN - detected.length));
  return (
    <section className="card space-y-4 p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold"><Smile size={20} aria-hidden /> Emotions in this post</h2>
          <p className="mt-1 text-sm text-ink-3">
            Extension trained on GoEmotions (27 emotions + neutral). A post can show several emotions; an emotion counts
            as detected when it passes its own threshold (the thin line), chosen on validation data.
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {detected.length ? detected.map((e) => (
            <span key={e.emotion} className="rounded-full bg-accent-soft px-2.5 py-1 text-xs font-medium capitalize text-accent">
              {e.emotion}
            </span>
          )) : <span className="text-xs text-ink-3">no emotion passed its threshold</span>}
        </div>
      </div>
      <ul className="space-y-2">
        {[...detected, ...others].map((e) => <EmotionBar key={e.emotion} e={e} strong={e.detected} />)}
      </ul>
      <p className="text-xs text-ink-3">
        Model: {result.display_name}. Emotions are a separate task from stress: the stress result above comes from the
        Dreaddit model.
      </p>
    </section>
  );
}

const MODEL_LABEL: Record<string, string> = { tfidf: "TF-IDF + LR", roberta: "RoBERTa" };

/** Results-page section for the GoEmotions extension. */
export function EmotionResults({
  metrics,
  perLabel,
  significance,
}: {
  metrics: EmotionMetricRow[];
  perLabel: EmotionLabelRow[] | null;
  significance: SignificanceRow[] | null;
}) {
  const best = metrics.find((m) => m.model === "roberta") ?? metrics[metrics.length - 1];
  const sig = significance?.[0];
  const rows = (perLabel ?? []).filter((r) => r.model === best.model).sort((a, b) => b.f1 - a.f1);
  const base = (perLabel ?? []).filter((r) => r.model !== best.model);
  return (
    <div className="space-y-4">
      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-ink-3">
              <th className="px-5 py-3 font-medium">Model</th>
              <th className="px-5 py-3 font-medium">Macro F1 (mean over 28 labels)</th>
              <th className="px-5 py-3 font-medium">Micro F1 (all decisions pooled)</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => (
              <tr key={m.model} className="border-b border-border last:border-0">
                <td className="px-5 py-3 font-medium">{MODEL_LABEL[m.model] ?? m.model}</td>
                <td className="px-5 py-3 tabular-nums">
                  {score(m.macro_f1)} <span className="text-ink-3">[{score(m.macro_f1_low)}, {score(m.macro_f1_high)}]</span>
                </td>
                <td className="px-5 py-3 tabular-nums">
                  {score(m.micro_f1)} <span className="text-ink-3">[{score(m.micro_f1_low)}, {score(m.micro_f1_high)}]</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {sig && (
          <p className="border-t border-border px-5 py-3 text-sm text-ink-2">
            RoBERTa vs TF-IDF macro F1: <strong>{points(sig.difference)} pts</strong>{" "}
            <span className="text-ink-3">(95% CI [{points(sig.difference_low)}, {points(sig.difference_high)}], paired bootstrap {pValue(sig.p_value)})</span>
          </p>
        )}
      </div>
      {rows.length > 0 && (
        <div className="card p-5 sm:p-6">
          <p className="mb-1 font-semibold">F1 per emotion · {MODEL_LABEL[best.model] ?? best.model}</p>
          <p className="mb-4 text-sm text-ink-3">
            5,427 test comments. Grey tick = TF-IDF for comparison. Rare emotions (few training examples) are hardest.
          </p>
          <ul className="grid gap-x-8 gap-y-1.5 sm:grid-cols-2">
            {rows.map((r) => {
              const b = base.find((x) => x.emotion === r.emotion);
              return (
                <li key={r.emotion} className="grid grid-cols-[7.5rem_1fr_2.5rem] items-center gap-3 text-sm"
                    title={`precision ${score(r.precision)} · recall ${score(r.recall)} · ${r.support} test comments`}>
                  <span className="capitalize">{r.emotion}</span>
                  <span className="relative h-2.5 rounded-full bg-surface-2">
                    <span className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${r.f1 * 100}%` }} />
                    {b && <span className="absolute -inset-y-0.5 w-0.5 bg-ink-3" style={{ left: `${b.f1 * 100}%` }} />}
                  </span>
                  <span className="text-right tabular-nums">{r.f1.toFixed(2)}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
