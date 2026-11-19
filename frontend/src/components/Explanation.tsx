import { Info, ListOrdered } from "lucide-react";
import type { ReactNode } from "react";

import type { ExplainResponse, Segment } from "../api";

/** Background tint for a segment: stress color for +, calm color for −, stronger = more influence. */
function tint(score: number): string | undefined {
  const strength = Math.round(Math.min(Math.abs(score), 1) * 55);
  if (strength < 6) return undefined;
  const color = score > 0 ? "var(--stress)" : "var(--calm)";
  return `color-mix(in srgb, ${color} ${strength}%, transparent)`;
}

/** The post with each word or sentence tinted by its influence. */
function HighlightedText({ text, segments }: { text: string; segments: Segment[] }) {
  const parts: ReactNode[] = [];
  let cursor = 0;
  segments.forEach((s, i) => {
    if (s.start > cursor) parts.push(text.slice(cursor, s.start));
    const effect = s.delta !== null ? ` (${s.delta >= 0 ? "+" : "−"}${Math.abs(s.delta * 100).toFixed(1)} points)` : "";
    parts.push(
      <mark
        key={i}
        className="rounded-[5px] px-0.5 py-0.5 text-ink [box-decoration-break:clone]"
        style={{ background: tint(s.score) ?? "transparent" }}
        title={`Influence ${s.score >= 0 ? "toward stress" : "toward no stress"}: ${s.score.toFixed(2)}${effect}`}
      >
        {text.slice(s.start, s.end)}
      </mark>,
    );
    cursor = s.end;
  });
  if (cursor < text.length) parts.push(text.slice(cursor));
  return <p className="whitespace-pre-wrap text-[15px] leading-8">{parts}</p>;
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-4 text-xs text-ink-3">
      <span className="inline-flex items-center gap-1.5">
        <span className="h-3 w-6 rounded" style={{ background: tint(0.9) }} /> pushes toward stress
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="h-3 w-6 rounded" style={{ background: tint(-0.9) }} /> pushes toward no stress
      </span>
      <span>darker = stronger</span>
    </div>
  );
}

/** Ranked list of the segments that raised P(stress) the most. */
function TopDrivers({ result }: { result: ExplainResponse }) {
  const top = [...result.segments].filter((s) => s.score > 0.1).sort((a, b) => b.score - a.score).slice(0, 3);
  if (!top.length) return null;
  const sentences = result.unit === "sentence";
  return (
    <div>
      <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold">
        <ListOrdered size={16} aria-hidden /> Strongest stress signals
      </h3>
      <ol className="space-y-2">
        {top.map((s, i) => (
          <li key={s.start} className="flex items-start gap-3 rounded-xl bg-surface-2 px-3 py-2 text-sm">
            <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-stress-soft text-[11px] font-semibold text-stress">
              {i + 1}
            </span>
            <span className={`min-w-0 flex-1 ${sentences ? "" : "font-medium"}`}>{sentences ? `“${s.text}”` : s.text}</span>
            {s.delta !== null && (
              <span className="shrink-0 font-medium tabular-nums text-stress">
                +{(s.delta * 100).toFixed(1)} pts
              </span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

/** "Why this prediction?" panel shown under the result. */
export default function Explanation({ result }: { result: ExplainResponse }) {
  const unit = result.unit === "sentence" ? "sentence" : "word";
  return (
    <section className="card space-y-5 p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Why this prediction?</h2>
          <p className="mt-1 text-sm text-ink-3">
            Each {unit} is tinted by how much it moved the model toward <em>stress</em> or <em>no stress</em>.
          </p>
        </div>
        <Legend />
      </div>
      <div className="rounded-xl border border-border bg-bg p-4 sm:p-5">
        <HighlightedText text={result.text} segments={result.segments} />
      </div>
      <TopDrivers result={result} />
      <p className="flex items-start gap-2 text-xs text-ink-3">
        <Info size={14} className="mt-px shrink-0" aria-hidden />
        <span>
          Method: {result.method}. Explanations show what this model relies on, not why the writer feels the way
          they do.
        </span>
      </p>
    </section>
  );
}
