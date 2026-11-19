import { ArrowLeft, ArrowRight, Check, Loader2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api, type LabelSheet } from "../api";

const ANNOTATORS = [
  { key: "dhamodhar", name: "Dhamodhar" },
  { key: "kushvanth", name: "Kushvanth" },
  { key: "uday", name: "Uday" },
];

const RULE = "Does the writer sound stressed, worried or upset?";

function ChooseAnnotator({ onPick }: { onPick: (key: string) => void }) {
  return (
    <div className="card mx-auto max-w-xl space-y-5 p-6 sm:p-8">
      <div>
        <h2 className="text-xl font-semibold">Who is labelling?</h2>
        <p className="mt-1 text-sm text-ink-3">
          Each team member labels the same 50 training posts on their own. Don't look at anyone else's labels.
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {ANNOTATORS.map((a) => (
          <button
            key={a.key}
            type="button"
            onClick={() => onPick(a.key)}
            className="rounded-xl border border-border px-4 py-3 font-medium transition-colors hover:border-accent hover:bg-accent-soft hover:text-accent"
          >
            {a.name}
          </button>
        ))}
      </div>
    </div>
  );
}

function Progress({ done, total }: { done: number; total: number }) {
  return (
    <div>
      <div className="mb-1.5 flex justify-between text-sm">
        <span className="text-ink-2">Progress</span>
        <span className="font-medium tabular-nums">{done} / {total}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-surface-2">
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${(done / total) * 100}%` }} />
      </div>
    </div>
  );
}

function Labeller({ annotator }: { annotator: string }) {
  const [sheet, setSheet] = useState<LabelSheet | null>(null);
  const [index, setIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.labelSheet(annotator)
      .then((s) => {
        setSheet(s);
        const firstOpen = s.items.findIndex((i) => i.label === null);
        setIndex(firstOpen === -1 ? 0 : firstOpen);
      })
      .catch((e: Error) => setError(e.message));
  }, [annotator]);

  const save = useCallback(
    async (label: 0 | 1) => {
      if (!sheet || saving) return;
      setSaving(true);
      setError(null);
      try {
        const next = await api.saveLabel(annotator, sheet.items[index].id, label);
        setSheet(next);
        const after = next.items.findIndex((item, i) => i > index && item.label === null);
        const anyOpen = next.items.findIndex((item) => item.label === null);
        setIndex(after !== -1 ? after : anyOpen !== -1 ? anyOpen : index);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setSaving(false);
      }
    },
    [annotator, index, saving, sheet],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!sheet || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return;
      const key = e.key.toLowerCase();
      if (key === "y" || key === "1") save(1);
      else if (key === "n" || key === "0") save(0);
      else if (key === "arrowright") setIndex((i) => Math.min(i + 1, sheet.n_total - 1));
      else if (key === "arrowleft") setIndex((i) => Math.max(i - 1, 0));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [save, sheet]);

  if (error && !sheet) return <p className="text-stress">{error}</p>;
  if (!sheet) return <p className="text-ink-3">Loading posts…</p>;
  const item = sheet.items[index];
  const finished = sheet.n_done === sheet.n_total;

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <Progress done={sheet.n_done} total={sheet.n_total} />
      {finished && (
        <p className="rounded-xl bg-calm-soft px-4 py-3 text-sm text-ink-2">
          <strong className="text-calm">All 50 labelled. Thank you!</strong> Your labels are saved in{" "}
          <code>data/label_check.csv</code>; agreement appears on the <Link to="/results#labels" className="font-medium text-accent underline">Results page</Link>.
          You can still go back and change any answer.
        </p>
      )}
      <article className="card space-y-5 p-6 sm:p-8">
        <div className="flex items-center justify-between text-sm text-ink-3">
          <span>Post {index + 1} of {sheet.n_total}</span>
          {item.label !== null && (
            <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${item.label ? "bg-stress-soft text-stress" : "bg-calm-soft text-calm"}`}>
              your answer: {item.label ? "yes" : "no"}
            </span>
          )}
        </div>
        <p className="whitespace-pre-wrap text-[15px] leading-7">{item.text}</p>
        <div className="border-t border-border pt-5">
          <p className="mb-3 text-center font-medium">{RULE}</p>
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => save(1)}
              disabled={saving}
              className={`inline-flex items-center justify-center gap-2 rounded-xl border-2 px-4 py-3 font-semibold transition-colors disabled:opacity-50 ${
                item.label === 1 ? "border-stress bg-stress text-white" : "border-stress/40 text-stress hover:bg-stress-soft"
              }`}
            >
              <Check size={18} aria-hidden /> Yes <kbd className="text-xs font-normal opacity-70">Y</kbd>
            </button>
            <button
              type="button"
              onClick={() => save(0)}
              disabled={saving}
              className={`inline-flex items-center justify-center gap-2 rounded-xl border-2 px-4 py-3 font-semibold transition-colors disabled:opacity-50 ${
                item.label === 0 ? "border-calm bg-calm text-white" : "border-calm/40 text-calm hover:bg-calm-soft"
              }`}
            >
              <X size={18} aria-hidden /> No <kbd className="text-xs font-normal opacity-70">N</kbd>
            </button>
          </div>
        </div>
      </article>
      {error && <p role="alert" className="text-sm text-stress">{error}</p>}
      <div className="flex items-center justify-between text-sm">
        <button type="button" onClick={() => setIndex((i) => Math.max(i - 1, 0))} disabled={index === 0}
                className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-ink-2 hover:bg-surface-2 disabled:opacity-40">
          <ArrowLeft size={16} aria-hidden /> Previous
        </button>
        {saving && <Loader2 size={16} className="animate-spin text-ink-3" aria-label="Saving" />}
        <button type="button" onClick={() => setIndex((i) => Math.min(i + 1, sheet.n_total - 1))}
                disabled={index === sheet.n_total - 1}
                className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-ink-2 hover:bg-surface-2 disabled:opacity-40">
          Next <ArrowRight size={16} aria-hidden />
        </button>
      </div>
      <p className="text-center text-xs text-ink-3">Keyboard: Y = yes, N = no, ← → to move. Answers save automatically.</p>
    </div>
  );
}

/** Label-check page: each team member labels the same 50 training posts (dataset label hidden). */
export default function Label() {
  const [params, setParams] = useSearchParams();
  const annotator = params.get("who");
  const known = ANNOTATORS.find((a) => a.key === annotator);

  return (
    <div className="space-y-8">
      <header className="max-w-3xl">
        <p className="mb-2 text-sm font-medium uppercase tracking-wider text-accent">Label check</p>
        <h1 className="text-4xl font-semibold tracking-tight">
          {known ? `Labelling as ${known.name}` : "Check the dataset labels"}
        </h1>
        <p className="mt-3 text-ink-2">
          Our biggest risk is noisy crowd labels. Read each post and answer one question: <em>{RULE}</em> The
          dataset's own label is hidden so it cannot influence you.
        </p>
        {known && (
          <button type="button" onClick={() => setParams({})} className="mt-3 text-sm font-medium text-accent hover:underline">
            Not {known.name}? Switch person
          </button>
        )}
      </header>
      {known ? <Labeller annotator={known.key} /> : <ChooseAnnotator onPick={(key) => setParams({ who: key })} />}
    </div>
  );
}
