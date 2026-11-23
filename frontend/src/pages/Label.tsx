import { ArrowLeft, ArrowRight, Check, Loader2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api, type LabelItem, type LabelSheet } from "../api";

const ANNOTATORS = [
  { key: "dhamodhar", name: "Dhamodhar" },
  { key: "kushvanth", name: "Kushvanth" },
  { key: "uday", name: "Uday" },
];

const RULE = "Does the writer sound stressed, worried or upset?";
const PAGE_SIZE = 10;

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

/** One post with its own Yes / No buttons. */
function PostCard({
  item,
  number,
  saving,
  onSave,
}: {
  item: LabelItem;
  number: number;
  saving: boolean;
  onSave: (label: 0 | 1) => void;
}) {
  return (
    <article
      className={`card space-y-4 p-5 transition-colors ${item.label === null ? "" : "border-accent/30 bg-accent-soft/30"}`}
    >
      <div className="flex items-center justify-between text-sm text-ink-3">
        <span className="font-medium">Post {number}</span>
        {saving ? (
          <Loader2 size={16} className="animate-spin" aria-label="Saving" />
        ) : item.label === null ? (
          <span>not answered</span>
        ) : (
          <span className="inline-flex items-center gap-1 text-accent"><Check size={14} aria-hidden /> saved</span>
        )}
      </div>
      <p className="whitespace-pre-wrap text-[15px] leading-7">{item.text}</p>
      <div className="flex flex-wrap items-center gap-3 border-t border-border pt-4">
        <span className="mr-auto text-sm font-medium text-ink-2">{RULE}</span>
        <button
          type="button"
          onClick={() => onSave(1)}
          disabled={saving}
          aria-pressed={item.label === 1}
          className={`inline-flex min-w-24 items-center justify-center gap-1.5 rounded-xl border-2 px-4 py-2 font-semibold transition-colors disabled:opacity-50 ${
            item.label === 1 ? "border-stress bg-stress text-white" : "border-stress/40 text-stress hover:bg-stress-soft"
          }`}
        >
          <Check size={16} aria-hidden /> Yes
        </button>
        <button
          type="button"
          onClick={() => onSave(0)}
          disabled={saving}
          aria-pressed={item.label === 0}
          className={`inline-flex min-w-24 items-center justify-center gap-1.5 rounded-xl border-2 px-4 py-2 font-semibold transition-colors disabled:opacity-50 ${
            item.label === 0 ? "border-calm bg-calm text-white" : "border-calm/40 text-calm hover:bg-calm-soft"
          }`}
        >
          <X size={16} aria-hidden /> No
        </button>
      </div>
    </article>
  );
}

function Pager({ page, pages, onPage }: { page: number; pages: number; onPage: (p: number) => void }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <button type="button" onClick={() => onPage(page - 1)} disabled={page === 0}
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm text-ink-2 hover:bg-surface-2 disabled:opacity-40">
        <ArrowLeft size={16} aria-hidden /> Previous page
      </button>
      <div className="flex gap-1" role="group" aria-label="Pages">
        {Array.from({ length: pages }, (_, p) => (
          <button
            key={p}
            type="button"
            onClick={() => onPage(p)}
            aria-current={p === page ? "page" : undefined}
            className={`h-9 w-9 rounded-lg text-sm font-medium ${p === page ? "bg-accent text-accent-ink" : "text-ink-2 hover:bg-surface-2"}`}
          >
            {p + 1}
          </button>
        ))}
      </div>
      <button type="button" onClick={() => onPage(page + 1)} disabled={page === pages - 1}
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm text-ink-2 hover:bg-surface-2 disabled:opacity-40">
        Next page <ArrowRight size={16} aria-hidden />
      </button>
    </div>
  );
}

function Labeller({ annotator }: { annotator: string }) {
  const [sheet, setSheet] = useState<LabelSheet | null>(null);
  const [page, setPage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [savingId, setSavingId] = useState<string | null>(null);

  useEffect(() => {
    api.labelSheet(annotator)
      .then((s) => {
        setSheet(s);
        const firstOpen = s.items.findIndex((i) => i.label === null);
        setPage(firstOpen === -1 ? 0 : Math.floor(firstOpen / PAGE_SIZE));
      })
      .catch((e: Error) => setError(e.message));
  }, [annotator]);

  async function save(id: string, label: 0 | 1) {
    if (savingId) return;
    setSavingId(id);
    setError(null);
    try {
      setSheet(await api.saveLabel(annotator, id, label));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSavingId(null);
    }
  }

  function goTo(p: number) {
    setPage(p);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  if (error && !sheet) return <p className="text-stress">{error}</p>;
  if (!sheet) return <p className="text-ink-3">Loading posts…</p>;
  const pages = Math.ceil(sheet.n_total / PAGE_SIZE);
  const start = page * PAGE_SIZE;
  const items = sheet.items.slice(start, start + PAGE_SIZE);
  const pageDone = items.filter((i) => i.label !== null).length;
  const finished = sheet.n_done === sheet.n_total;

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <Progress done={sheet.n_done} total={sheet.n_total} />
      {finished && (
        <p className="rounded-xl bg-calm-soft px-4 py-3 text-sm text-ink-2">
          <strong className="text-calm">All 50 labelled. Thank you!</strong> Your labels are saved in{" "}
          <code>data/label_check.csv</code>; agreement appears on the{" "}
          <Link to="/results#labels" className="font-medium text-accent underline">Results page</Link>. You can still
          change any answer.
        </p>
      )}
      <Pager page={page} pages={pages} onPage={goTo} />
      <p className="text-sm text-ink-3">
        Page {page + 1} of {pages} · posts {start + 1}–{start + items.length} · {pageDone} of {items.length} answered on this page.
        Answers save as soon as you click.
      </p>
      {error && <p role="alert" className="text-sm text-stress">{error}</p>}
      <div className="space-y-4">
        {items.map((item, i) => (
          <PostCard key={item.id} item={item} number={start + i + 1} saving={savingId === item.id}
                    onSave={(label) => save(item.id, label)} />
        ))}
      </div>
      <Pager page={page} pages={pages} onPage={goTo} />
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
