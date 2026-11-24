import { ArrowLeft, ChevronDown, Lightbulb, ScanSearch } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { api, type MetricRow } from "../api";
import { SectionHeading } from "../components/Analysis";
import { EmotionBars, EmotionPanel, EmotionScoreTable } from "../components/Emotions";
import Explanation from "../components/Explanation";
import ResultCard from "../components/ResultCard";
import { getLastAnalysis } from "../lastAnalysis";
import { headline, score } from "../format";
import { useAsync } from "../hooks";

/** Out of every 10 posts, how many a model gets right (from test-set accuracy). */
function outOf10(rows: MetricRow[], model: string) {
  const row = headline(rows, model);
  return row ? Math.round(row.accuracy * 10) : null;
}

/** "How good is it?" in one picture: 10 dots, the green ones are posts it gets right. */
function ScoreBoard({ rows }: { rows: MetricRow[] }) {
  const best = headline(rows, "roberta");
  const right = outOf10(rows, "roberta");
  if (!best || right === null) return null;
  const simple = outOf10(rows, "tfidf");
  const guess = outOf10(rows, "majority");
  return (
    <section className="space-y-5">
      <SectionHeading id="score" title="How good is our model?" />
      <div className="card flex flex-col items-center gap-4 p-8 text-center">
        <span className="flex flex-wrap justify-center gap-2" role="img" aria-label={`${right} out of 10 posts right`}>
          {Array.from({ length: 10 }, (_, i) => (
            <span key={i} className={`h-8 w-8 rounded-full ${i < right ? "bg-calm" : "border-2 border-border bg-surface-2"}`} />
          ))}
        </span>
        <p className="text-4xl font-semibold text-accent sm:text-5xl">{right} out of 10 posts right</p>
        <p className="max-w-xl text-lg text-ink-2">
          We tested it on {best.n} posts it had never seen before.
          {simple !== null && guess !== null && ` A simple model gets ${simple} out of 10, and just guessing gets ${guess}.`}
        </p>
      </div>
    </section>
  );
}

/** The result for the post the user just analyzed. */
function YourPost() {
  const analysis = getLastAnalysis();
  if (!analysis) {
    return (
      <header className="space-y-5">
        <div>
          <p className="mb-2 text-sm font-medium uppercase tracking-wider text-accent">Results</p>
          <h1 className="text-4xl font-semibold tracking-tight">Your result</h1>
        </div>
        <div className="card flex flex-wrap items-center justify-between gap-4 p-6">
          <div className="flex items-center gap-3 text-ink-2">
            <span className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-surface-2 text-ink-3">
              <ScanSearch size={22} aria-hidden />
            </span>
            You have not analyzed a post yet. Paste a post on the Analyze page to see its result here.
          </div>
          <Link to="/" className="rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-accent-ink hover:opacity-90">
            Analyze a post
          </Link>
        </div>
      </header>
    );
  }
  const { text, result, emotions } = analysis;
  return (
    <header className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="mb-2 text-sm font-medium uppercase tracking-wider text-accent">Results</p>
          <h1 className="text-4xl font-semibold tracking-tight">Your result</h1>
        </div>
        <Link to="/" className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:underline">
          <ArrowLeft size={16} aria-hidden /> Analyze another post
        </Link>
      </div>
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,23rem)]">
        <section className="card space-y-3 p-5 sm:p-6">
          <h2 className="font-medium">Your post</h2>
          <p className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded-xl bg-bg p-4 leading-relaxed text-ink-2">{text}</p>
        </section>
        <ResultCard result={result} />
      </div>
      <Explanation result={result} />
      <EmotionPanel result={emotions} />
    </header>
  );
}

/** One plain-English sentence under a chart. */
function Meaning({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-3 rounded-2xl border border-accent/25 bg-accent-soft p-4 leading-relaxed text-ink">
      <Lightbulb className="mt-0.5 shrink-0 text-accent" size={20} aria-hidden />
      <p><span className="font-semibold">What this means: </span>{children}</p>
    </div>
  );
}

/** Technical text, hidden until the reader asks for it. */
function Details({ children, label = "Show details" }: { children: ReactNode; label?: string }) {
  return (
    <details className="group rounded-2xl border border-border bg-surface">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-2xl px-4 py-3 text-sm font-medium text-accent hover:bg-surface-2 [&::-webkit-details-marker]:hidden">
        <ChevronDown size={16} className="transition-transform group-open:rotate-180" aria-hidden />
        <span className="group-open:hidden">{label}</span>
        <span className="hidden group-open:inline">Hide details</span>
      </summary>
      <div className="space-y-4 border-t border-border p-4 text-sm leading-relaxed text-ink-2">{children}</div>
    </details>
  );
}

export default function Results() {
  const state = useAsync(api.metrics);

  if (state.status !== "ready" || !state.data.metrics.length) {
    return (
      <div className="space-y-12">
        <YourPost />
        {state.status === "loading" && <p className="text-ink-3">Loading model scores…</p>}
        {state.status === "error" && <p className="text-stress">{state.error}</p>}
      </div>
    );
  }
  const data = state.data;

  const emoRob = data.emotions_metrics?.find((m) => m.model === "roberta");
  const emoTf = data.emotions_metrics?.find((m) => m.model === "tfidf");

  return (
    <div className="space-y-12">
      <YourPost />

      <ScoreBoard rows={data.metrics} />

      {data.emotions_metrics?.length ? (
        <section className="space-y-5">
          <SectionHeading id="emotions" title="Which emotions can it find?" />
          <Meaning>
            A second model finds 27 emotions, like joy, fear and anger. Each bar shows how well it finds one emotion:
            the longer the bar, the better. Common emotions like gratitude are easy; rare ones like grief are still hard.
          </Meaning>
          <EmotionBars
            metrics={data.emotions_metrics}
            perLabel={data.emotions_per_label}
            title="How well it finds each emotion"
            description="Longer bar = better (1.0 = perfect). The grey tick shows the simple word-counting model, for comparison."
          />
          <Details>
            {emoRob && emoTf && (
              <p>
                Overall, RoBERTa scores {score(emoRob.macro_f1)} macro F1 against {score(emoTf.macro_f1)} for TF-IDF.
              </p>
            )}
            <EmotionScoreTable metrics={data.emotions_metrics} significance={data.emotions_significance} />
            <p>
              A separate task on a separate dataset: GoEmotions (Demszky et al., 2020), 58,000 Reddit comments labelled
              with 27 emotions plus neutral, where one comment can have several. Each emotion&apos;s decision threshold
              was chosen on the validation split; the scores are on the 5,427 test comments. Macro F1 is the average over
              the 28 labels; micro F1 pools all decisions. The stress results above do not change.
            </p>
          </Details>
        </section>
      ) : null}
    </div>
  );
}
