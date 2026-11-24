import { ArrowRight, BrainCircuit, FileText, Loader2, ScanSearch, Sparkles } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { api, type EmotionsResponse, type ExplainResponse, type MetricsResponse, type ModelInfo } from "../api";
import { EmotionPanel } from "../components/Emotions";
import Explanation from "../components/Explanation";
import ResultCard from "../components/ResultCard";
import { EXAMPLES } from "../examples";
import { MODEL_NAMES, headline, points, pValue, score } from "../format";
import { type AsyncState, useAsync } from "../hooks";

const DEFAULT_LIMIT = 5000;

function ModelBadge({ info }: { info: AsyncState<ModelInfo> }) {
  if (info.status === "loading") return <span className="text-ink-3">Connecting to the API…</span>;
  if (info.status === "error") return <span className="text-stress">{info.error}</span>;
  const { model_name, test_f1 } = info.data;
  return (
    <span className="inline-flex items-center gap-2 text-ink-3">
      <span className="h-2 w-2 rounded-full bg-calm" aria-hidden />
      Using {MODEL_NAMES[model_name] ?? model_name}
      {test_f1 !== null && <> · test F1 {score(test_f1)}</>}
    </span>
  );
}

/** Three headline findings, read from the saved results. */
function KeyFindings({ metrics }: { metrics: AsyncState<MetricsResponse> }) {
  if (metrics.status !== "ready") return null;
  const { metrics: rows, significance } = metrics.data;
  const best = headline(rows, "roberta") ?? headline(rows, "tfidf");
  if (!best) return null;
  const gain = significance?.find((s) => s.model_b === best.model);
  const items = [
    { value: score(best.f1), label: `${MODEL_NAMES[best.model]} F1 on ${best.n} held-out posts` },
    gain && { value: points(gain.difference), label: `F1 points over ${MODEL_NAMES[gain.model_a]} (${pValue(gain.p_value)})` },
    { value: `${(best.accuracy * 100).toFixed(1)}%`, label: "accuracy on the test set" },
  ].filter(Boolean) as { value: string; label: string }[];
  return (
    <dl className="mt-8 grid max-w-2xl grid-cols-1 gap-3 sm:grid-cols-3">
      {items.map((item) => (
        <div key={item.label} className="rounded-2xl border border-border bg-surface/70 px-4 py-3">
          <dt className="sr-only">{item.label}</dt>
          <dd className="text-2xl font-semibold tabular-nums text-accent">{item.value}</dd>
          <dd className="mt-0.5 text-xs leading-snug text-ink-3">{item.label}</dd>
        </div>
      ))}
    </dl>
  );
}

const STEPS = [
  { icon: FileText, title: "Paste a post", text: "Any Reddit-style text, up to 5,000 characters." },
  { icon: BrainCircuit, title: "Fine-tuned RoBERTa", text: "Trained on 2,838 Dreaddit posts, with calibrated probabilities." },
  { icon: ScanSearch, title: "See the evidence", text: "Each sentence is scored by how much it moves the prediction." },
];

function HowItWorks() {
  return (
    <section aria-labelledby="how" className="space-y-4">
      <h2 id="how" className="text-sm font-semibold uppercase tracking-wider text-ink-3">How it works</h2>
      <ol className="grid gap-4 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, text }, i) => (
          <li key={title} className="card flex gap-4 p-5">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent">
              <Icon size={20} aria-hidden />
            </span>
            <div>
              <p className="font-medium"><span className="text-ink-3">{i + 1}.</span> {title}</p>
              <p className="mt-1 text-sm text-ink-3">{text}</p>
            </div>
          </li>
        ))}
      </ol>
      <Link to="/results" className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:underline">
        See the full evaluation <ArrowRight size={16} aria-hidden />
      </Link>
    </section>
  );
}

export default function Home() {
  const info = useAsync(api.modelInfo);
  const metrics = useAsync(api.metrics);
  const [text, setText] = useState("");
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [emotions, setEmotions] = useState<EmotionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const limit = info.status === "ready" ? info.data.max_text_chars : DEFAULT_LIMIT;
  const tooLong = text.length > limit;
  const canSubmit = text.trim().length > 0 && !tooLong && !loading;

  function reset(next = "") {
    setText(next);
    setResult(null);
    setEmotions(null);
    setError(null);
  }

  async function analyze() {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    try {
      const [explained, emotionResult] = await Promise.all([
        api.explain(text),
        api.emotions(text).catch(() => null),
      ]);
      setResult(explained);
      setEmotions(emotionResult);
    } catch (e) {
      setResult(null);
      setEmotions(null);
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-12">
      <section className="relative">
        <div
          className="pointer-events-none absolute -top-24 right-0 -z-10 h-72 w-72 rounded-full bg-accent/15 blur-3xl"
          aria-hidden
        />
        <p className="mb-4 inline-flex items-center gap-1.5 rounded-full bg-accent-soft px-3 py-1 text-xs font-medium text-accent">
          <Sparkles size={14} aria-hidden /> NLP research demo · Dreaddit
        </p>
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-balance sm:text-5xl">
          Does this post show signs of stress?
        </h1>
        <p className="mt-4 max-w-2xl text-lg text-ink-2">
          A fine-tuned language model reads a Reddit-style post, estimates how likely it is that the writer sounds
          stressed, and shows which sentences drove that judgement.
        </p>
        <KeyFindings metrics={metrics} />
      </section>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,23rem)]">
        <section className="card space-y-4 p-5 sm:p-6">
          <label htmlFor="post" className="block font-medium">Post text</label>
          <textarea
            id="post"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
            }}
            rows={9}
            placeholder="Type or paste a post here…"
            className="w-full resize-y rounded-xl border border-border bg-bg p-4 leading-relaxed outline-none placeholder:text-ink-3 focus:border-accent focus:ring-2 focus:ring-accent/25"
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className={`text-xs tabular-nums ${tooLong ? "text-stress" : "text-ink-3"}`}>
              {text.length.toLocaleString()} / {limit.toLocaleString()} characters
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => reset()}
                className="rounded-xl px-4 py-2.5 text-sm font-medium text-ink-2 hover:bg-surface-2"
              >
                Clear
              </button>
              <button
                type="button"
                onClick={analyze}
                disabled={!canSubmit}
                className="inline-flex items-center gap-2 rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-accent-ink shadow-sm transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading && <Loader2 size={16} className="animate-spin" aria-hidden />}
                Analyze
              </button>
            </div>
          </div>
          {error && (
            <p role="alert" className="rounded-lg bg-stress-soft px-3 py-2 text-sm text-stress">{error}</p>
          )}

          <div className="border-t border-border pt-4">
            <p className="mb-2 text-sm text-ink-3">Try an example</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.title}
                  type="button"
                  onClick={() => reset(ex.text)}
                  className="rounded-full border border-border px-3 py-1.5 text-sm text-ink-2 transition-colors hover:border-accent hover:text-accent"
                >
                  {ex.title}
                </button>
              ))}
            </div>
          </div>
          <p className="text-xs"><ModelBadge info={info} /></p>
        </section>

        {result ? (
          <ResultCard result={result} />
        ) : (
          <section className="card grid min-h-72 place-items-center p-8 text-center text-sm text-ink-3">
            <div className="space-y-3">
              <span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-surface-2">
                <ScanSearch size={22} aria-hidden />
              </span>
              <p>Your result will appear here.<br />Tip: press Ctrl + Enter to analyze.</p>
            </div>
          </section>
        )}
      </div>

      {result && <Explanation result={result} />}

      {result && <EmotionPanel result={emotions} />}

      <HowItWorks />
    </div>
  );
}
