import { Loader2, Sparkles } from "lucide-react";
import { useState } from "react";

import { api, type ModelInfo, type PredictResponse } from "../api";
import ResultCard from "../components/ResultCard";
import { EXAMPLES } from "../examples";
import { MODEL_NAMES, score } from "../format";
import { useAsync } from "../hooks";

const DEFAULT_LIMIT = 5000;

function ModelBadge({ info }: { info: ReturnType<typeof useAsync<ModelInfo>> }) {
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

export default function Home() {
  const info = useAsync(api.modelInfo);
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const limit = info.status === "ready" ? info.data.max_text_chars : DEFAULT_LIMIT;
  const tooLong = text.length > limit;
  const canSubmit = text.trim().length > 0 && !tooLong && !loading;

  async function analyze() {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await api.predict(text));
    } catch (e) {
      setResult(null);
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-10">
      <section className="max-w-2xl">
        <p className="mb-3 inline-flex items-center gap-1.5 rounded-full bg-accent-soft px-3 py-1 text-xs font-medium text-accent">
          <Sparkles size={14} aria-hidden /> NLP research demo
        </p>
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Does this post show signs of stress?
        </h1>
        <p className="mt-3 text-ink-2">
          Paste a Reddit-style post. A model trained on the Dreaddit dataset estimates whether the writer
          sounds stressed, and how confident it is.
        </p>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1fr_minmax(0,24rem)]">
        <section className="card space-y-4 p-5 sm:p-6">
          <label htmlFor="post" className="block font-medium">Post text</label>
          <textarea
            id="post"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
            }}
            rows={8}
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
                onClick={() => {
                  setText("");
                  setResult(null);
                  setError(null);
                }}
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
                  onClick={() => {
                    setText(ex.text);
                    setResult(null);
                    setError(null);
                  }}
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
          <section className="card grid place-items-center p-8 text-center text-sm text-ink-3">
            <p>Your result will appear here.<br />Tip: press Ctrl + Enter to analyze.</p>
          </section>
        )}
      </div>
    </div>
  );
}
