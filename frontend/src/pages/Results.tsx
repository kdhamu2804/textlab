import { CheckCircle2, ChevronDown, CircleAlert, Lightbulb } from "lucide-react";
import { type ReactNode, useState } from "react";

import { api, type ErrorRow, type MetricRow, type MetricsResponse } from "../api";
import { Calibration, ConfusionMatrices, LabelAgreement, SectionHeading, Significance } from "../components/Analysis";
import CiBarChart, { type CiValue } from "../components/CiBarChart";
import { EmotionResults } from "../components/Emotions";
import {
  METRIC_NAMES,
  MODEL_NAMES,
  MODEL_ORDER,
  SERIES_COLORS,
  type MetricKey,
  headline,
  pct,
  points,
  pValue,
  score,
} from "../format";
import { useAsync, useIsDark } from "../hooks";

const TOPICS = ["abuse", "anxiety", "money", "ptsd", "relationships"];
/** Success criterion from the proposal: RoBERTa beats TF-IDF by at least 3 F1 points. */
const TARGET_GAIN = 0.03;

const SECTIONS = [
  ["answer", "Answer"],
  ["comparison", "Model comparison"],
  ["significance", "Significance"],
  ["confusion", "Mistakes"],
  ["new-topic", "New topics"],
  ["calibration", "Calibration"],
  ["labels", "Label check"],
  ["app", "App checks"],
  ["errors", "Errors"],
  ["emotions", "Emotions (extension)"],
] as const;

function ci(row: MetricRow, metric: MetricKey): CiValue {
  return { value: row[metric], low: row[`${metric}_low`], high: row[`${metric}_high`], n: row.n };
}

function find(rows: MetricRow[], model: string, setting: string, testSet: string) {
  return rows.find((r) => r.model === model && r.setting === setting && r.test_set === testSet);
}

function Stat({ label, value, note, accent = false }: { label: string; value: string; note?: string; accent?: boolean }) {
  return (
    <div className="card p-5">
      <p className="text-sm text-ink-3">{label}</p>
      <p className={`mt-1 text-3xl font-semibold tabular-nums ${accent ? "text-accent" : ""}`}>{value}</p>
      {note && <p className="mt-1 text-xs leading-snug text-ink-3">{note}</p>}
    </div>
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

/** Technical text and charts, hidden until the reader asks for them. */
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

/** Headline numbers and the answer to the research question. */
function Answer({ data }: { data: MetricsResponse }) {
  const rob = headline(data.metrics, "roberta");
  const tf = headline(data.metrics, "tfidf");
  const sig = data.significance?.find((s) => s.model_a === "tfidf" && s.model_b === "roberta");
  const newTopic = data.metrics.filter((r) => r.model === "roberta" && r.setting === "new_topic");
  const worst = newTopic.length ? newTopic.reduce((a, b) => (a.f1 < b.f1 ? a : b)) : undefined;
  const tfBest = data.metrics.filter((r) => r.model === "tfidf" && r.setting === "in_topic");
  const beatsSeen = worst && tfBest.length
    ? newTopic.every((r) => r.f1 > (tfBest.find((t) => t.test_set === r.test_set)?.f1 ?? 1))
    : false;
  const ece = data.calibration_summary?.find((s) => s.model === "roberta");

  if (!rob || !tf || !sig) {
    return (
      <div className="card flex gap-3 p-5 text-sm text-ink-2">
        <CircleAlert className="shrink-0 text-ink-3" size={20} aria-hidden />
        RoBERTa results appear here after <code>scripts.train_roberta</code>, <code>scripts.evaluate</code> and{" "}
        <code>scripts.analyze</code> have been run.
      </div>
    );
  }
  const met = sig.difference >= TARGET_GAIN && sig.difference_low > 0;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="RoBERTa F1 (stress)" value={score(rob.f1)} accent
              note={`95% CI [${score(rob.f1_low)}, ${score(rob.f1_high)}] · ${rob.n} test posts`} />
        <Stat label="Gain over TF-IDF" value={`${points(sig.difference)} pts`}
              note={`95% CI [${points(sig.difference_low)}, ${points(sig.difference_high)}] · ${pValue(sig.p_value)}`} />
        {worst && <Stat label="Worst unseen topic" value={score(worst.f1)}
                        note={`F1 on ${worst.test_set} when it was held out of training`} />}
        {ece && <Stat label="Calibration error (ECE)" value={score(ece.ece)} note="After temperature scaling; 0 = perfect" />}
      </div>
      <div className={`flex gap-3 rounded-2xl border p-5 ${met ? "border-calm/30 bg-calm-soft" : "border-stress/30 bg-stress-soft"}`}>
        {met
          ? <CheckCircle2 className="mt-0.5 shrink-0 text-calm" size={22} aria-hidden />
          : <CircleAlert className="mt-0.5 shrink-0 text-stress" size={22} aria-hidden />}
        <div className="space-y-1 text-sm leading-relaxed text-ink-2">
          <p className="text-base font-semibold text-ink">
            {met ? "Yes: RoBERTa beats the TF-IDF baseline, and the gain is statistically significant." : "The success criterion was not met."}
          </p>
          <p>
            RoBERTa scores {score(rob.f1)} and the simple TF-IDF model scores {score(tf.f1)} (1.0 would be perfect).
            Our goal was a gain of at least 3 points; we got {points(sig.difference)}.
          </p>
        </div>
      </div>
      <Details>
        <p>
          Research question: <em>can our app find stress with RoBERTa and beat a TF-IDF model by at least 3 F1 points?</em>{" "}
          RoBERTa reaches F1 {score(rob.f1)} vs {score(tf.f1)} ({points(sig.difference)} points; the 95% CI of the gain
          excludes zero; paired bootstrap {pValue(sig.p_value)}, McNemar {pValue(sig.mcnemar_p)}).
          {beatsSeen && " On every held-out topic, RoBERTa still scores higher than TF-IDF does on topics it was trained on."}
        </p>
        <p>
          Every number on this page is loaded live from <code>results/</code>. F1, precision and recall are for the
          <em> stress</em> class. Models were tuned on a validation split of the training data; the 715 test posts were
          used only for these final scores.
        </p>
      </Details>
    </div>
  );
}

function ModelComparison({ rows, colors }: { rows: MetricRow[]; colors: string[] }) {
  const models = MODEL_ORDER.filter((m) => find(rows, m, "standard", "all"));
  const metrics = Object.keys(METRIC_NAMES) as MetricKey[];
  const values = Object.fromEntries(
    metrics.map((metric) => [
      metric,
      Object.fromEntries(models.map((m) => [m, ci(find(rows, m, "standard", "all")!, metric)])),
    ]),
  );
  return (
    <CiBarChart
      title="All three models on the 715 test posts"
      description="Bars show the score; whiskers show the 95% bootstrap confidence interval (1,000 resamples)."
      categories={metrics.map((m) => ({ key: m, name: METRIC_NAMES[m] }))}
      series={models.map((m) => ({ key: m, name: MODEL_NAMES[m], color: colors[MODEL_ORDER.indexOf(m)] }))}
      values={values}
    />
  );
}

function NewTopic({ rows, colors }: { rows: MetricRow[]; colors: string[] }) {
  const models = MODEL_ORDER.filter((m) => m !== "majority" && rows.some((r) => r.model === m && r.setting === "new_topic"));
  const [model, setModel] = useState(models[models.length - 1]);
  const values = Object.fromEntries(
    TOPICS.map((t) => {
      const seen = find(rows, model, "in_topic", t);
      const held = find(rows, model, "new_topic", t);
      return [t, { in_topic: seen && ci(seen, "f1"), new_topic: held && ci(held, "f1") }];
    }),
  );
  return (
    <div className="space-y-3">
      {models.length > 1 && (
        <div className="inline-flex gap-1 rounded-xl border border-border bg-surface p-1" role="group" aria-label="Choose model">
          {models.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setModel(m)}
              aria-pressed={m === model}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                m === model ? "bg-accent text-accent-ink" : "text-ink-2 hover:bg-surface-2"
              }`}
            >
              {MODEL_NAMES[m]}
            </button>
          ))}
        </div>
      )}
      <CiBarChart
        title={`Seen vs unseen topics · ${MODEL_NAMES[model]}`}
        description="Stress F1 on each topic's test posts: standard model (topic seen in training) vs. a model trained on the other four topics only."
        categories={TOPICS.map((t) => ({ key: t, name: t }))}
        series={[
          { key: "in_topic", name: "Seen topic", color: colors[0] },
          { key: "new_topic", name: "New topic (held out)", color: colors[1] },
        ]}
        values={values}
      />
    </div>
  );
}

function AppChecks({ data }: { data: MetricsResponse }) {
  const { api_check, latency } = data;
  if (!api_check && !latency) return null;
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {api_check && (
        <Stat
          label="API matches offline model"
          value={pct(api_check.label_match_pct / 100, 1)}
          note={`${api_check.n_label_match} of ${api_check.n_posts} test posts · ${MODEL_NAMES[api_check.model_name] ?? api_check.model_name}`}
        />
      )}
      {latency && (
        <>
          <Stat label="Mean prediction time" value={`${latency.server_mean_ms.toFixed(0)} ms`}
                note={`Inside the API, ${latency.n_requests} requests on CPU (goal: under 1,000 ms)`} />
          <Stat label="95th percentile" value={`${latency.server_p95_ms.toFixed(0)} ms`}
                note={`Full round trip: mean ${latency.client_mean_ms.toFixed(0)} ms, p95 ${latency.client_p95_ms.toFixed(0)} ms`} />
        </>
      )}
    </div>
  );
}

type ErrorFilter = "all" | "false positive" | "false negative";

/** Error groups as stacked bars (false positives / false negatives); click a group to filter. */
function ErrorGroups({
  errors,
  selected,
  onSelect,
}: {
  errors: ErrorRow[];
  selected: string | null;
  onSelect: (group: string | null) => void;
}) {
  const grouped = errors.filter((e) => e.group);
  if (!grouped.length) return null;
  const groups = [...new Set(grouped.map((e) => e.group as string))]
    .map((g) => ({
      name: g,
      fp: grouped.filter((e) => e.group === g && e.error_type === "false positive").length,
      fn: grouped.filter((e) => e.group === g && e.error_type === "false negative").length,
    }))
    .sort((a, b) => b.fp + b.fn - (a.fp + a.fn));
  const max = Math.max(...groups.map((g) => g.fp + g.fn));
  const reviewers = [...new Set(grouped.map((e) => e.reviewed_by).filter(Boolean))];
  return (
    <div className="card space-y-4 p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold">Error types</h3>
          <p className="mt-1 text-sm text-ink-3">
            Each misclassified post was read and assigned one group (<code>data/error_groups.csv</code>). Click a group to
            filter the posts below.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-ink-3">
          <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-stress" />false positive</span>
          <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-accent" />false negative</span>
        </div>
      </div>
      <ul className="space-y-1.5">
        {groups.map((g) => (
          <li key={g.name}>
            <button
              type="button"
              onClick={() => onSelect(selected === g.name ? null : g.name)}
              aria-pressed={selected === g.name}
              className={`grid w-full grid-cols-[minmax(0,14rem)_1fr_2rem] items-center gap-3 rounded-lg px-2 py-1.5 text-left text-sm transition-colors ${
                selected === g.name ? "bg-accent-soft" : "hover:bg-surface-2"
              }`}
            >
              <span className="truncate">{g.name}</span>
              <span className="flex h-3 overflow-hidden rounded-full bg-surface-2">
                <span className="bg-stress" style={{ width: `${(g.fp / max) * 100}%` }} />
                <span className="bg-accent" style={{ width: `${(g.fn / max) * 100}%` }} />
              </span>
              <span className="text-right tabular-nums text-ink-2">{g.fp + g.fn}</span>
            </button>
          </li>
        ))}
      </ul>
      <p className="text-xs text-ink-3">
        {reviewers.length
          ? `Reviewed by: ${reviewers.join(", ")}.`
          : "Draft grouping prepared with AI assistance; awaiting review by the team."}
      </p>
    </div>
  );
}

function Errors({ data }: { data: MetricsResponse }) {
  const [filter, setFilter] = useState<ErrorFilter>("all");
  const [open, setOpen] = useState(false);
  const [group, setGroup] = useState<string | null>(null);
  if (!data.errors?.length) return null;
  const counts = {
    all: data.errors.length,
    "false positive": data.errors.filter((e) => e.error_type === "false positive").length,
    "false negative": data.errors.filter((e) => e.error_type === "false negative").length,
  };
  const filtered = data.errors.filter(
    (e) => (filter === "all" || e.error_type === filter) && (group === null || e.group === group),
  );
  const shown = open ? filtered : filtered.slice(0, 6);
  return (
    <div className="space-y-4">
      <ErrorGroups errors={data.errors} selected={group} onSelect={setGroup} />
      <div className="inline-flex gap-1 rounded-xl border border-border bg-surface p-1" role="group" aria-label="Filter errors">
        {(Object.keys(counts) as ErrorFilter[]).map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            aria-pressed={f === filter}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium capitalize transition-colors ${
              f === filter ? "bg-accent text-accent-ink" : "text-ink-2 hover:bg-surface-2"
            }`}
          >
            {f} <span className="tabular-nums opacity-70">{counts[f]}</span>
          </button>
        ))}
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {shown.map((e) => (
          <article key={e.id} className="card space-y-2 p-4 text-sm">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded-full bg-surface-2 px-2 py-0.5">{e.topic}</span>
              <span className="rounded-full bg-stress-soft px-2 py-0.5 text-stress">{e.error_type}</span>
              {e.group && <span className="rounded-full bg-accent-soft px-2 py-0.5 text-accent">{e.group}</span>}
              <span className="ml-auto text-ink-3">
                true: {e.true_label} · predicted {pct(e.confidence)}
              </span>
            </div>
            <p className="line-clamp-6 leading-relaxed text-ink-2">{e.text}</p>
          </article>
        ))}
      </div>
      {filtered.length > 6 && (
        <button type="button" onClick={() => setOpen((o) => !o)} className="text-sm font-medium text-accent hover:underline">
          {open ? "Show fewer" : `Show all ${filtered.length}`}
        </button>
      )}
    </div>
  );
}

function SectionNav() {
  return (
    <nav aria-label="Sections" className="sticky top-16 z-[5] -mx-4 border-b border-border bg-bg/90 px-4 py-2 backdrop-blur">
      <ul className="flex gap-1 overflow-x-auto text-sm [scrollbar-width:none]">
        {SECTIONS.map(([id, label]) => (
          <li key={id}>
            <a href={`#${id}`} className="block whitespace-nowrap rounded-lg px-3 py-1.5 text-ink-2 hover:bg-surface-2 hover:text-ink">
              {label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

export default function Results() {
  const state = useAsync(api.metrics);
  const colors = SERIES_COLORS[useIsDark() ? "dark" : "light"];

  if (state.status === "loading") return <p className="text-ink-3">Loading results…</p>;
  if (state.status === "error") return <p className="text-stress">{state.error}</p>;
  const data = state.data;
  if (!data.metrics.length) return <p className="text-ink-3">No results yet. Run the evaluation scripts first.</p>;

  // Numbers for the plain-English lines, read from the same results as the charts.
  const rob = headline(data.metrics, "roberta");
  const tf = headline(data.metrics, "tfidf");
  const sig = data.significance?.find((s) => s.model_a === "tfidf" && s.model_b === "roberta");
  const confRob = data.confusion?.find((c) => c.model === "roberta");
  const confTf = data.confusion?.find((c) => c.model === "tfidf");
  const drops = TOPICS.map((t) => {
    const seen = find(data.metrics, "roberta", "in_topic", t);
    const held = find(data.metrics, "roberta", "new_topic", t);
    return seen && held ? seen.f1 - held.f1 : null;
  }).filter((d): d is number => d !== null);
  const maxDrop = drops.length ? Math.max(...drops) : null;
  const ece = data.calibration_summary?.find((s) => s.model === "roberta")?.ece;
  const eceBefore = data.temperature?.test_ece_before;
  const vsDataset = data.label_agreement?.find((r) => r.rater_b === "dataset_label");
  const labelN = vsDataset ? Number(vsDataset.n) : 0;
  const labelDisagree = vsDataset ? Math.round((labelN * (100 - Number(vsDataset.agreement_pct))) / 100) : 0;
  const falseAlarms = data.errors?.filter((e) => e.error_type === "false positive").length ?? 0;
  const emoRob = data.emotions_metrics?.find((m) => m.model === "roberta");
  const emoTf = data.emotions_metrics?.find((m) => m.model === "tfidf");

  return (
    <div className="space-y-12">
      <header className="max-w-3xl">
        <p className="mb-2 text-sm font-medium uppercase tracking-wider text-accent">Evaluation</p>
        <h1 className="text-4xl font-semibold tracking-tight">Results</h1>
        <p className="mt-3 text-ink-2">
          How well our models work, explained in plain words. Each section starts with what it means; click
          <strong className="text-ink"> Show details</strong> to see the technical numbers and charts.
        </p>
      </header>

      <SectionNav />

      <section className="space-y-5">
        <SectionHeading id="answer" title="Answer to the research question" />
        <Answer data={data} />
      </section>

      <section className="space-y-5">
        <SectionHeading id="comparison" title="Model comparison" />
        {rob && tf && (
          <Meaning>
            Taller bars are better. RoBERTa ({score(rob.f1)}) beats the simple word-counting model ({score(tf.f1)}). The
            “always say stress” model looks fine on recall only because it never misses a stress post, but half of its
            alarms are wrong.
          </Meaning>
        )}
        <ModelComparison rows={data.metrics} colors={colors} />
        <Details>
          <p>
            F1 balances precision (how many “stress” answers are right) and recall (how many stress posts are found).
            The majority-class floor scores F1 0.68 because it labels every post as stress (recall 1.0), which is why
            precision and accuracy are reported alongside F1. Whiskers are 95% bootstrap confidence intervals.
          </p>
        </Details>
      </section>

      {data.significance?.length ? (
        <section className="space-y-5">
          <SectionHeading id="significance" title="Is the difference real?" />
          <Meaning>
            We re-checked the comparison 10,000 times on reshuffled test posts, and RoBERTa came out ahead every time
            {sig ? ` (${pValue(sig.p_value)})` : ""}. So the win is real, not luck.
          </Meaning>
          <Details>
            <p>
              Both models are scored on the same test posts, so we use paired tests: a paired bootstrap of the F1
              difference (10,000 resamples, one-sided) and McNemar&apos;s exact test on the posts where exactly one model
              is right.
            </p>
            <Significance rows={data.significance} />
          </Details>
        </section>
      ) : null}

      {data.confusion?.length ? (
        <section className="space-y-5">
          <SectionHeading id="confusion" title="Kinds of mistakes" />
          <Meaning>
            RoBERTa makes fewer of both kinds of mistakes
            {confRob && confTf
              ? `: ${confRob.fp} false alarms (TF-IDF: ${confTf.fp}) and ${confRob.fn} missed stress posts (TF-IDF: ${confTf.fn}).`
              : ": fewer false alarms and fewer missed stress posts."}
          </Meaning>
          <Details>
            <p>
              A confusion matrix counts four outcomes: true negatives, false positives (said stress, was not), false
              negatives (missed a stress post) and true positives.
            </p>
            <ConfusionMatrices rows={data.confusion} />
          </Details>
        </section>
      ) : null}

      <section className="space-y-5">
        <SectionHeading id="new-topic" title="Does it work on topics it has never seen?" />
        <Meaning>
          We trained RoBERTa without one topic and then tested it on that topic.
          {maxDrop !== null && ` Its score dropped by at most ${score(maxDrop)}.`} So it learned real signs of stress, not
          just topic words.
        </Meaning>
        <NewTopic rows={data.metrics} colors={colors} />
        <Details>
          <p>
            Leave-one-topic-out: for each of the five topics, a model is trained on the other four and tested on that
            topic, then compared with the standard model that did see the topic. A small drop means the model learned
            signs of stress rather than topic words.
          </p>
        </Details>
      </section>

      {data.calibration?.length && data.calibration_summary?.length ? (
        <section className="space-y-5">
          <SectionHeading id="calibration" title="Can the confidence score be trusted?" />
          <Meaning>
            When the app says “80% sure”, it should be right about 80% of the time. Raw RoBERTa was over-confident, so
            we corrected it
            {ece !== undefined && eceBefore !== undefined
              ? `: its calibration error dropped from ${score(eceBefore)} to ${score(ece)} (0 is perfect).`
              : "."}
          </Meaning>
          <Details>
            <p>
              Temperature scaling divides the model&apos;s raw scores by one number, fitted on validation data. It never
              changes which answer the model gives, only how confident it is.
            </p>
            <Calibration bins={data.calibration} summary={data.calibration_summary} temperature={data.temperature} />
          </Details>
        </section>
      ) : null}

      <section className="space-y-5">
        <SectionHeading id="labels" title="Label check" />
        <Meaning>
          {vsDataset
            ? `Three of us labelled the same ${labelN} posts ourselves. We disagreed with the dataset on about ${labelDisagree} of them, so some dataset labels are noisy, and we report that.`
            : "Three of us label the same 50 posts ourselves, to check how good the dataset's labels are."}
        </Meaning>
        <Details>
          <p>
            Our biggest risk was noisy crowd labels. Three team members label the same 50 training posts with one rule:
            “Does the writer sound stressed, worried or upset?” Agreement is shown as a percentage and as Cohen&apos;s
            kappa (agreement corrected for chance; 1 = perfect, 0 = guessing).
          </p>
          <LabelAgreement rows={data.label_agreement} />
        </Details>
      </section>

      <section className="space-y-5">
        <SectionHeading id="app" title="App checks" />
        <Meaning>
          {data.api_check && data.latency
            ? `The website gave exactly the same answer as our tested model on ${data.api_check.n_label_match} of ${data.api_check.n_posts} posts, and answers in about ${data.latency.server_mean_ms.toFixed(0)} ms, far under our 1-second goal.`
            : "The website must give the same answers as our tested model, in under one second."}
        </Meaning>
        <AppChecks data={data} />
      </section>

      {data.errors?.length ? (
        <section className="space-y-5">
          <SectionHeading id="errors" title="Where the model is wrong" />
          <Meaning>
            Most mistakes ({falseAlarms} of {data.errors.length} we studied) are false alarms: calm posts about serious
            topics, like someone calmly describing a past trauma. Click a group below to read real examples.
          </Meaning>
          <Errors data={data} />
          <Details>
            <p>
              {data.errors.length} misclassified test posts (random sample, seed 42) from <code>results/errors.csv</code>,
              grouped by hand into error types in <code>data/error_groups.csv</code>.
            </p>
          </Details>
        </section>
      ) : null}

      {data.emotions_metrics?.length ? (
        <section className="space-y-5">
          <SectionHeading id="emotions" title="Extension: detecting 27 emotions" />
          <Meaning>
            A second model finds emotions like joy, fear and anger in a post.
            {emoRob && emoTf && ` RoBERTa scores ${score(emoRob.macro_f1)} against ${score(emoTf.macro_f1)} for the simple model.`}{" "}
            Common emotions like gratitude are easy; rare ones like grief are still hard.
          </Meaning>
          <EmotionResults metrics={data.emotions_metrics} perLabel={data.emotions_per_label}
                          significance={data.emotions_significance} />
          <Details>
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
