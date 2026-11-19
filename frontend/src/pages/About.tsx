import { ArrowDown, ArrowRight } from "lucide-react";
import type { ReactNode } from "react";

import CrisisNote from "../components/CrisisNote";

const TEAM = [
  { name: "Dhamodhar Konduru", role: "Data pipeline, label check, majority and TF-IDF baselines" },
  { name: "Kushvanth Thati", role: "RoBERTa fine-tuning, new-topic tests, error analysis" },
  { name: "Uday Kiran Thurumella", role: "Web app, latency test, API consistency check" },
];

const TOPICS: [string, string][] = [
  ["abuse", "domesticviolence, survivorsofabuse"],
  ["anxiety", "anxiety, stress"],
  ["money", "assistance, homeless, almosthomeless, food_pantry"],
  ["ptsd", "ptsd"],
  ["relationships", "relationships"],
];

const PIPELINE = [
  { title: "Data", text: "Dreaddit, 2,838 train / 715 test posts, 5 topics" },
  { title: "Validation split", text: "80/20 of training data, stratified, seed 42" },
  { title: "Models", text: "Majority · TF-IDF + LR · RoBERTa-base" },
  { title: "Calibration", text: "Temperature scaling on validation logits" },
  { title: "Evaluation", text: "F1 + bootstrap CIs, paired tests, leave-one-topic-out" },
  { title: "Serving", text: "FastAPI + React, API checked against offline model" },
];

const REFERENCES = [
  "Turcan, E. & McKeown, K. (2019). Dreaddit: A Reddit Dataset for Stress Analysis in Social Media. Proc. LOUHI Workshop @ EMNLP.",
  "Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.",
  "Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML.",
  "Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
  "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions. Psychometrika 12(2).",
  "Zeiler, M. D. & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. ECCV (occlusion analysis).",
];

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card space-y-4 p-6 sm:p-7">
      <h2 className="text-xl font-semibold">{title}</h2>
      <div className="space-y-3 leading-relaxed text-ink-2">{children}</div>
    </section>
  );
}

function Pipeline() {
  return (
    <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {PIPELINE.map((step, i) => (
        <li key={step.title} className="relative rounded-xl border border-border bg-bg p-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-accent">Step {i + 1}</p>
          <p className="mt-1 font-medium text-ink">{step.title}</p>
          <p className="mt-0.5 text-sm text-ink-3">{step.text}</p>
        </li>
      ))}
    </ol>
  );
}

function Box({ title, lines }: { title: string; lines: string[] }) {
  return (
    <div className="flex-1 rounded-xl border border-border bg-bg p-4">
      <p className="font-medium text-ink">{title}</p>
      <ul className="mt-1 space-y-0.5 text-sm text-ink-3">
        {lines.map((l) => <li key={l}>{l}</li>)}
      </ul>
    </div>
  );
}

function Architecture() {
  return (
    <div className="flex flex-col items-stretch gap-2 md:flex-row md:items-center">
      <Box title="Browser" lines={["React + TypeScript", "Analyze · Results · About"]} />
      <ArrowRight className="mx-auto hidden shrink-0 text-ink-3 md:block" size={20} aria-hidden />
      <ArrowDown className="mx-auto shrink-0 text-ink-3 md:hidden" size={20} aria-hidden />
      <Box title="FastAPI backend" lines={["/predict · /explain", "/metrics · /model-info"]} />
      <ArrowRight className="mx-auto hidden shrink-0 text-ink-3 md:block" size={20} aria-hidden />
      <ArrowDown className="mx-auto shrink-0 text-ink-3 md:hidden" size={20} aria-hidden />
      <Box title="Model + results" lines={["RoBERTa (TF-IDF fallback)", "CSV results from the pipeline"]} />
    </div>
  );
}

export default function About() {
  return (
    <div className="space-y-6">
      <header className="max-w-3xl pb-2">
        <p className="mb-2 text-sm font-medium uppercase tracking-wider text-accent">About</p>
        <h1 className="text-4xl font-semibold tracking-tight">Mental Health Detection from Text</h1>
        <p className="mt-3 text-lg text-ink-2">
          A web app that finds stress in Reddit posts. NLP final project (application track) by team TextLab,
          MS in Artificial Intelligence, Saint Louis University.
        </p>
      </header>

      <Section title="Motivation">
        <p>
          Many people write about stress online before they ask for help. A tool that flags stressed posts could help
          moderators and peer-support communities notice who may need support. We ask whether a fine-tuned language
          model does this measurably better than a word-count baseline, and whether it still works on topics it was
          not trained on.
        </p>
      </Section>

      <Section title="Dataset">
        <p>
          <strong className="text-ink">Dreaddit</strong> (Turcan &amp; McKeown, 2019): short Reddit post segments
          labelled by Amazon Mechanical Turk workers as <em>stress</em> or <em>no stress</em>. We use the official split
          of 2,838 training and 715 test posts, and group the ten subreddits into five topics:
        </p>
        <ul className="grid gap-2 sm:grid-cols-2">
          {TOPICS.map(([topic, subs]) => (
            <li key={topic} className="rounded-lg bg-surface-2 px-3 py-2 text-sm">
              <span className="font-medium text-ink">{topic}</span> — r/{subs.split(", ").join(", r/")}
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Method">
        <Pipeline />
        <ol className="list-decimal space-y-1.5 pl-5">
          <li><strong className="text-ink">Majority class</strong>: always predicts the most common label (the floor).</li>
          <li><strong className="text-ink">TF-IDF + logistic regression</strong>: word 1–2 grams; C chosen from six values by validation F1.</li>
          <li>
            <strong className="text-ink">RoBERTa</strong>: <code>roberta-base</code> fine-tuned for up to 6 epochs (lr 2e-5,
            batch 16, max 256 tokens) with early stopping on validation F1, then temperature-scaled so its
            probabilities are calibrated.
          </li>
        </ol>
        <p>
          The test set is used once, for the final scores. We report stress-class F1, precision and recall plus
          accuracy with 95% bootstrap confidence intervals, compare models with a paired bootstrap and McNemar's test,
          and run a leave-one-topic-out test. In the app, each sentence is scored by <em>occlusion</em>: how much the
          stress probability drops when that sentence is removed.
        </p>
      </Section>

      <Section title="System architecture">
        <Architecture />
        <p className="text-sm">
          The same predictor code is used by the evaluation scripts and the API, and an automated check confirms the
          API returns the same label as the offline model for all 715 test posts.
        </p>
      </Section>

      <Section title="Limitations and ethics">
        <ul className="list-disc space-y-1.5 pl-5">
          <li>Labels describe how stressed a post <em>sounds</em> to annotators, not a clinical assessment of the writer.</li>
          <li>The data comes from ten English-language subreddits; results may not carry over to other platforms, groups or languages.</li>
          <li>Each topic has only ~130–160 test posts, so per-topic scores have wide confidence intervals.</li>
          <li>The model sees one short post without context, and can be fooled by sarcasm, quotes or stories about other people.</li>
          <li>Explanations show what the model relies on, not why a person feels the way they do.</li>
          <li>This tool must not be used to make decisions about individuals without human review. No user text is stored.</li>
        </ul>
      </Section>

      <Section title="Team">
        <ul className="grid gap-3 sm:grid-cols-3">
          {TEAM.map(({ name, role }) => (
            <li key={name} className="rounded-xl bg-surface-2 p-4">
              <p className="font-medium text-ink">{name}</p>
              <p className="mt-1 text-sm text-ink-3">{role}</p>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="References">
        <ol className="list-decimal space-y-1.5 pl-5 text-sm">
          {REFERENCES.map((r) => <li key={r}>{r}</li>)}
        </ol>
      </Section>

      <CrisisNote />
    </div>
  );
}
