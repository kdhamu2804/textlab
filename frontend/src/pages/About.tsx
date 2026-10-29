import type { ReactNode } from "react";

import CrisisNote from "../components/CrisisNote";

const TEAM = ["Dhamodhar Konduru", "Kushvanth Thati", "Uday Kiran Thurumella"];

const TOPICS: [string, string][] = [
  ["abuse", "domesticviolence, survivorsofabuse"],
  ["anxiety", "anxiety, stress"],
  ["money", "assistance, homeless, almosthomeless, food_pantry"],
  ["ptsd", "ptsd"],
  ["relationships", "relationships"],
];

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card space-y-3 p-6">
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="space-y-3 leading-relaxed text-ink-2">{children}</div>
    </section>
  );
}

export default function About() {
  return (
    <div className="space-y-6">
      <header className="max-w-2xl">
        <h1 className="text-3xl font-semibold tracking-tight">About this project</h1>
        <p className="mt-3 text-ink-2">
          <em>Mental Health Detection from Text: A Web App That Finds Stress in Reddit Posts</em> — an NLP final
          project by team TextLab, MS in Artificial Intelligence, Saint Louis University.
        </p>
      </header>

      <Section title="Dataset">
        <p>
          <strong className="text-ink">Dreaddit</strong> (Turcan &amp; McKeown, 2019): short Reddit post segments
          labelled by crowd workers as <em>stress</em> or <em>no stress</em>. We use the official split of 2,838
          training and 715 test posts. Subreddits are grouped into five topics:
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
        <ol className="list-decimal space-y-1.5 pl-5">
          <li><strong className="text-ink">Majority class</strong> — always predicts the most common label (the floor).</li>
          <li><strong className="text-ink">TF-IDF + logistic regression</strong> — word 1–2 grams; regularization tuned on a validation split.</li>
          <li><strong className="text-ink">RoBERTa</strong> — <code>roberta-base</code> fine-tuned on a Colab GPU with early stopping on validation F1.</li>
        </ol>
        <p>
          Models are tuned only on a validation split of the training data (seed 42); the test set is used once,
          for the final scores. We report F1, precision and recall for the stress class plus accuracy, each with a 95%
          bootstrap confidence interval (1,000 resamples). A <strong className="text-ink">new-topic test</strong>{" "}
          trains on four topics and tests on the fifth, to see how well the model handles topics it has never seen.
        </p>
      </Section>

      <Section title="Limitations">
        <ul className="list-disc space-y-1.5 pl-5">
          <li>Labels describe how stressed a post <em>sounds</em> to annotators, not a clinical assessment of the writer.</li>
          <li>The data comes from ten English-language subreddits; results may not carry over to other platforms, groups or languages.</li>
          <li>Each topic has only ~130–160 test posts, so per-topic scores have wide confidence intervals.</li>
          <li>The model sees one short post without context, and can be fooled by sarcasm, quotes or stories about other people.</li>
          <li>Confidence scores are model probabilities, not a measure of how serious a situation is.</li>
        </ul>
      </Section>

      <Section title="Team">
        <ul className="flex flex-wrap gap-2">
          {TEAM.map((name) => (
            <li key={name} className="rounded-full bg-accent-soft px-3 py-1.5 text-sm font-medium text-accent">{name}</li>
          ))}
        </ul>
      </Section>

      <CrisisNote />
    </div>
  );
}
