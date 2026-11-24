import { BookOpen, Check, ChevronDown, ClipboardCheck, GraduationCap, ShieldCheck, X } from "lucide-react";
import type { ReactNode } from "react";

import CrisisNote from "../components/CrisisNote";

const TEAM = [
  { name: "Dhamodhar Konduru", role: "Collected the posts and built the simple model" },
  { name: "Kushvanth Thati", role: "Trained the AI model and studied its mistakes" },
  { name: "Uday Kiran Thurumella", role: "Built the website and checked its speed" },
];

const LEARNING = [
  {
    icon: BookOpen,
    title: "We gathered examples",
    text: "3,553 real Reddit posts. People had already read each one and marked it “stressed” or “not stressed”.",
  },
  {
    icon: GraduationCap,
    title: "The AI practised",
    text: "It studied 2,838 of these posts to learn what stress sounds like in writing.",
  },
  {
    icon: ClipboardCheck,
    title: "We tested it",
    text: "On the other 715 posts, which it had never seen, it was right about 8 out of 10 times.",
  },
];

const CAN = [
  "Tell if a short English post sounds stressed",
  "Show which sentences made it decide",
  "List the emotions in the post, like fear or joy",
];

const CANNOT = [
  "Tell how a person really feels: it only reads the words",
  "Always understand jokes, quotes or sarcasm",
  "Replace a doctor, counsellor or a real person",
];

const REFERENCES = [
  "Turcan, E. & McKeown, K. (2019). Dreaddit: A Reddit Dataset for Stress Analysis in Social Media. Proc. LOUHI Workshop @ EMNLP.",
  "Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.",
  "Demszky, D. et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL.",
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

function List({ items, good }: { items: string[]; good: boolean }) {
  const Icon = good ? Check : X;
  return (
    <ul className="space-y-2.5">
      {items.map((item) => (
        <li key={item} className="flex gap-3">
          <span className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full ${good ? "bg-calm-soft text-calm" : "bg-stress-soft text-stress"}`}>
            <Icon size={14} aria-hidden />
          </span>
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}

export default function About() {
  return (
    <div className="space-y-6">
      <header className="max-w-3xl pb-2">
        <p className="mb-2 text-sm font-medium uppercase tracking-wider text-accent">About</p>
        <h1 className="text-4xl font-semibold tracking-tight">What is TextLab?</h1>
        <p className="mt-3 text-lg text-ink-2">
          TextLab is a website that reads a short post and tells you if the writer sounds stressed. Three students
          made it for a university project at Saint Louis University.
        </p>
      </header>

      <Section title="Why we made it">
        <p>
          Many people write about their worries online before they talk to anyone. On big websites like Reddit there
          are too many posts for helpers to read. A tool that spots stressed posts can help people notice who might
          need support, sooner.
        </p>
      </Section>

      <Section title="How it learned">
        <ol className="grid gap-3 sm:grid-cols-3">
          {LEARNING.map(({ icon: Icon, title, text }, i) => (
            <li key={title} className="rounded-xl border border-border bg-bg p-4">
              <span className="mb-3 grid h-10 w-10 place-items-center rounded-xl bg-accent-soft text-accent">
                <Icon size={20} aria-hidden />
              </span>
              <p className="font-medium text-ink"><span className="text-ink-3">{i + 1}.</span> {title}</p>
              <p className="mt-1 text-sm">{text}</p>
            </li>
          ))}
        </ol>
      </Section>

      <div className="grid gap-6 md:grid-cols-2">
        <Section title="What it can do">
          <List items={CAN} good />
        </Section>
        <Section title="What it cannot do">
          <List items={CANNOT} good={false} />
        </Section>
      </div>

      <Section title="Your privacy">
        <p className="flex gap-3">
          <ShieldCheck className="mt-0.5 shrink-0 text-calm" size={20} aria-hidden />
          <span>We do not save anything you type. Your post is read once to give you a result, and then it is gone.</span>
        </p>
      </Section>

      <Section title="Who made it">
        <ul className="grid gap-3 sm:grid-cols-3">
          {TEAM.map(({ name, role }) => (
            <li key={name} className="rounded-xl bg-surface-2 p-4">
              <p className="font-medium text-ink">{name}</p>
              <p className="mt-1 text-sm text-ink-3">{role}</p>
            </li>
          ))}
        </ul>
      </Section>

      <details className="group card">
        <summary className="flex cursor-pointer list-none items-center gap-2 p-6 font-semibold sm:p-7 [&::-webkit-details-marker]:hidden">
          <ChevronDown size={18} className="text-accent transition-transform group-open:rotate-180" aria-hidden />
          For students and teachers: technical details
        </summary>
        <div className="space-y-4 border-t border-border p-6 text-sm leading-relaxed text-ink-2 sm:p-7">
          <p>
            <strong className="text-ink">Data:</strong> Dreaddit (Turcan &amp; McKeown, 2019), the official split of 2,838
            training and 715 test posts from ten subreddits, grouped into five topics. 20% of the training data is kept
            for validation (stratified, seed 42); the test set is used once.
          </p>
          <p>
            <strong className="text-ink">Models:</strong> a majority-class floor, TF-IDF + logistic regression, and
            <code> roberta-base</code> fine-tuned with early stopping, then temperature-scaled for calibrated
            probabilities. Explanations use sentence occlusion.
          </p>
          <p>
            <strong className="text-ink">Evaluation:</strong> stress-class F1 with 95% bootstrap confidence intervals,
            a paired bootstrap and McNemar&apos;s test, leave-one-topic-out testing, a 50-post team label check, and an
            API check against the offline model.
          </p>
          <p>
            <strong className="text-ink">Extension:</strong> a second RoBERTa trained on GoEmotions (Demszky et al.,
            2020) for 27 emotions plus neutral, with one threshold per emotion chosen on validation data.
          </p>
          <p><strong className="text-ink">System:</strong> a React website and a FastAPI server.</p>
          <div>
            <p className="mb-2 font-semibold text-ink">References</p>
            <ol className="list-decimal space-y-1.5 pl-5">
              {REFERENCES.map((r) => <li key={r}>{r}</li>)}
            </ol>
          </div>
        </div>
      </details>

      <CrisisNote />
    </div>
  );
}
