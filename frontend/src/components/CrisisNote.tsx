import { HeartHandshake } from "lucide-react";

export const DISCLAIMER = "This is a research demo, not a medical diagnosis.";

function LifelineLink() {
  return (
    <a
      href="https://988lifeline.org"
      target="_blank"
      rel="noreferrer"
      className="font-medium text-accent underline underline-offset-2"
    >
      988 Suicide &amp; Crisis Lifeline
    </a>
  );
}

/** Research-demo disclaimer plus the 988 Lifeline (call or text 988). */
export default function CrisisNote({ compact = false }: { compact?: boolean }) {
  if (compact) {
    return (
      <p>
        {DISCLAIMER} In crisis? Call or text <a href="tel:988" className="font-medium text-accent">988</a>{" "}
        (<LifelineLink />).
      </p>
    );
  }
  return (
    <div className="flex gap-3 rounded-xl bg-accent-soft p-4 text-sm text-ink-2" role="note">
      <HeartHandshake className="mt-0.5 shrink-0 text-accent" size={20} aria-hidden />
      <p>
        <strong className="text-ink">{DISCLAIMER}</strong> If you or someone you know is in crisis, call or text{" "}
        <a href="tel:988" className="font-semibold text-accent underline underline-offset-2">988</a> to reach the{" "}
        <LifelineLink /> (US, free, 24/7).
      </p>
    </div>
  );
}
