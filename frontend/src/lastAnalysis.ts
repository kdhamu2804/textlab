import type { EmotionsResponse, ExplainResponse } from "./api";

/** The post the user analyzed most recently, shown on the Results page. */
export interface Analysis {
  text: string;
  result: ExplainResponse;
  emotions: EmotionsResponse | null;
}

// Kept in memory only: nothing is written to storage, and a page refresh clears it.
let last: Analysis | null = null;

export function setLastAnalysis(analysis: Analysis): void {
  last = analysis;
}

export function getLastAnalysis(): Analysis | null {
  return last;
}
