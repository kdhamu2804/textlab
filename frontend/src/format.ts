/** Display names and number formatting shared across pages. */

export const MODEL_NAMES: Record<string, string> = {
  majority: "Majority class",
  tfidf: "TF-IDF + LR",
  roberta: "RoBERTa",
};

export const MODEL_ORDER = ["majority", "tfidf", "roberta"];

export const METRIC_NAMES = {
  f1: "F1 (stress)",
  precision: "Precision",
  recall: "Recall",
  accuracy: "Accuracy",
} as const;

export type MetricKey = keyof typeof METRIC_NAMES;

export const pct = (x: number, digits = 0): string => `${(x * 100).toFixed(digits)}%`;

export const score = (x: number): string => x.toFixed(3);

/** Signed difference in points, e.g. 0.081 -> "+8.1". */
export const points = (x: number, digits = 1): string => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(digits)}`;

/** p-value for display; bootstrap p-values have a floor of 1/n_resamples. */
export const pValue = (p: number, floor = 0.001): string => (p < floor ? `p < ${floor}` : `p = ${p.toFixed(3)}`);

/** Find the standard-split, full-test-set metrics row of a model. */
export function headline<T extends { model: string; setting: string; test_set: string }>(
  rows: T[],
  model: string,
): T | undefined {
  return rows.find((r) => r.model === model && r.setting === "standard" && r.test_set === "all");
}

/**
 * Chart colors: slots 1-3 of a validated colorblind-safe categorical palette,
 * with separate steps for light and dark surfaces. Fixed per entity.
 */
export const SERIES_COLORS = {
  light: ["#2a78d6", "#eb6834", "#1baf7a"],
  dark: ["#3987e5", "#d95926", "#199e70"],
};
