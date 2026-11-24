/** Typed client for the TextLab FastAPI backend. */

/** In development the API runs on :8000; a production build is served by the API itself. */
export const API_URL: string =
  import.meta.env.VITE_API_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

export type Label = "stress" | "no stress";

export interface PredictResponse {
  label: Label;
  confidence: number;
  stress_probability: number;
  model_name: string;
  latency_ms: number;
}

export interface Segment {
  text: string;
  start: number;
  end: number;
  /** -1..1: positive pushes toward stress, negative toward no stress. */
  score: number;
  /** Drop in P(stress) when the segment is removed (sentence occlusion only). */
  delta: number | null;
}

export interface ExplainResponse extends PredictResponse {
  text: string;
  unit: "word" | "sentence";
  method: string;
  segments: Segment[];
}

export interface ModelInfo {
  model_name: string;
  display_name: string;
  is_fallback: boolean;
  test_f1: number | null;
  test_f1_ci: [number, number] | null;
  trained_at: string | null;
  temperature: number | null;
  max_text_chars: number;
}

export interface MetricRow {
  model: string;
  setting: "standard" | "new_topic" | "in_topic";
  test_set: string;
  n: number;
  f1: number;
  f1_low: number;
  f1_high: number;
  precision: number;
  precision_low: number;
  precision_high: number;
  recall: number;
  recall_low: number;
  recall_high: number;
  accuracy: number;
  accuracy_low: number;
  accuracy_high: number;
}

export interface ApiCheck {
  model_name: string;
  n_posts: number;
  n_label_match: number;
  label_match_pct: number;
  max_prob_diff: number;
}

export interface Latency {
  model_name: string;
  n_requests: number;
  server_mean_ms: number;
  server_p95_ms: number;
  client_mean_ms: number;
  client_p95_ms: number;
}

export interface ErrorRow {
  id: string;
  topic: string;
  text: string;
  true_label: Label;
  predicted_label: Label;
  confidence: number;
  error_type: string;
  group: string | null;
  reviewed_by?: string | null;
}

export interface ConfusionRow {
  model: string;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface SignificanceRow {
  model_a: string;
  model_b: string;
  metric: string;
  difference: number;
  difference_low: number;
  difference_high: number;
  p_value: number;
  n_resamples: number;
  only_a_correct: number;
  only_b_correct: number;
  mcnemar_p: number;
}

export interface CalibrationBin {
  model: string;
  bin_low: number;
  bin_high: number;
  n: number;
  mean_predicted: number;
  observed_rate: number;
}

export interface CalibrationSummary {
  model: string;
  ece: number;
  brier: number;
  n: number;
}

export interface Temperature {
  temperature: number;
  n_val: number;
  val_nll_before: number;
  val_nll_after: number;
  test_ece_before: number;
  test_ece_after: number;
}

export interface AgreementRow {
  [key: string]: string | number | null;
}

export interface EmotionScore {
  emotion: string;
  probability: number;
  threshold: number;
  detected: boolean;
}

export interface EmotionsResponse {
  available: boolean;
  model_name: string | null;
  display_name: string | null;
  emotions: EmotionScore[];
  latency_ms: number | null;
}

export interface EmotionMetricRow {
  model: string;
  n: number;
  macro_f1: number;
  macro_f1_low: number;
  macro_f1_high: number;
  micro_f1: number;
  micro_f1_low: number;
  micro_f1_high: number;
}

export interface EmotionLabelRow {
  model: string;
  emotion: string;
  support: number;
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface LabelItem {
  id: string;
  text: string;
  label: 0 | 1 | null;
}

export interface LabelSheet {
  annotator: string;
  items: LabelItem[];
  n_done: number;
  n_total: number;
}

export interface MetricsResponse {
  metrics: MetricRow[];
  api_check: ApiCheck | null;
  latency: Latency | null;
  errors: ErrorRow[] | null;
  confusion: ConfusionRow[] | null;
  significance: SignificanceRow[] | null;
  calibration: CalibrationBin[] | null;
  calibration_summary: CalibrationSummary[] | null;
  temperature: Temperature | null;
  label_agreement: AgreementRow[] | null;
  emotions_metrics: EmotionMetricRow[] | null;
  emotions_per_label: EmotionLabelRow[] | null;
  emotions_significance: SignificanceRow[] | null;
}

/** Fetch JSON and turn API errors into readable messages. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error(`Cannot reach the API at ${API_URL}. Is the backend running?`);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`);
  }
  return body as T;
}

export const api = {
  predict: (text: string) =>
    request<PredictResponse>("/predict", { method: "POST", body: JSON.stringify({ text }) }),
  explain: (text: string) =>
    request<ExplainResponse>("/explain", { method: "POST", body: JSON.stringify({ text }) }),
  emotions: (text: string) =>
    request<EmotionsResponse>("/emotions", { method: "POST", body: JSON.stringify({ text }) }),
  labelSheet: (annotator: string) => request<LabelSheet>(`/label-check/${encodeURIComponent(annotator)}`),
  saveLabel: (annotator: string, postId: string, label: 0 | 1 | null) =>
    request<LabelSheet>(`/label-check/${encodeURIComponent(annotator)}/${encodeURIComponent(postId)}`, {
      method: "PUT",
      body: JSON.stringify({ label }),
    }),
  modelInfo: () => request<ModelInfo>("/model-info"),
  metrics: () => request<MetricsResponse>("/metrics"),
};
