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

export interface ModelInfo {
  model_name: string;
  display_name: string;
  is_fallback: boolean;
  test_f1: number | null;
  test_f1_ci: [number, number] | null;
  trained_at: string | null;
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
}

export interface MetricsResponse {
  metrics: MetricRow[];
  api_check: ApiCheck | null;
  latency: Latency | null;
  errors: ErrorRow[] | null;
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
  modelInfo: () => request<ModelInfo>("/model-info"),
  metrics: () => request<MetricsResponse>("/metrics"),
};
