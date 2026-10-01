// Same-origin API client with relative URLs (works behind the RunPod proxy) and CSRF handling.

export interface ApiError {
  code: string;
  message: string;
  fix_hint: string;
  context?: Record<string, unknown>;
}

export class ApiFailure extends Error {
  constructor(public status: number, public error: ApiError) {
    super(`[${error.code}] ${error.message}`);
  }
}

const CSRF_KEY = "archrender.csrf";

export function setCsrf(token: string | null): void {
  if (token) sessionStorage.setItem(CSRF_KEY, token);
  else sessionStorage.removeItem(CSRF_KEY);
}

function csrf(): string {
  return sessionStorage.getItem(CSRF_KEY) ?? "";
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const method = (init.method ?? "GET").toUpperCase();
  if (method !== "GET") headers.set("X-CSRF-Token", csrf());
  if (init.body && typeof init.body === "string") headers.set("Content-Type", "application/json");
  const res = await fetch(`api/v1/${path}`, { ...init, headers, credentials: "same-origin" });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const err: ApiError = data?.error ?? { code: "HTTP", message: res.statusText, fix_hint: "" };
    if (res.status === 401) window.dispatchEvent(new CustomEvent("archrender:unauthorized"));
    throw new ApiFailure(res.status, err);
  }
  return data as T;
}

export const post = <T>(path: string, body?: unknown) =>
  api<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export function blobUrl(projectId: string, sha256: string): string {
  return `api/v1/projects/${projectId}/blobs/${sha256}`;
}

export interface User { id: string; name: string; role: string }
export interface Project { id: string; name: string; created_at: string; latitude: number | null; longitude: number | null }
export interface Doc { id: string; filename: string; kind: string; size: number; created_at: string }
export interface CasRef { sha256: string; size: number; media_type: string; name: string | null }
export interface Check {
  name: string; family: string; passed: boolean; value: number | null; base_value: number | null;
  delta: number | null; threshold: number | null; comparator: string; estimator: string; mock: boolean;
  evidence: Record<string, unknown>;
}
export interface ViewResult {
  view_id: string; camera_id: string; status: string; reason: string; delivered: CasRef; delivered_jpg: CasRef;
  base: CasRef; checks: Check[]; attempts: number; uses_mocks: boolean;
}
export interface Gate { gate: string; status: string; policy: string; evidence: Record<string, unknown>; decided_by: string | null; notes: string | null }
export interface Run {
  id: string; project_id: string; job_id: string; status: string; created_at: string; config: Record<string, unknown>;
  gates: Gate[];
  result: null | { views: ViewResult[]; glb: CasRef; report: CasRef; uses_mocks: boolean; assumptions: { key: string; value: unknown; reason: string; stage: string }[] };
  bundle: null | { id: string; sha256: string; size: number; status: string };
}
export interface Job { id: string; status: string; progress: number; stage: string | null; error: ApiError | null }
export interface PageInfo {
  page_id: string; document_id: string; filename: string; index: number; kind: string;
  preview: CasRef | null; label: string | null; model_label: string | null; confidence: number | null;
  needs_review: boolean; overridden: boolean; scale: number | null; north_deg: number | null;
  title: Record<string, string>; tags: number; analysed: boolean;
}
export interface ReviewItem { id: string; kind: string; subject_id: string; status: string; payload: Record<string, unknown> }
export const PAGE_CLASSES = [
  "floor_plan", "ceiling_plan", "section", "elevation", "detail", "site_plan",
  "schedule", "text_document", "photo", "moodboard", "other",
] as const;

// ---- plans (S2 / Gate A) ----------------------------------------------------------------------
export interface Prov {
  method: string; confidence: number; note?: string | null;
  assist?: { trigger: string; evidence_coverage: number; snap_residual_px: number; accepted: boolean; user_confirmed: boolean } | null;
}
export interface Fact<T> { value: T; unit?: string | null; provenance: Prov[]; status: string }
export interface Pt { x: number; y: number }
export type Centerline =
  | { kind: "segment"; a: Pt; b: Pt }
  | { kind: "arc"; center: Pt; radius: number; start_deg: number; end_deg: number };
export interface PlanWall { id: string; level: string; centerline: Centerline; thickness_m: Fact<number>; height_m: Fact<number>; kind: string }
export interface PlanOpening {
  id: string; host_wall: string; offset_m: Fact<number>; width_m: Fact<number>; height_m: Fact<number>; sill_m: Fact<number>;
  type: string; hinge: string | null; swing_side: string | null;
}
export interface PlanRoom { id: string; level: string; polygon: Pt[]; holes: Pt[][]; name: Fact<string>; area_label_m2: Fact<number> | null }
export interface Issue { code: string; severity: string; message: string; fix_hint: string; element_ids: string[]; location: Pt | null }
export interface PlanConflict { key: string; candidates: Record<string, unknown>[]; proposed: number; rule: string; resolution: number | null }
export interface Plan {
  version: string; source: string; levels: { id: string; name: string; elevation_m: number }[];
  walls: PlanWall[]; openings: PlanOpening[]; rooms: PlanRoom[]; issues: Issue[]; conflicts: PlanConflict[];
}
export interface Suggestion { id: string; page: string; kind: string; hint_plan: number[][]; reason: string; decision: string | null }
export interface Background { page_id: string; image: CasRef; width_px: number; height_px: number; matrix: number[][] }
export interface PlanVersionSummary {
  id: string; number: number; parent_id: string | null; root_id: string; status: string; origin: string;
  issues: Issue[]; blocking: number; created_at: string; approved_by: string | null;
  extraction: { sources?: { page: string; source: string; level: string; notes: string[]; assist?: { triggers: { kind: string; detail: string }[]; source: string | null; accepted: number } | null }[] };
}
export interface PlanVersion extends PlanVersionSummary { plan: Plan; suggestions: Suggestion[]; backgrounds: Background[] }
