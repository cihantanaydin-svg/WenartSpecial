import { useCallback, useEffect, useRef, useState } from "react";
import { api, blobUrl, PAGE_CLASSES, type Job, type PageInfo, type ReviewItem } from "../api";
import { ErrorBox } from "../App";

// Pages with their S1 classification: previews, confidence, review flags and class correction.
// While the project's S1 job is queued or running the panel polls it and reloads when it ends.
export function Pages({ projectId, refresh }: { projectId: string; refresh: number }) {
  const [pages, setPages] = useState<PageInfo[]>([]);
  const [review, setReview] = useState<ReviewItem[]>([]);
  const [s1, setS1] = useState<Job | null>(null);
  const [err, setErr] = useState<unknown>(null);

  const load = useCallback(() => {
    api<PageInfo[]>(`projects/${projectId}/pages`).then(setPages).catch(setErr);
    api<ReviewItem[]>(`projects/${projectId}/review`).then(setReview).catch(setErr);
    api<Job | null>(`projects/${projectId}/understand`).then(setS1).catch(setErr);
  }, [projectId]);
  useEffect(load, [load, refresh]);

  const busy = s1 !== null && (s1.status === "queued" || s1.status === "running");
  useEffect(() => {
    if (!busy) return;
    const t = setTimeout(load, 2000);
    return () => clearTimeout(t);
  }, [busy, load, s1]);
  const wasBusy = useRef(false);
  useEffect(() => {
    if (wasBusy.current && !busy) load(); // pages may have been fetched just before the job ended
    wasBusy.current = busy;
  }, [busy, load]);

  const setClass = async (pageId: string, label: string) => {
    setErr(null);
    try {
      await api(`projects/${projectId}/pages/${pageId}/class`, { method: "PUT", body: JSON.stringify({ label }) });
      load();
    } catch (x) {
      setErr(x);
    }
  };

  if (pages.length === 0) return null;
  return (
    <div className="card" data-testid="pages">
      <h2>Pages {review.length > 0 && <span className="badge waiting_gate">{review.length} to review</span>}</h2>
      {busy && s1 && <p className="muted small" data-testid="s1-progress">Analysing pages… {Math.round(s1.progress * 100)}%</p>}
      {s1?.status === "failed" && s1.error && (
        <div className="error" role="alert">
          <strong>{s1.error.code}</strong>: page analysis failed: {s1.error.message}
          {s1.error.fix_hint && <div className="hint">→ {s1.error.fix_hint}</div>}
        </div>
      )}
      <ErrorBox error={err} />
      <div className="pages">
        {pages.map((p) => (
          <figure key={p.page_id} className={`page-card${p.needs_review ? " review" : ""}`} data-testid={`page-${p.page_id}`}>
            {p.preview ? <img src={blobUrl(projectId, p.preview.sha256)} alt={`${p.filename} p${p.index + 1}`} loading="lazy" /> : <div className="noimg">{p.kind}</div>}
            <figcaption>
              <div className="muted small">{p.filename}{p.index > 0 ? ` · p${p.index + 1}` : ""}</div>
              {p.analysed ? (
                <>
                  <select value={p.label ?? ""} onChange={(e) => setClass(p.page_id, e.target.value)} aria-label="page class">
                    {PAGE_CLASSES.map((c) => <option key={c} value={c}>{c.replace("_", " ")}</option>)}
                  </select>
                  <span className="muted small">
                    {p.overridden ? ` you set this (model: ${p.model_label})` : ` ${Math.round((p.confidence ?? 0) * 100)}%`}
                    {p.needs_review ? " · check" : ""}
                  </span>
                  {p.scale ? <div className="small">scale 1:{p.scale}</div> : null}
                  {p.north_deg !== null ? <div className="small">north {p.north_deg.toFixed(1)}°</div> : null}
                  {p.tags ? <div className="small">{p.tags} tags</div> : null}
                </>
              ) : (
                <span className="muted small">analysing…</span>
              )}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
