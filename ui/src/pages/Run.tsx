import { useCallback, useEffect, useRef, useState } from "react";
import { api, blobUrl, post, type Check, type Run, type ViewResult } from "../api";
import { ErrorBox } from "../App";
import { Compare } from "../components/Compare";
import { GlbViewer } from "../components/GlbViewer";

interface Progress { fraction: number; message: string; status: string }

function useJobEvents(jobId: string | undefined, onChange: () => void): Progress {
  const [p, setP] = useState<Progress>({ fraction: 0, message: "", status: "" });
  const cb = useRef(onChange);
  cb.current = onChange;
  useEffect(() => {
    if (!jobId) return;
    // EventSource reconnects automatically and sends Last-Event-ID; cookie auth applies.
    const es = new EventSource(`api/v1/jobs/${jobId}/events`);
    const onProgress = (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      setP((prev) => ({ ...prev, fraction: d.progress ?? prev.fraction, message: d.message || d.stage || prev.message }));
    };
    const onStatus = (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      setP((prev) => ({ ...prev, status: d.status }));
      cb.current();
    };
    es.addEventListener("progress", onProgress);
    es.addEventListener("status", onStatus);
    es.addEventListener("gate", () => cb.current());
    es.addEventListener("end", () => {
      es.close();
      cb.current();
    });
    return () => es.close();
  }, [jobId]);
  return p;
}

function fmt(v: number | null): string {
  return v === null || v === undefined ? "" : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(3);
}

function ChecksTable({ checks }: { checks: Check[] }) {
  if (!checks.length) return <p className="muted">Delivered image is the unrefined Cycles render (ground truth).</p>;
  return (
    <div className="table-wrap">
    <table className="checks">
      <thead><tr><th>Check</th><th>Result</th><th>Value</th><th>Base</th><th>Δ</th><th>Limit</th><th>Estimator</th></tr></thead>
      <tbody>
        {checks.map((c) => (
          <tr key={c.name} className={c.passed ? "" : "failrow"}>
            <td>{c.name}<span className="muted"> · {c.family}</span></td>
            <td className={c.passed ? "ok" : "bad"}>{c.passed ? "pass" : "fail"}</td>
            <td>{fmt(c.value)}</td><td>{fmt(c.base_value)}</td><td>{fmt(c.delta)}</td>
            <td>{c.comparator} {c.threshold ?? ""}</td>
            <td>{c.estimator}{c.mock && <span className="warn"> (mock)</span>}</td>
          </tr>
        ))}
      </tbody>
    </table>
    </div>
  );
}

function ViewCard({ projectId, v }: { projectId: string; v: ViewResult }) {
  const [open, setOpen] = useState(false);
  const label = { refined: "QA passed", hard_composite: "hard composite", fallback_base: "Cycles fallback" }[v.status] ?? v.status;
  const cls = v.status === "refined" ? "ok" : "warn";
  return (
    <figure className={`card view${open ? " open" : ""}`} data-testid={`view-${v.view_id}`}>
      <Compare before={blobUrl(projectId, v.base.sha256)} after={blobUrl(projectId, v.delivered_jpg.sha256)} />
      <figcaption>
        <strong>{v.view_id}</strong> <span className={`badge ${cls}`}>{label}</span>
        {v.uses_mocks && <span className="badge warn">mock estimators</span>}
        <div className="muted">{v.reason} · {v.attempts} attempt(s)</div>
        <button className="link" onClick={() => setOpen(!open)}>{open ? "Hide" : "Why / QA details"}</button>
        {open && <ChecksTable checks={v.checks} />}
      </figcaption>
    </figure>
  );
}

export function RunPage({ runId }: { runId: string }) {
  const [run, setRun] = useState<Run | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [notes, setNotes] = useState("");
  const load = useCallback(() => {
    api<Run>(`runs/${runId}`).then(setRun).catch(setErr);
  }, [runId]);
  useEffect(load, [load]);
  const progress = useJobEvents(run?.job_id, load);

  const decide = async (gate: string, approve: boolean) => {
    setErr(null);
    try {
      setRun(await post<Run>(`runs/${runId}/gates/${gate}`, { approve, notes: notes || null }));
      setNotes("");
    } catch (x) {
      setErr(x);
    }
  };

  if (!run) return <section><ErrorBox error={err} /><p className="muted">Loading…</p></section>;
  const pending = run.gates.filter((g) => g.status === "pending");
  const running = !["succeeded", "failed", "cancelled"].includes(run.status);
  return (
    <section>
      <p><a href={`#/projects/${run.project_id}`}>← Project</a></p>
      <h1>Run {run.id} <span className={`badge ${run.status}`} data-testid="run-status">{run.status}</span></h1>
      <ErrorBox error={err} />
      {running && (
        <div className="card">
          <progress max={1} value={progress.fraction} style={{ width: "100%" }} />
          <div className="muted">{progress.message || "queued"}</div>
        </div>
      )}
      {pending.map((g) => (
        <div key={g.gate} className="card gate" data-testid={`gate-${g.gate}`}>
          <h2>Review needed: {g.gate}</h2>
          <pre className="evidence">{JSON.stringify(g.evidence, null, 2)}</pre>
          <label>Notes<input value={notes} onChange={(e) => setNotes(e.target.value)} name="notes" /></label>
          <div className="row">
            <button onClick={() => decide(g.gate, true)}>Approve</button>
            <button className="danger" onClick={() => decide(g.gate, false)}>Reject</button>
          </div>
        </div>
      ))}
      <h2>Gates</h2>
      <ul className="list">
        {run.gates.map((g) => (
          <li key={g.gate}>{g.gate}: <span className={`badge ${g.status}`}>{g.status}</span> <span className="muted">({g.policy}){g.decided_by ? ` by ${g.decided_by}` : ""}{g.notes ? `: ${g.notes}` : ""}</span></li>
        ))}
      </ul>
      {run.result && (
        <>
          {run.result.uses_mocks && (
            <div className="banner">Mock models were used: a pipeline test, not a client deliverable.</div>
          )}
          <h2>Views</h2>
          <div className="views">
            {run.result.views.map((v) => <ViewCard key={v.view_id} projectId={run.project_id} v={v} />)}
          </div>
          <h2>3D model</h2>
          <GlbViewer url={blobUrl(run.project_id, run.result.glb.sha256)} />
          <h2>Deliverables</h2>
          <p className="row">
            {run.bundle && <a className="button" href={`api/v1/bundles/${run.bundle.id}/download`} data-testid="download">Download bundle</a>}
            <a className="button secondary" href={blobUrl(run.project_id, run.result.report.sha256)} target="_blank" rel="noreferrer">QA report</a>
          </p>
          <h2>Assumptions</h2>
          <table className="checks">
            <tbody>
              {run.result.assumptions.map((a) => (
                <tr key={`${a.stage}-${a.key}`}><td>{a.stage}</td><td>{a.key}</td><td><code>{String(a.value)}</code></td><td className="muted">{a.reason}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
