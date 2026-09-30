import { useCallback, useEffect, useState } from "react";
import { api, post, type Doc, type Job, type Project, type Run } from "../api";
import { ErrorBox, navigate } from "../App";
import { Pages } from "../components/Pages";
import { uploadFile } from "../upload";

interface FileProgress { name: string; phase: string; fraction: number; result?: string; error?: unknown }

async function waitJob(jobId: string): Promise<Job> {
  for (;;) {
    const job = await api<Job>(`jobs/${jobId}`);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) return job;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

export function ProjectPage({ projectId }: { projectId: string }) {
  const [project, setProject] = useState<Project | null>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [files, setFiles] = useState<FileProgress[]>([]);
  const [err, setErr] = useState<unknown>(null);
  const [views, setViews] = useState(3);
  const [width, setWidth] = useState(3840);
  const [height, setHeight] = useState(2160);
  const [samples, setSamples] = useState("");
  const [policy, setPolicy] = useState("on_low_confidence");
  const [refresh, setRefresh] = useState(0);

  const load = useCallback(() => {
    api<Project>(`projects/${projectId}`).then(setProject).catch(setErr);
    api<Doc[]>(`projects/${projectId}/documents`).then(setDocs).catch(setErr);
    api<Run[]>(`projects/${projectId}/runs`).then(setRuns).catch(setErr);
  }, [projectId]);
  useEffect(load, [load]);

  const onFiles = async (list: FileList | null) => {
    if (!list) return;
    const arr = Array.from(list);
    setFiles(arr.map((f) => ({ name: f.name, phase: "queued", fraction: 0 })));
    for (const [i, f] of arr.entries()) {
      const update = (patch: Partial<FileProgress>) =>
        setFiles((prev) => prev.map((p, j) => (j === i ? { ...p, ...patch } : p)));
      try {
        const jobId = await uploadFile(projectId, f, (phase, fraction) => update({ phase, fraction }));
        update({ phase: "processing", fraction: 1 });
        const job = await waitJob(jobId);
        if (job.status === "succeeded") update({ phase: "done", result: "ingested" });
        else update({ phase: "failed", error: job.error?.message, result: job.error?.fix_hint ?? "" });
      } catch (x) {
        update({ phase: "failed", error: x });
      }
    }
    load();
    setRefresh((n) => n + 1);
  };

  const startRun = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    try {
      const body: Record<string, unknown> = { views, width, height, gate_policy: policy };
      if (samples) body.samples = Number(samples);
      const res = await post<{ run_id: string }>(`projects/${projectId}/runs`, body);
      navigate(`runs/${res.run_id}`);
    } catch (x) {
      setErr(x);
    }
  };

  return (
    <section>
      <p><a href="#/projects">← Projects</a></p>
      <h1>{project?.name ?? "…"}</h1>
      <ErrorBox error={err} />
      <div className="grid2">
        <div className="card">
          <h2>Documents</h2>
          <label className="drop">
            Drop or choose plans, CAD/BIM, photos, schedules, briefs (resumable upload)
            <input type="file" multiple onChange={(e) => onFiles(e.target.files)} data-testid="file-input" />
          </label>
          {files.map((f) => (
            <div key={f.name} className="progress-row">
              <span>{f.name}</span>
              <progress max={1} value={f.fraction} />
              <span className={f.phase === "failed" ? "bad" : "muted"}>{f.phase}</span>
              {f.error ? <ErrorBox error={f.error} /> : null}
            </div>
          ))}
          <ul className="list">
            {docs.map((d) => (
              <li key={d.id}>{d.filename} <span className="muted">· {d.kind} · {(d.size / 1024).toFixed(1)} KB</span></li>
            ))}
            {docs.length === 0 && <li className="muted">No documents yet.</li>}
          </ul>
        </div>
        <div className="card">
          <h2>New run</h2>
          <form onSubmit={startRun} className="stack">
            <label>Views<input type="number" min={1} max={12} value={views} onChange={(e) => setViews(Number(e.target.value))} name="views" /></label>
            <label>Width<input type="number" value={width} onChange={(e) => setWidth(Number(e.target.value))} name="width" /></label>
            <label>Height<input type="number" value={height} onChange={(e) => setHeight(Number(e.target.value))} name="height" /></label>
            <label>Samples (blank = profile default)<input value={samples} onChange={(e) => setSamples(e.target.value)} name="samples" /></label>
            <label>Review policy
              <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                <option value="on_low_confidence">on_low_confidence</option>
                <option value="always">always</option>
                <option value="never">never</option>
              </select>
            </label>
            <button type="submit" disabled={docs.length === 0}>Start run</button>
          </form>
        </div>
      </div>
      <Pages projectId={projectId} refresh={refresh} />
      <h2>Runs</h2>
      <ul className="list">
        {runs.map((r) => (
          <li key={r.id}><a href={`#/runs/${r.id}`}>{r.id}</a> <span className={`badge ${r.status}`}>{r.status}</span> <span className="muted">{new Date(r.created_at).toLocaleString()}</span></li>
        ))}
        {runs.length === 0 && <li className="muted">No runs yet.</li>}
      </ul>
    </section>
  );
}
