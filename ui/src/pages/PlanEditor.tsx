import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api, blobUrl, post, type Fact, type Plan, type PlanVersion, type PlanVersionSummary, type PlanWall, type Pt,
} from "../api";
import { ErrorBox } from "../App";

// Gate A plan editor: the plan over its source page, issues, VLM-assisted elements, suggestions,
// conflicts; edits are RFC 6902 operations applied locally and saved as one new draft version.

type Op = { op: "add" | "remove" | "replace"; path: string; value?: unknown };
type Sel = { kind: "wall" | "opening" | "room"; id: string } | null;
type Tool = "select" | "wall" | "calibrate";

function getAt(doc: any, path: string): [any, string] {
  const toks = path.slice(1).split("/").map((t) => t.replace(/~1/g, "/").replace(/~0/g, "~"));
  let cur = doc;
  for (const t of toks.slice(0, -1)) cur = Array.isArray(cur) ? cur[Number(t)] : cur[t];
  return [cur, toks[toks.length - 1]];
}

function applyOp(doc: any, op: Op): void {
  const [parent, key] = getAt(doc, op.path);
  if (op.op === "add") {
    if (Array.isArray(parent)) parent.splice(key === "-" ? parent.length : Number(key), 0, op.value);
    else parent[key] = op.value;
  } else if (op.op === "remove") {
    if (Array.isArray(parent)) parent.splice(Number(key), 1);
    else delete parent[key];
  } else {
    if (Array.isArray(parent)) parent[Number(key)] = op.value;
    else parent[key] = op.value;
  }
}

const userFact = <T,>(value: T): Fact<T> => ({ value, provenance: [{ method: "user", confidence: 1 }], status: "user_confirmed" });
const assisted = (f: Fact<unknown>) => f.provenance.some((p) => p.method === "vlm_assisted" && p.assist && !p.assist.user_confirmed);

function wallPoly(w: PlanWall): Pt[] {
  const t = w.thickness_m.value / 2;
  const cl = w.centerline;
  if (cl.kind === "segment") {
    const dx = cl.b.x - cl.a.x, dy = cl.b.y - cl.a.y, L = Math.hypot(dx, dy) || 1;
    const nx = (-dy / L) * t, ny = (dx / L) * t;
    return [
      { x: cl.a.x + nx, y: cl.a.y + ny }, { x: cl.b.x + nx, y: cl.b.y + ny },
      { x: cl.b.x - nx, y: cl.b.y - ny }, { x: cl.a.x - nx, y: cl.a.y - ny },
    ];
  }
  const sweep = ((cl.end_deg - cl.start_deg) % 360 + 360) % 360 || 360;
  const n = Math.max(8, Math.ceil(sweep / 4));
  const ring = (r: number) => Array.from({ length: n + 1 }, (_, i) => {
    const a = ((cl.start_deg + (sweep * i) / n) * Math.PI) / 180;
    return { x: cl.center.x + r * Math.cos(a), y: cl.center.y + r * Math.sin(a) };
  });
  return [...ring(cl.radius + t), ...ring(cl.radius - t).reverse()];
}

function openingQuad(plan: Plan, oi: number): Pt[] | null {
  const o = plan.openings[oi];
  const w = plan.walls.find((x) => x.id === o.host_wall);
  if (!w || w.centerline.kind !== "segment") return null;
  const { a, b } = w.centerline;
  const L = Math.hypot(b.x - a.x, b.y - a.y) || 1;
  const ux = (b.x - a.x) / L, uy = (b.y - a.y) / L, t = w.thickness_m.value / 2 + 0.01;
  const c0 = o.offset_m.value - o.width_m.value / 2, c1 = o.offset_m.value + o.width_m.value / 2;
  const p = (s: number, k: number) => ({ x: a.x + ux * s - uy * k, y: a.y + uy * s + ux * k });
  return [p(c0, t), p(c1, t), p(c1, -t), p(c0, -t)];
}

const pts = (ps: Pt[]) => ps.map((p) => `${p.x},${-p.y}`).join(" ");
const centroid = (ps: Pt[]) => ({ x: ps.reduce((s, p) => s + p.x, 0) / ps.length, y: ps.reduce((s, p) => s + p.y, 0) / ps.length });

export function PlanEditor({ projectId }: { projectId: string }) {
  const [versions, setVersions] = useState<PlanVersionSummary[]>([]);
  const [vid, setVid] = useState<string | null>(null);
  const [version, setVersion] = useState<PlanVersion | null>(null);
  const [ops, setOps] = useState<Op[]>([]);
  const [sel, setSel] = useState<Sel>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [clicks, setClicks] = useState<Pt[]>([]);
  const [note, setNote] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [drag, setDrag] = useState<{ wall: number; end: "a" | "b" } | null>(null);
  const [dragPt, setDragPt] = useState<Pt | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const justDragged = useRef(false);

  const loadList = useCallback(async (select?: string) => {
    try {
      const res = await api<{ versions: PlanVersionSummary[] }>(`projects/${projectId}/plans`);
      setVersions(res.versions);
      setVid((cur) => select ?? cur ?? res.versions[0]?.id ?? null);
    } catch (x) {
      setErr(x);
    }
  }, [projectId]);
  useEffect(() => { loadList(); }, [loadList]);
  useEffect(() => {
    if (!vid) return;
    api<PlanVersion>(`projects/${projectId}/plans/${vid}`).then((v) => {
      setVersion(v); setOps([]); setSel(null); setClicks([]);
    }).catch(setErr);
  }, [projectId, vid]);

  // the plan with the pending edits applied
  const plan: Plan | null = useMemo(() => {
    if (!version) return null;
    const d = structuredClone(version.plan);
    for (const op of ops) applyOp(d, op);
    return d;
  }, [version, ops]);

  const bounds = useMemo(() => {
    if (!version) return null;
    const ps = version.plan.walls.flatMap(wallPoly);
    if (!ps.length) return { x: -1, y: -1, w: 2, h: 2 };
    const xs = ps.map((p) => p.x), ys = ps.map((p) => -p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const m = 0.08 * Math.max(x1 - x0, y1 - y0);
    return { x: x0 - m, y: y0 - m, w: x1 - x0 + 2 * m, h: y1 - y0 + 2 * m };
  }, [version]);
  useEffect(() => { setView(bounds); }, [bounds]);

  const toPlan = (e: { clientX: number; clientY: number }): Pt | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return { x: p.x, y: -p.y };
  };
  // snap to a wall end (12 cm), else onto a wall's centre line when the point is on or by the wall
  const snap = (p: Pt): Pt => {
    if (!plan) return p;
    let best: Pt | null = null, bd = 0.12;
    for (const w of plan.walls) {
      if (w.centerline.kind !== "segment") continue;
      for (const q of [w.centerline.a, w.centerline.b]) {
        const d = Math.hypot(q.x - p.x, q.y - p.y);
        if (d < bd) { bd = d; best = q; }
      }
    }
    if (best) return { x: best.x, y: best.y };
    let line: Pt | null = null, ld = Infinity;
    for (const w of plan.walls) {
      if (w.centerline.kind !== "segment") continue;
      const { a, b } = w.centerline;
      const dx = b.x - a.x, dy = b.y - a.y, L2 = dx * dx + dy * dy || 1;
      const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / L2));
      const q = { x: a.x + t * dx, y: a.y + t * dy };
      const d = Math.hypot(q.x - p.x, q.y - p.y);
      if (d <= w.thickness_m.value / 2 + 0.1 && d < ld) { ld = d; line = q; }
    }
    return line ?? p;
  };

  const push = (...more: Op[]) => setOps((o) => [...o, ...more]);
  const wallsEdited = ops.some((o) => o.path.startsWith("/walls"));

  const run = async (fn: () => Promise<PlanVersion | void>) => {
    setErr(null); setBusy(true);
    try {
      const v = await fn();
      if (v) { await loadList(v.id); setVid(v.id); setVersion(v); setOps([]); setSel(null); }
    } catch (x) {
      setErr(x);
    } finally {
      setBusy(false);
    }
  };
  const save = () => run(() => post<PlanVersion>(`projects/${projectId}/plans/${vid}/edits`, { ops, note, rederive_rooms: wallsEdited }));
  const approve = () => run(async () => {
    await post(`projects/${projectId}/plans/${vid}/approve`);
    await loadList(vid!);
    setVersion(await api<PlanVersion>(`projects/${projectId}/plans/${vid}`));
  });

  const onBackgroundClick = (e: React.MouseEvent) => {
    if (justDragged.current) { justDragged.current = false; return; }
    const p = toPlan(e);
    if (!p || !plan) return;
    if (tool === "select") { setSel(null); return; }
    const q = snap(p);
    const next = [...clicks, q];
    if (next.length < 2) { setClicks(next); return; }
    setClicks([]);
    if (tool === "wall") {
      const ids = new Set(plan.walls.map((w) => w.id));
      let n = plan.walls.length + 1;
      while (ids.has(`WE${n}`)) n++;
      const thick = plan.walls.filter((w) => w.kind !== "exterior").map((w) => w.thickness_m.value);
      const t = thick.length ? thick.sort((a, b) => a - b)[Math.floor(thick.length / 2)] : 0.12;
      push({
        op: "add", path: "/walls/-",
        value: {
          id: `WE${n}`, level: plan.levels[0].id, centerline: { kind: "segment", a: next[0], b: next[1] },
          thickness_m: userFact(t), height_m: plan.walls[0]?.height_m ?? userFact(2.7), kind: "interior",
        },
      });
      setTool("select");
    } else if (tool === "calibrate") {
      const len = window.prompt("Real length between the two points (metres):");
      setTool("select");
      if (!len) return;
      run(() => post<PlanVersion>(`projects/${projectId}/plans/${vid}/calibrate`, {
        a: [next[0].x, next[0].y], b: [next[1].x, next[1].y], length_m: Number(len.replace(",", ".")),
      }));
    }
  };

  const onMove = (e: React.MouseEvent) => { if (drag) setDragPt(snap(toPlan(e) ?? { x: 0, y: 0 })); };
  const onUp = () => {
    if (drag && dragPt) {
      push({ op: "replace", path: `/walls/${drag.wall}/centerline/${drag.end}`, value: dragPt });
      justDragged.current = true;
    }
    setDrag(null); setDragPt(null);
  };
  const onWheel = (e: React.WheelEvent) => {
    if (!view) return;
    const p = toPlan(e);
    if (!p) return;
    const k = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    const cx = p.x, cy = -p.y;
    setView({ x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k, w: view.w * k, h: view.h * k });
  };

  const deleteSel = () => {
    if (!sel || !plan) return;
    if (sel.kind === "wall") {
      const hosted = plan.openings.map((o, i) => [o, i] as const).filter(([o]) => o.host_wall === sel.id).map(([, i]) => i);
      const del: Op[] = hosted.sort((a, b) => b - a).map((i) => ({ op: "remove", path: `/openings/${i}` }));
      del.push({ op: "remove", path: `/walls/${plan.walls.findIndex((w) => w.id === sel.id)}` });
      push(...del);
    } else {
      const arr = sel.kind === "opening" ? plan.openings : plan.rooms;
      push({ op: "remove", path: `/${sel.kind === "opening" ? "openings" : "rooms"}/${arr.findIndex((x) => x.id === sel.id)}` });
    }
    setSel(null);
  };

  if (!versions.length) {
    return (
      <section>
        <p><a href={`#/projects/${projectId}`}>← Project</a></p>
        <h1>Plan (Gate A)</h1>
        <ErrorBox error={err} />
        <p className="muted">No plan yet: upload a floor plan (DXF, IFC, PDF or a scan); it is extracted after page analysis.</p>
      </section>
    );
  }
  const unconfirmed = plan ? [
    ...plan.walls.filter((w) => assisted(w.thickness_m)).map((w) => w.id),
    ...plan.openings.filter((o) => assisted(o.width_m)).map((o) => o.id),
  ] : [];
  const issueIds = new Set((plan?.issues ?? []).flatMap((i) => i.element_ids));
  const editable = version?.status !== "superseded";
  const selWall = sel?.kind === "wall" ? plan?.walls.findIndex((w) => w.id === sel.id) ?? -1 : -1;
  const selOpening = sel?.kind === "opening" ? plan?.openings.findIndex((o) => o.id === sel.id) ?? -1 : -1;
  const selRoom = sel?.kind === "room" ? plan?.rooms.findIndex((r) => r.id === sel.id) ?? -1 : -1;
  const sw = view ? view.w / 400 : 0.02;

  return (
    <section>
      <p><a href={`#/projects/${projectId}`}>← Project</a></p>
      <h1>Plan (Gate A)</h1>
      <ErrorBox error={err} />
      <div className="row">
        <label>Version
          <select value={vid ?? ""} onChange={(e) => setVid(e.target.value)} data-testid="plan-version">
            {versions.map((v) => (
              <option key={v.id} value={v.id}>v{v.number} · {v.status} · {v.origin}{v.blocking ? ` · ${v.blocking} blocking` : ""}</option>
            ))}
          </select>
        </label>
        {version && <span className={`badge ${version.status}`} data-testid="plan-status">{version.status}</span>}
        <span className="muted small">{plan?.source} · {plan?.walls.length} walls · {plan?.openings.length} openings · {plan?.rooms.length} rooms</span>
      </div>
      <div className="editor">
        <div className="canvas card">
          <div className="row toolbar">
            {(["select", "wall", "calibrate"] as Tool[]).map((t) => (
              <button key={t} className={tool === t ? "" : "button secondary"} onClick={() => { setTool(t); setClicks([]); }} disabled={!editable || (t === "calibrate" && ops.length > 0)}>
                {t === "select" ? "Select" : t === "wall" ? "Add wall" : "Calibrate scale"}
              </button>
            ))}
            <button className="button secondary" onClick={() => setView(bounds)}>Fit</button>
            {tool !== "select" && <span className="muted small">click {2 - clicks.length} point(s)</span>}
          </div>
          {view && plan && (
            <svg ref={svgRef} viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`} className="plan-svg" data-testid="plan-svg"
              onWheel={onWheel} onMouseMove={onMove} onMouseUp={onUp} onClick={onBackgroundClick}>
              {version?.backgrounds.map((b) => {
                const [[a, bb, tx], [c, d, ty]] = b.matrix;
                return <image key={b.page_id} href={blobUrl(projectId, b.image.sha256)} width={b.width_px} height={b.height_px}
                  transform={`matrix(${a} ${-c} ${bb} ${-d} ${tx} ${-ty})`} opacity={0.45} preserveAspectRatio="none" />;
              })}
              {plan.rooms.map((r) => {
                const c = centroid(r.polygon);
                return (
                  <g key={r.id} onClick={(e) => { if (tool === "select") { e.stopPropagation(); setSel({ kind: "room", id: r.id }); } }}>
                    <polygon points={pts(r.polygon)} className={`room${sel?.id === r.id ? " sel" : ""}${issueIds.has(r.id) ? " issue" : ""}`} data-testid={`room-${r.id}`} />
                    <text x={c.x} y={-c.y} fontSize={view.w / 60} textAnchor="middle" className="label">{r.name.value}</text>
                  </g>
                );
              })}
              {plan.walls.map((w) => (
                <polygon key={w.id} points={pts(wallPoly(w))} strokeWidth={sw}
                  className={`wall${sel?.id === w.id ? " sel" : ""}${assisted(w.thickness_m) ? " assisted" : ""}${issueIds.has(w.id) ? " issue" : ""}`}
                  data-testid={`wall-${w.id}`}
                  onClick={(e) => { if (tool === "select") { e.stopPropagation(); setSel({ kind: "wall", id: w.id }); } }} />
              ))}
              {plan.openings.map((o, i) => {
                const q = openingQuad(plan, i);
                return q && (
                  <polygon key={o.id} points={pts(q)} strokeWidth={sw}
                    className={`opening ${o.type}${sel?.id === o.id ? " sel" : ""}${assisted(o.width_m) ? " assisted" : ""}${issueIds.has(o.id) ? " issue" : ""}`}
                    data-testid={`opening-${o.id}`}
                    onClick={(e) => { if (tool === "select") { e.stopPropagation(); setSel({ kind: "opening", id: o.id }); } }} />
                );
              })}
              {version?.suggestions.filter((s) => !s.decision).map((s) => (
                <line key={s.id} x1={s.hint_plan[0][0]} y1={-s.hint_plan[0][1]} x2={s.hint_plan[1][0]} y2={-s.hint_plan[1][1]}
                  className="suggestion" strokeWidth={sw * 2} />
              ))}
              {selWall >= 0 && plan.walls[selWall].centerline.kind === "segment" && editable && (["a", "b"] as const).map((end) => {
                const cl = plan.walls[selWall].centerline as { a: Pt; b: Pt };
                const p = drag?.wall === selWall && drag.end === end && dragPt ? dragPt : cl[end];
                return <circle key={end} cx={p.x} cy={-p.y} r={sw * 5} className="handle" data-testid={`handle-${end}`}
                  onMouseDown={(e) => { e.stopPropagation(); setDrag({ wall: selWall, end }); }} />;
              })}
              {clicks.map((p, i) => <circle key={i} cx={p.x} cy={-p.y} r={sw * 4} className="handle" />)}
            </svg>
          )}
          <p className="muted small">Wheel to zoom. Orange dashed: measured from VLM hints, confirm or delete. Purple dashed: VLM suggestions without drawing evidence.</p>
        </div>
        <div className="side">
          {plan && sel && (
            <div className="card" data-testid="inspector">
              <h2>{sel.kind} {sel.id}</h2>
              {selWall >= 0 && (
                <label>Thickness (m)
                  <input type="number" step="0.01" defaultValue={plan.walls[selWall].thickness_m.value} key={`t${sel.id}${ops.length}`}
                    onBlur={(e) => push({ op: "replace", path: `/walls/${selWall}/thickness_m/value`, value: Number(e.target.value) })} />
                </label>
              )}
              {selOpening >= 0 && (
                <>
                  <label>Type
                    <select value={plan.openings[selOpening].type} onChange={(e) => push({ op: "replace", path: `/openings/${selOpening}/type`, value: e.target.value })}>
                      {["door", "double_door", "sliding_door", "french_door", "window", "opening", "pass"].map((t) => <option key={t}>{t}</option>)}
                    </select>
                  </label>
                  <label>Width (m)
                    <input type="number" step="0.01" defaultValue={plan.openings[selOpening].width_m.value} key={`w${sel.id}${ops.length}`}
                      onBlur={(e) => push({ op: "replace", path: `/openings/${selOpening}/width_m/value`, value: Number(e.target.value) })} />
                  </label>
                  <label>Offset along the wall (m)
                    <input type="number" step="0.01" defaultValue={plan.openings[selOpening].offset_m.value} key={`o${sel.id}${ops.length}`}
                      onBlur={(e) => push({ op: "replace", path: `/openings/${selOpening}/offset_m/value`, value: Number(e.target.value) })} />
                  </label>
                </>
              )}
              {selRoom >= 0 && (
                <label>Name
                  <input name="room_name" defaultValue={plan.rooms[selRoom].name.value} key={`n${sel.id}${ops.length}`}
                    onBlur={(e) => e.target.value !== plan.rooms[selRoom].name.value && push({ op: "replace", path: `/rooms/${selRoom}/name/value`, value: e.target.value })} />
                </label>
              )}
              {unconfirmed.includes(sel.id) && (
                <button disabled={busy || ops.length > 0} onClick={() => run(() => post<PlanVersion>(`projects/${projectId}/plans/${vid}/assists/confirm`, { element_ids: [sel.id] }))}>
                  Confirm (matches the drawing)
                </button>
              )}
              {editable && <button className="danger" onClick={deleteSel}>Delete {sel.kind}</button>}
            </div>
          )}
          <div className="card">
            <h2>Changes</h2>
            {ops.length === 0 ? <p className="muted small">No pending edits.</p> : <p className="small" data-testid="pending">{ops.length} pending edit(s){wallsEdited ? " · rooms are re-derived from the walls" : ""}</p>}
            <label>Note<input value={note} onChange={(e) => setNote(e.target.value)} name="edit_note" /></label>
            <div className="row">
              <button onClick={save} disabled={busy || ops.length === 0}>Save as new version</button>
              <button className="button secondary" onClick={() => setOps([])} disabled={ops.length === 0}>Discard</button>
            </div>
            <div className="row">
              <button onClick={approve} disabled={busy || ops.length > 0 || !version || version.status === "approved" || version.blocking > 0} data-testid="approve-plan">
                Approve plan
              </button>
              {unconfirmed.length > 1 && (
                <button className="button secondary" disabled={busy || ops.length > 0}
                  onClick={() => run(() => post<PlanVersion>(`projects/${projectId}/plans/${vid}/assists/confirm`, { element_ids: unconfirmed }))}>
                  Confirm all {unconfirmed.length} assisted
                </button>
              )}
            </div>
          </div>
          <div className="card">
            <h2>Issues {plan && plan.issues.length > 0 && <span className="badge warn">{plan.issues.length}</span>}</h2>
            <ul className="list small" data-testid="issues">
              {plan?.issues.map((i, k) => (
                <li key={k} className={i.severity === "error" || i.severity === "blocker" ? "bad" : "warn"}>
                  <button className="link" onClick={() => {
                    const id = i.element_ids[0];
                    if (!id || !plan) return;
                    const kind = plan.walls.some((w) => w.id === id) ? "wall" : plan.openings.some((o) => o.id === id) ? "opening" : "room";
                    setSel({ kind, id });
                  }}>{i.code}</button>: {i.message}
                  <div className="muted">→ {i.fix_hint}</div>
                </li>
              ))}
              {plan && plan.issues.length === 0 && <li className="muted">None.</li>}
            </ul>
            {ops.length > 0 && <p className="muted small">Issues are re-checked when the edits are saved.</p>}
          </div>
          {plan && plan.conflicts.some((c) => c.resolution === null) && (
            <div className="card">
              <h2>Conflicts</h2>
              {plan.conflicts.filter((c) => c.resolution === null).map((c) => (
                <div key={c.key} className="small">
                  <strong>{c.key}</strong> <span className="muted">{c.rule}</span>
                  {c.candidates.map((cand, k) => (
                    <div key={k} className="row">
                      <span>{k === c.proposed ? "★ " : ""}{String(cand.method ?? "")} {String(cand.m_per_unit ?? cand.value ?? "")}</span>
                      <button className="link" disabled={busy || ops.length > 0} onClick={() => run(() => post<PlanVersion>(`projects/${projectId}/plans/${vid}/resolve`, { key: c.key, choice: k }))}>use this</button>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
          {version && version.suggestions.some((s) => !s.decision) && (
            <div className="card">
              <h2>VLM suggestions</h2>
              <ul className="list small">
                {version.suggestions.filter((s) => !s.decision).map((s) => (
                  <li key={s.id}>{s.kind}: <span className="muted">{s.reason}</span>
                    <div className="row">
                      <button className="link" disabled={busy || ops.length > 0} onClick={() => run(() => post<PlanVersion>(`projects/${projectId}/plans/${vid}/suggestions/${s.id}`, { action: "accept" }))}>accept</button>
                      <button className="link" disabled={busy} onClick={() => run(async () => {
                        await post(`projects/${projectId}/plans/${vid}/suggestions/${s.id}`, { action: "reject" });
                        setVersion(await api<PlanVersion>(`projects/${projectId}/plans/${vid}`));
                      })}>reject</button>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {version?.extraction.sources?.map((s) => (
            <div key={s.page} className="card small">
              <strong>{s.source}</strong> {s.page} · {s.level}
              {s.assist && s.assist.triggers.length > 0 && <div className="warn">{s.assist.triggers.length} region(s) flagged{s.assist.source ? `, ${s.assist.accepted} assisted` : " (no VLM serving)"}</div>}
              <details><summary>notes</summary><ul>{s.notes.map((n, k) => <li key={k}>{n}</li>)}</ul></details>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
