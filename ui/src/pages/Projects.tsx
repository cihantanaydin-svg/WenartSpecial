import { useEffect, useState } from "react";
import { api, post, type Project, type User } from "../api";
import { ErrorBox, navigate } from "../App";

export function ProjectsPage({ user }: { user: User }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [err, setErr] = useState<unknown>(null);

  const load = () => api<Project[]>("projects").then(setProjects).catch(setErr);
  useEffect(() => {
    load();
  }, []);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    try {
      const p = await post<Project>("projects", {
        name,
        latitude: lat ? Number(lat) : null,
        longitude: lon ? Number(lon) : null,
      });
      navigate(`projects/${p.id}`);
    } catch (x) {
      setErr(x);
    }
  };

  const canCreate = user.role === "admin" || user.role === "editor";
  return (
    <section>
      <h1>Projects</h1>
      {canCreate && (
        <form className="card row" onSubmit={create}>
          <label>Name<input name="project_name" value={name} onChange={(e) => setName(e.target.value)} required /></label>
          <label>Latitude<input value={lat} onChange={(e) => setLat(e.target.value)} placeholder="41.01" inputMode="decimal" /></label>
          <label>Longitude<input value={lon} onChange={(e) => setLon(e.target.value)} placeholder="28.97" inputMode="decimal" /></label>
          <button type="submit" disabled={!name.trim()}>Create project</button>
        </form>
      )}
      <ErrorBox error={err} />
      <ul className="list">
        {projects.map((p) => (
          <li key={p.id}>
            <a href={`#/projects/${p.id}`}>{p.name}</a>
            <span className="muted"> · {new Date(p.created_at).toLocaleString()}</span>
          </li>
        ))}
        {projects.length === 0 && <li className="muted">No projects yet.</li>}
      </ul>
    </section>
  );
}
