import { useCallback, useEffect, useState } from "react";
import { ApiFailure, api, post, setCsrf, type User } from "./api";
import { ProjectsPage } from "./pages/Projects";
import { ProjectPage } from "./pages/Project";
import { RunPage } from "./pages/Run";
import { PlanEditor } from "./pages/PlanEditor";

// Hash routing (#/projects, #/projects/:id, #/runs/:id) so any proxy prefix works without rewrites.
function useHashRoute(): string[] {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const on = () => setHash(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash.replace(/^#\/?/, "").split("/").filter(Boolean);
}

export function navigate(path: string): void {
  window.location.hash = `#/${path}`;
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  if (error instanceof ApiFailure) {
    return (
      <div className="error" role="alert">
        <strong>{error.error.code}</strong>: {error.error.message}
        {error.error.fix_hint && <div className="hint">→ {error.error.fix_hint}</div>}
      </div>
    );
  }
  return <div className="error" role="alert">{String(error)}</div>;
}

function Login({ onLogin }: { onLogin: (u: User) => void }) {
  const [key, setKey] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    try {
      const res = await post<{ user: User; csrf_token: string }>("auth/session", { api_key: key.trim() });
      setCsrf(res.csrf_token);
      onLogin(res.user);
    } catch (x) {
      setErr(x);
    }
  };
  return (
    <main className="narrow">
      <h1>ArchRender</h1>
      <p className="muted">Paste your API key. The first admin key comes from <code>archrender bootstrap</code>.</p>
      <form onSubmit={submit} className="card">
        <label>
          API key
          <input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="ark_…" autoFocus name="api_key" />
        </label>
        <button type="submit" disabled={!key.trim()}>Log in</button>
      </form>
      <ErrorBox error={err} />
    </main>
  );
}

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);
  const route = useHashRoute();

  const refresh = useCallback(async () => {
    try {
      setUser(await api<User>("auth/me"));
    } catch {
      setUser(null);
    } finally {
      setChecked(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    const onUnauth = () => setUser(null);
    window.addEventListener("archrender:unauthorized", onUnauth);
    return () => window.removeEventListener("archrender:unauthorized", onUnauth);
  }, [refresh]);

  if (!checked) return <main className="narrow muted">Loading…</main>;
  if (!user) return <Login onLogin={setUser} />;

  const logout = async () => {
    await api("auth/session", { method: "DELETE" }).catch(() => undefined);
    setCsrf(null);
    setUser(null);
  };

  let page: React.ReactNode;
  if (route[0] === "projects" && route[1] && route[2] === "plan") page = <PlanEditor projectId={route[1]} />;
  else if (route[0] === "projects" && route[1]) page = <ProjectPage projectId={route[1]} />;
  else if (route[0] === "runs" && route[1]) page = <RunPage runId={route[1]} />;
  else page = <ProjectsPage user={user} />;

  return (
    <>
      <header className="topbar">
        <a href="#/projects" className="brand">ArchRender</a>
        <span className="muted">{user.name} · {user.role}</span>
        <button className="link" onClick={logout}>Log out</button>
      </header>
      <main>{page}</main>
    </>
  );
}
