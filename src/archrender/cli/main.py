"""``archrender`` command-line interface (same API as the UI)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from archrender.client.client import TERMINAL, ArchRenderClient, ArchRenderClientError


def _client(args: argparse.Namespace) -> ArchRenderClient:
    if not args.url:
        raise SystemExit(
            "error: --url or ARCHRENDER_URL is required (e.g. https://<pod>-8000.proxy.runpod.net)"
        )
    return ArchRenderClient(args.url, args.api_key)


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print(
                f"{k}: {v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)}"
            )
    else:
        print(obj)


def _event_line(ev: dict[str, Any]) -> None:
    d = ev.get("data", {})
    kind = ev.get("event")
    if kind == "progress":
        print(
            f"  [{d.get('progress', 0) * 100:5.1f}%] {d.get('message') or d.get('stage')}",
            flush=True,
        )
    elif kind == "status":
        print(
            f"  status → {d.get('status')}" + (f" ({d['gate']})" if d.get("gate") else ""),
            flush=True,
        )
    elif kind == "gate":
        print(f"  gate {d.get('gate')} waiting for review", flush=True)
    elif kind == "error":
        print(f"  error {d.get('code')}: {d.get('message')}\n    → {d.get('fix_hint')}", flush=True)


def cmd_bootstrap(args: argparse.Namespace) -> int:
    res = _client(args).bootstrap(args.token, args.name)
    if args.json:
        _print(res, True)
    else:
        print(f"admin api key (shown once): {res['api_key']}")
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    c = _client(args)
    if args.action == "create":
        _print(c.create_project(args.name, args.lat, args.lon), args.json)
    else:
        for p in c.list_projects():
            print(f"{p['id']}  {p['name']}")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    c = _client(args)
    rc = 0
    for f in args.files:
        path = Path(f)
        print(f"uploading {path.name} ({path.stat().st_size} bytes)")
        _upload_id, job_id = c.upload(
            args.project, path, on_chunk=lambda i, n: print(f"  chunk {i}/{n}", flush=True)
        )
        job = c.wait(job_id, stop_at_gate=False)
        if job["status"] == "succeeded":
            print(f"  → document {job['result']['document_id']} ({job['result'].get('kind')})")
        else:
            err = job.get("error") or {}
            print(
                f"  → failed {err.get('code')}: {err.get('message')}\n    → {err.get('fix_hint')}"
            )
            rc = 1
    return rc


def cmd_run(args: argparse.Namespace) -> int:
    c = _client(args)
    config: dict[str, Any] = {
        "views": args.views,
        "width": args.width,
        "height": args.height,
        "gate_policy": args.gate_policy,
        "seed": args.seed,
        "blend_file": args.blend,
    }
    if args.samples:
        config["samples"] = args.samples
    if args.rooms:
        config["room_ids"] = args.rooms.split(",")
    if args.material:
        pairs = [m.partition("=") for m in args.material]
        bad = [m for m, (_, sep, v) in zip(args.material, pairs, strict=True) if not sep or not v]
        if bad:
            print(f"--material expects SURFACE=MATERIAL_ID, got {bad}", file=sys.stderr)
            return 2
        config["materials"] = {k: v for k, _, v in pairs}
    res = c.start_run(args.project, **config)
    print(f"run {res['run_id']} (job {res['job_id']})")
    if args.wait:
        job = c.wait(res["job_id"], _event_line)
        run = c.get_run(res["run_id"])
        return _report_run(run, job)
    return 0


def _report_run(run: dict[str, Any], job: dict[str, Any]) -> int:
    print(f"run status: {run['status']}")
    for g in run["gates"]:
        if g["status"] == "pending":
            print(f"  gate {g['gate']} pending → archrender gate approve {run['id']} {g['gate']}")
    if run.get("result"):
        for v in run["result"]["views"]:
            print(
                f"  {v['view_id']}: {v['status']}{' (mock estimators)' if v['uses_mocks'] else ''}"
            )
        if run["result"].get("uses_mocks"):
            print(
                "  NOTE: mock models were used; this is a pipeline test, not a client deliverable."
            )
    if job["status"] == "failed":
        err = job.get("error") or {}
        print(f"  failed {err.get('code')}: {err.get('message')}\n    → {err.get('fix_hint')}")
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    c = _client(args)
    if args.id.startswith("run_"):
        run = c.get_run(args.id)
        if args.follow and run["status"] not in TERMINAL:
            c.wait(run["job_id"], _event_line)
            run = c.get_run(args.id)
        if args.json:
            _print(run, True)
            return 0
        return _report_run(run, c.get_job(run["job_id"]))
    job = c.get_job(args.id)
    if args.follow and job["status"] not in TERMINAL:
        job = c.wait(args.id, _event_line)
    _print(job, args.json)
    return 0 if job["status"] != "failed" else 1


def cmd_gate(args: argparse.Namespace) -> int:
    run = _client(args).decide_gate(args.run, args.gate, args.action == "approve", args.notes)
    print(f"gate {args.gate}: {args.action}d; run status {run['status']}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    c = _client(args)
    run = c.get_run(args.run)
    if not run.get("bundle"):
        print(f"run {args.run} has no bundle yet (status {run['status']})")
        return 1
    dest = Path(args.output or f"archrender_{args.run}.zip")
    c.download_bundle(run["bundle"]["id"], dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="archrender", description="ArchRender CLI")
    p.add_argument("--url", default=os.environ.get("ARCHRENDER_URL"))
    p.add_argument("--api-key", default=os.environ.get("ARCHRENDER_API_KEY"))
    p.add_argument("--json", action="store_true", help="machine-readable output")
    # The same options are also accepted after the subcommand (`archrender run --url …`).
    # SUPPRESS keeps an absent option from overwriting the value parsed before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=argparse.SUPPRESS, help="server URL (env ARCHRENDER_URL)")
    common.add_argument(
        "--api-key", default=argparse.SUPPRESS, help="API key (env ARCHRENDER_API_KEY)"
    )
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser(
        "bootstrap", parents=[common], help="exchange the bootstrap token for the first admin key"
    )
    b.add_argument("--token", required=True)
    b.add_argument("--name", default="admin")
    b.set_defaults(fn=cmd_bootstrap)

    pr = sub.add_parser("project", parents=[common], help="create or list projects")
    pr.add_argument("action", choices=["create", "list"])
    pr.add_argument("name", nargs="?")
    pr.add_argument("--lat", type=float)
    pr.add_argument("--lon", type=float)
    pr.set_defaults(fn=cmd_project)

    up = sub.add_parser("upload", parents=[common], help="upload documents (resumable, chunked)")
    up.add_argument("project")
    up.add_argument("files", nargs="+")
    up.set_defaults(fn=cmd_upload)

    r = sub.add_parser("run", parents=[common], help="start a section run")
    r.add_argument("project")
    r.add_argument("--rooms", help="comma-separated room ids (default: all rooms)")
    r.add_argument("--views", type=int, default=3)
    r.add_argument("--width", type=int, default=3840)
    r.add_argument("--height", type=int, default=2160)
    r.add_argument("--samples", type=int)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument(
        "--gate-policy",
        choices=["always", "on_low_confidence", "never"],
        default="on_low_confidence",
    )
    r.add_argument("--blend", action="store_true", help="include the .blend file in the bundle")
    r.add_argument(
        "--material",
        action="append",
        default=[],
        metavar="SURFACE=MATERIAL_ID",
        help="material choice, e.g. floor=stone_porcelain_grey or wall:W1=paint_warm_white (repeatable)",
    )
    r.add_argument(
        "--wait", action="store_true", help="follow progress until done or a gate needs review"
    )
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", parents=[common], help="show a run (run_…) or job (job_…)")
    s.add_argument("id")
    s.add_argument("--follow", action="store_true")
    s.set_defaults(fn=cmd_status)

    g = sub.add_parser("gate", parents=[common], help="approve or reject a pending gate")
    g.add_argument("action", choices=["approve", "reject"])
    g.add_argument("run")
    g.add_argument("gate", choices=["A_plan", "B_brief", "C_cameras", "D_final"])
    g.add_argument("--notes")
    g.set_defaults(fn=cmd_gate)

    d = sub.add_parser("download", parents=[common], help="download a run's bundle (resumable)")
    d.add_argument("run")
    d.add_argument("-o", "--output")
    d.set_defaults(fn=cmd_download)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "project" and args.action == "create" and not args.name:
        print("error: project create needs a NAME", file=sys.stderr)
        return 2
    try:
        return int(args.fn(args))
    except ArchRenderClientError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
