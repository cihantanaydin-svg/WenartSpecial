#!/usr/bin/env python3
"""End-to-end smoke test against a running ArchRender (pod or local container).

    python scripts/smoke_test.py --url https://<POD>-8000.proxy.runpod.net --api-key ark_…
    python scripts/smoke_test.py --url http://127.0.0.1:8000 --bootstrap-token <admin token>

Creates a project, uploads a synthetic floor plan (DXF, fixed seed) plus a small raster sheet,
waits for page analysis (S1: OCR + classifier), runs a section with gate policy 'never' (S2 extracts
the plan), waits, downloads the bundle, and prints the pages, the plan version, stage timings,
render device, VRAM peaks (when recorded) and QA metrics. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archrender.client.client import ArchRenderClient, ArchRenderClientError  # noqa: E402


def _plan_dxf() -> bytes:
    """A synthetic apartment floor plan (walls, doors, windows, room names, dimensions; cm)."""
    import numpy as np

    from archrender.synth.dxf import plan_dxf
    from archrender.synth.plan import random_spec

    data, _ = plan_dxf(random_spec(np.random.default_rng(3), variant="manhattan"))
    return data


def _sheet_png() -> bytes:
    """A small scanned-looking sheet with Turkish text, so S1 runs OCR inside the image."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1400, 900), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=44)
    d.rectangle((80, 80, 1320, 820), outline="black", width=6)
    d.text((140, 140), "ZEMİN KAT PLANI", fill="black", font=font)
    d.text((140, 220), "Ölçek 1/50", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG", dpi=(150, 150))
    return buf.getvalue()


def _wait_understanding(c: ArchRenderClient, project_id: str, timeout: float) -> dict | None:
    """The latest S1 job once it ends (intake queues it before it completes), or None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = c.understanding_status(project_id)
        if job and job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(2)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--bootstrap-token")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args(argv)
    t0 = time.time()
    try:
        key = args.api_key
        if not key:
            if not args.bootstrap_token:
                print("need --api-key or --bootstrap-token", file=sys.stderr)
                return 2
            key = ArchRenderClient(args.url).bootstrap(args.bootstrap_token, "smoke-admin")[
                "api_key"
            ]
            print(f"bootstrapped admin key: {key}")
        c = ArchRenderClient(args.url, key, timeout=60)
        project = c.create_project(f"smoke {time.strftime('%Y-%m-%d %H:%M')}", 41.0082, 28.9784)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "Kat Planı.dxf"
            f.write_bytes(_plan_dxf())
            sheet = Path(tmp) / "Tarama.png"
            sheet.write_bytes(_sheet_png())
            for path in (f, sheet):
                _, intake = c.upload(project["id"], path)
                job = c.wait(intake, stop_at_gate=False)
                if job["status"] != "succeeded":
                    print(f"FAIL intake {path.name}: {job.get('error')}", file=sys.stderr)
                    return 1
        s1 = _wait_understanding(c, project["id"], 900)
        if s1 is None or s1["status"] != "succeeded":
            print(f"FAIL page analysis (S1): {s1 and s1.get('error')}", file=sys.stderr)
            return 1
        pages = c.pages(project["id"])
        print("\n== pages (S1)")
        for pg in pages:
            print(
                f"  {pg['filename']:<16} {pg['label']:<14} {pg['confidence']:.2f}"
                f"{' review' if pg['needs_review'] else ''}"
            )
        if len(pages) != 2 or not all(pg["analysed"] for pg in pages):
            print(f"FAIL: expected 2 analysed pages, got {pages}", file=sys.stderr)
            return 1
        run = c.start_run(
            project["id"],
            views=args.views,
            width=args.width,
            height=args.height,
            samples=args.samples,
            gate_policy="never",
        )
        job = c.wait(
            run["job_id"],
            lambda e: print(f"  {e.get('event')}: {json.dumps(e.get('data'))[:160]}"),
            stop_at_gate=False,
        )
        if job["status"] != "succeeded":
            print(f"FAIL run: {job.get('error')}", file=sys.stderr)
            return 1
        r = c.get_run(run["run_id"])
        result = r["result"]
        plan = c.plan_version(project["id"], result["plan_version"])
        print("\n== plan (S2)")
        print(
            f"  {plan['id']} v{plan['number']} {plan['status']}: {len(plan['plan']['walls'])} walls,"
            f" {len(plan['plan']['openings'])} openings, {len(plan['plan']['rooms'])} rooms"
            f" from {plan['plan']['source']}; {plan['blocking']} blocking issues"
        )
        if plan["plan"]["source"] != "dxf" or len(plan["plan"]["rooms"]) < 2:
            print("FAIL: the plan was not extracted from the uploaded DXF", file=sys.stderr)
            return 1
        dest = Path(tempfile.gettempdir()) / f"smoke_{run['run_id']}.zip"
        c.download_bundle(r["bundle"]["id"], dest)
        with zipfile.ZipFile(dest) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("manifests/run_manifest.json"))
        missing = {"model/scene.glb", "qa/qa_report.html", "renders/view_1.png"} - names
        if missing:
            print(f"FAIL bundle missing {missing}", file=sys.stderr)
            return 1
        print("\n== stage timings (s)")
        for t in result["timings"]:
            print(f"  {t['stage']:<18} {'cached' if t['cached'] else f'{t["seconds"]:.2f}'}")
        print("\n== render environment")
        print(
            f"  blender={manifest['environment'].get('blender')} device={manifest['environment'].get('render_device')}"
        )
        peaks = [t for t in manifest["timings"] if t.get("vram_peak_mb")]
        print(f"  VRAM peaks: {peaks or 'not recorded (no GPU model stages in this build)'}")
        print("\n== QA (delivered images)")
        for v in result["views"]:
            print(
                f"  {v['view_id']}: {v['status']}{' [mock estimators]' if v['uses_mocks'] else ''}"
            )
            for chk in v["checks"]:
                print(
                    f"    {chk['name']:<22} {'pass' if chk['passed'] else 'FAIL'} value={chk['value']} Δ={chk['delta']}"
                )
        print(
            f"\nSMOKE TEST PASSED in {time.time() - t0:.1f}s · bundle {dest} ({dest.stat().st_size} bytes)"
        )
        return 0
    except ArchRenderClientError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
