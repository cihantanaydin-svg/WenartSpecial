"""UI end-to-end with Playwright (pre-installed Chromium) against a live server + worker."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from archrender.core.config import REPO_ROOT
from archrender.synth.sheets import floor_plan_page
from tests.conftest import STRICT
from tests.e2e.conftest import Live
from tests.helpers import MINIMAL_DXF

pytestmark = [pytest.mark.e2e, pytest.mark.blender]

CHROMIUM = os.environ.get("ARCHRENDER_CHROMIUM", "/opt/pw-browsers/chromium")


@pytest.fixture
def page():  # type: ignore[no-untyped-def]
    if not (REPO_ROOT / "ui" / "dist" / "index.html").exists():
        if STRICT:
            pytest.fail("ARCHRENDER_TEST_STRICT=1 but the UI is not built (run `make ui`)")
        pytest.skip("UI not built (run `make ui`)")
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        kwargs = {"executable_path": CHROMIUM} if Path(CHROMIUM).exists() else {}
        browser = p.chromium.launch(**kwargs)
        pg = browser.new_page(viewport={"width": 1280, "height": 900})
        yield pg
        browser.close()


def test_ui_full_flow(live: Live, page, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from playwright.sync_api import expect

    page.goto(live.url + "/")
    page.fill("input[name=api_key]", live.admin_key)
    page.click("text=Log in")
    expect(page.locator("h1", has_text="Projects")).to_be_visible()

    page.fill("input[name=project_name]", "Daire 3+1 Kadıköy")
    page.click("text=Create project")
    expect(page.locator("h1", has_text="Daire 3+1 Kadıköy")).to_be_visible()

    plan = tmp_path / "Kat Planı.dxf"
    plan.write_bytes(MINIMAL_DXF)
    page.set_input_files("[data-testid=file-input]", str(plan))
    expect(page.locator(".progress-row", has_text="done")).to_be_visible(timeout=30_000)
    expect(page.locator("li", has_text="Kat Planı.dxf")).to_be_visible(timeout=10_000)

    # S1 in the UI: the panel follows the analysis job, shows the class, and takes a correction
    sheet = tmp_path / "Zemin Kat Planı.pdf"
    sheet.write_bytes(floor_plan_page(np.random.default_rng(8)).pdf)
    page.set_input_files("[data-testid=file-input]", str(sheet))
    card = page.locator("[data-testid=pages] figure", has_text="Zemin Kat Planı.pdf")
    select = card.locator("select[aria-label='page class']")
    expect(select).to_have_value("floor_plan", timeout=120_000)
    expect(card).to_contain_text("scale 1:")
    select.select_option("ceiling_plan")
    expect(card).to_contain_text("you set this (model: floor_plan)")
    if os.environ.get("ARCHRENDER_E2E_SCREENSHOTS"):
        page.locator("[data-testid=pages]").screenshot(
            path=str(Path(os.environ["ARCHRENDER_E2E_SCREENSHOTS"]) / "pages_panel.png")
        )

    page.fill("input[name=views]", "1")
    page.fill("input[name=width]", "96")
    page.fill("input[name=height]", "54")
    page.fill("input[name=samples]", "8")
    page.click("text=Start run")
    expect(page.locator("[data-testid=gate-D_final]")).to_be_visible(timeout=120_000)
    page.fill("[data-testid=gate-D_final] input[name=notes]", "UI e2e approve")
    page.click("[data-testid=gate-D_final] >> text=Approve")
    expect(page.locator("[data-testid=run-status]")).to_have_text("succeeded", timeout=120_000)
    expect(page.locator("[data-testid=view-view_1]")).to_be_visible()
    expect(page.locator(".banner", has_text="Mock models were used")).to_be_visible()
    expect(page.locator("[data-testid=download]")).to_be_visible()
    page.click("text=Why / QA details")
    expect(page.locator("table.checks td", has_text="structural_edge_f")).to_be_visible()
    shots = os.environ.get("ARCHRENDER_E2E_SCREENSHOTS")
    if shots:
        page.screenshot(path=str(Path(shots) / "run_page.png"), full_page=True)
