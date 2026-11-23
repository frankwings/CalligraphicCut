"""Capture UI screenshots of the running cc-server for docs/USAGE.md (needs playwright + chromium)."""
import os
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8097/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "docs", "img")
P = os.path.join(ROOT, "tests", "data", "paper")


def visible_file_input(page):
    for loc in page.locator("input[type=file]").all():
        if loc.evaluate("e => e.closest('[role=tabpanel]') === null || e.closest('[role=tabpanel]').offsetParent !== null"):
            return loc
    raise RuntimeError("no visible file input")


def visible_run(page):
    return [b for b in page.get_by_role("button", name="Run", exact=True).all() if b.is_visible()][0]


with sync_playwright() as pw:
    br = pw.chromium.launch()
    page = br.new_page(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
    page.goto(URL, wait_until="networkidle")
    page.screenshot(path=f"{IMG}/01_home.png")

    # Tab 1: teaser 50 %
    visible_file_input(page).set_input_files(f"{P}/teaser_orig.jpg")
    page.wait_for_timeout(1500)
    visible_run(page).click()
    page.wait_for_selector("text=time", timeout=120000)
    page.wait_for_timeout(2500)
    page.screenshot(path=f"{IMG}/02_resize_teaser.png", full_page=True)

    # Tab 2: progressive on face
    page.get_by_role("tab", name="Progressive").click()
    page.wait_for_timeout(1000)
    visible_file_input(page).set_input_files(f"{P}/face_orig.png")
    page.wait_for_timeout(1500)
    visible_run(page).click()
    page.wait_for_selector("text=Red = next cut", timeout=120000)
    page.wait_for_timeout(2500)
    page.screenshot(path=f"{IMG}/03_progressive_face.png", full_page=True)
    br.close()
print("ok")
