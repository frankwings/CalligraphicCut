#!/usr/bin/env python3
"""One-step launcher for the CalligraphicCut lab.

    python launch.py              # first run: sets up .venv + dependencies (a few minutes), then
                                  # starts the lab and opens it in your browser
    python launch.py --port 9000  # another port
    python launch.py --no-browser

Works on Linux, macOS and Windows with Python >= 3.10. Uses only the standard library;
everything else is installed into ./.venv next to this file. Stop with Ctrl+C (or close the window).
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
STAMP = VENV / ".cc_installed"


def log(msg):
    print(f"[CalligraphicCut] {msg}", flush=True)


def ensure_env():
    if sys.version_info < (3, 10):
        sys.exit(f"Python >= 3.10 is required (this is {sys.version.split()[0]}).")
    if not PY.exists():
        log(f"Creating virtual environment in {VENV} ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    stale = not STAMP.exists() or (ROOT / "pyproject.toml").stat().st_mtime > STAMP.stat().st_mtime
    if stale:
        log("Installing dependencies (first run only; numpy, scipy, numba, pillow, gradio) ...")
        subprocess.check_call([str(PY), "-m", "pip", "install", "-q", "--upgrade", "pip"])
        subprocess.check_call([str(PY), "-m", "pip", "install", "-q", "-e", f"{ROOT}[server]"])
        STAMP.write_text(time.ctime())


def free_port(start):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    sys.exit("No free port found.")


def wait_up(url, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit("The server stopped during start-up; see the messages above.")
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    sys.exit("The server did not start in time.")


def _stop(signum, frame):
    raise KeyboardInterrupt


def main():
    for sig in ("SIGTERM", "SIGHUP"):           # kill / terminal window closed -> stop the server too
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _stop)
    ap = argparse.ArgumentParser(description="Start the CalligraphicCut lab")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    ensure_env()
    port = free_port(a.port)
    url = f"http://127.0.0.1:{port}/"
    log("Starting the lab (the first start compiles the kernels, about 20 s) ...")
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED="False")
    proc = subprocess.Popen([str(PY), "-m", "calligraphic_cut.server", "--port", str(port)], cwd=ROOT, env=env)
    try:
        wait_up(url, proc)
        log(f"Ready: {url}   (press Ctrl+C or close this window to stop)")
        if not a.no_browser:
            webbrowser.open(url)
        proc.wait()
    except KeyboardInterrupt:
        log("Stopping ...")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
