#!/usr/bin/env bash
# CalligraphicCut installer.
#
#   ./install.sh                 # venv + package + server deps + smoke test
#   ./install.sh --service       # ... and install/start the systemd --user service (port 8097)
#   ./install.sh --dev           # ... and test deps, run the full test suite
#   ./install.sh --system-site   # reuse system site-packages (faster if numpy/numba/gradio already installed)
#   PYTHON=python3.11 ./install.sh
#   CC_PORT=9000 ./install.sh --service
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
VENV="$ROOT/.venv"
PORT="${CC_PORT:-8097}"
SERVICE=0; DEV=0; SYSSITE=0
for a in "$@"; do
  case "$a" in
    --service) SERVICE=1 ;;
    --dev) DEV=1 ;;
    --system-site) SYSSITE=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. python
command -v "$PY" >/dev/null || die "$PY not found"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "Python >= 3.10 required ($("$PY" -V))"
"$PY" -c 'import venv, ensurepip' 2>/dev/null \
  || die "python venv/ensurepip missing (Debian/Ubuntu: sudo apt install python3-venv)"
say "Python: $("$PY" -V) at $(command -v "$PY")"

# 2. venv
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating venv $VENV"
  if [ "$SYSSITE" = 1 ]; then "$PY" -m venv --system-site-packages "$VENV"; else "$PY" -m venv "$VENV"; fi
else
  say "Reusing venv $VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip

# 3. package
EXTRAS="server"; [ "$DEV" = 1 ] && EXTRAS="server,dev"
say "Installing calligraphic-cut[$EXTRAS] (numpy, scipy, numba, pillow, gradio ...)"
"$VENV/bin/pip" install -q -e "$ROOT[$EXTRAS]"

# 4. smoke test (also warms the numba cache)
say "Smoke test"
"$VENV/bin/python" - <<'EOF'
import numpy as np, time
from calligraphic_cut import Params, resize
from calligraphic_cut import seam_carving as sc
yy, xx = np.mgrid[0:60, 0:80] + 0.5
img = np.repeat(np.clip(xx - (2 + 0.6 * yy) + 0.5, 0, 1)[..., None], 3, axis=2)
t = time.time(); out = resize(img, 50, None, Params()); dt = time.time() - t
assert out.shape == (60, 50, 3), out.shape
ref = resize(img, 50, None, Params(backend="numpy"))
assert np.array_equal(out, ref), "numba != numpy reference"
print(f"   ok: 80->50 px, numba == numpy reference ({dt:.1f}s incl. JIT compile)")
EOF
"$VENV/bin/cc-resize" --help >/dev/null && say "cc-resize OK"
"$VENV/bin/python" -c "import gradio" && say "gradio OK"

if [ "$DEV" = 1 ]; then
  say "Running test suite"
  (cd "$ROOT" && "$VENV/bin/python" -m pytest -q -p no:cacheprovider)
fi

# 5. systemd --user service
if [ "$SERVICE" = 1 ]; then
  command -v systemctl >/dev/null || die "systemctl not found; start manually: $VENV/bin/cc-server"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  [ -S "$XDG_RUNTIME_DIR/bus" ] && export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
  UNIT_DIR="$HOME/.config/systemd/user"; mkdir -p "$UNIT_DIR"
  say "Installing systemd user unit calligraphiccut (port $PORT)"
  cat > "$UNIT_DIR/calligraphiccut.service" <<EOF
[Unit]
Description=Calligraphic Cutting local lab (Gradio) on 127.0.0.1:$PORT
After=network.target

[Service]
WorkingDirectory=$ROOT
ExecStart=$VENV/bin/cc-server --host 127.0.0.1 --port $PORT
Environment=OMP_NUM_THREADS=2
Environment=GRADIO_ANALYTICS_ENABLED=False
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now calligraphiccut
  systemctl --user restart calligraphiccut
  say "Waiting for http://127.0.0.1:$PORT ..."
  for _ in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break; sleep 2; done
  curl -s -o /dev/null "http://127.0.0.1:$PORT/" || die "server did not come up: journalctl --user -u calligraphiccut"
  say "Service running: http://127.0.0.1:$PORT"
fi

cat <<EOF

Installed. Next:
  $VENV/bin/cc-server                       # local lab: http://127.0.0.1:$PORT
  $VENV/bin/cc-resize in.png out.png --width 300
  docs/USAGE.md                              # instructions with screenshots
EOF
