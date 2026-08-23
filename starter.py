"""One-click launcher: syncs the uv environment and starts FastAPI."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INSTALL_HINT = """\
uv was not found on PATH.

Install it, then re-run this script:

  Windows (PowerShell):
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

  macOS / Linux:
    curl -LsSf https://astral.sh/uv/install.sh | sh

See https://docs.astral.sh/uv/getting-started/installation/ for other options.
"""


def run(cmd: list[str], **kwargs) -> None:
    print(f">  {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def ensure_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        print(INSTALL_HINT, file=sys.stderr)
        raise SystemExit(1)
    print(f"OK  uv found at {uv}")
    return uv


def sync_environment(uv: str) -> None:
    """Create .venv (with the pinned Python) and install locked dependencies."""
    print("..  Syncing environment ...")
    run([uv, "sync"], cwd=str(ROOT))
    print("OK  Environment ready")


def start_server(uv: str) -> None:
    print("\n>>  Starting FastAPI server ...")
    print("    Open http://localhost:8000 in your browser\n")
    # Invoke uvicorn via `python -m` rather than the generated uvicorn.exe shim:
    # endpoint-security policies on Windows commonly block those shims with
    # "Access is denied (os error 5)", while the interpreter itself runs fine.
    subprocess.call(
        [uv, "run", "python", "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    uv_path = ensure_uv()
    sync_environment(uv_path)
    start_server(uv_path)
