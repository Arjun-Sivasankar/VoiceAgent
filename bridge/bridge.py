"""Ultron local system bridge.

Loopback-only HTTP service that performs allow-listed desktop actions for the
Ultron HUD (open apps/urls, media keys, volume, lock, screenshot, system info).

Security model:
- Binds 127.0.0.1 only.
- CORS is granted only to the HUD origins, and every action requires the
  custom header X-Ultron: 1 (a custom header forces a preflight, which other
  origins fail), so drive-by localhost CSRF from a random website is blocked.
- Actions are an explicit allowlist; there is no arbitrary command execution.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 7870
ALLOWED_ORIGINS = {"http://127.0.0.1:7860", "http://localhost:7860"}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_ROOT = os.path.join(REPO_ROOT, ".runtime")
SHOT_DIR = os.path.join(RUNTIME_ROOT, "screenshots")
LOG_DIR = os.path.join(RUNTIME_ROOT, "logs")

APP_MAP = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "files": "explorer.exe",
    "settings": "ms-settings:",
    "task manager": "taskmgr.exe",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "powershell": "powershell.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "paint": "mspaint.exe",
    "snipping tool": "ms-screenclip:",
    "spotify": "spotify:",
    "vscode": "code",
    "vs code": "code",
    "steam": "steam:",
    "browser": "https://www.google.com",
    "edge": "msedge",
    "chrome": "chrome",
}

VK = {
    "mute": 0xAD,
    "volume_down": 0xAE,
    "volume_up": 0xAF,
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
    "play_pause": 0xB3,
}

KEYEVENTF_KEYUP = 0x0002


def log(message: str) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(LOG_DIR, "bridge.log"), "a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
    except Exception:
        pass


def press_key(name: str) -> None:
    code = VK.get(name)
    if code is None:
        raise ValueError(f"unknown media key: {name}")
    user32 = ctypes.windll.user32
    for _ in range(3):
        user32.keybd_event(code, 0, 0, 0)
        user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.06)


def action_open_app(name: str) -> str:
    key = str(name or "").strip().lower()
    target = APP_MAP.get(key)
    if not target:
        raise ValueError(f"app not in allowlist: {name}")
    os.startfile(target)  # type: ignore[attr-defined]
    return f"opened {key}"


def action_open_url(url: str) -> str:
    value = str(url or "").strip()
    if not re.match(r"^https?://[^\s]+$", value, re.IGNORECASE):
        raise ValueError("only http/https urls are allowed")
    os.startfile(value)  # type: ignore[attr-defined]
    return f"opened {value}"


def action_media(command: str) -> str:
    press_key(str(command or "").strip().lower())
    return f"media {command}"


def action_lock() -> str:
    ctypes.windll.user32.LockWorkStation()
    return "workstation locked"


def action_screenshot() -> str:
    os.makedirs(SHOT_DIR, exist_ok=True)
    path = os.path.join(SHOT_DIR, "shot_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
        "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen;"
        "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height;"
        "$g=[System.Drawing.Graphics]::FromImage($bmp);"
        "$g.CopyFromScreen($b.Left,$b.Top,0,0,$bmp.Size);"
        f"$bmp.Save('{path}',[System.Drawing.Imaging.ImageFormat]::Png);"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return path


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def action_system_info() -> str:
    parts = [f"cpu_cores={os.cpu_count()}"]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        parts.append(f"ram_used={round((stat.ullTotalPhys - stat.ullAvailPhys) / 1e9, 1)}GB")
        parts.append(f"ram_total={round(stat.ullTotalPhys / 1e9, 1)}GB")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if gpu.returncode == 0 and gpu.stdout.strip():
            parts.append("gpu=" + gpu.stdout.strip().splitlines()[0])
    except Exception:
        pass
    try:
        drive, _ = os.path.splitdrive(os.path.abspath(REPO_ROOT))
        usage = shutil.disk_usage((drive + os.sep) if drive else os.path.abspath(os.sep))
        parts.append(f"d_free={round(usage.free / 1e9, 1)}GB")
    except Exception:
        pass
    return "; ".join(parts)


ACTIONS = {
    "open_app": lambda args: action_open_app(args.get("name", "")),
    "open_url": lambda args: action_open_url(args.get("url", "")),
    "media": lambda args: action_media(args.get("command", "")),
    "lock": lambda args: action_lock(),
    "screenshot": lambda args: action_screenshot(),
    "system_info": lambda args: action_system_info(),
}


class Handler(BaseHTTPRequestHandler):
    server_version = "UltronBridge/1.0"

    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Ultron")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/health":
            self._json(200, {"ok": True, "service": "ultron-bridge"})
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/action":
            self._json(404, {"ok": False, "error": "not found"})
            return
        if self.headers.get("X-Ultron") != "1":
            self._json(403, {"ok": False, "error": "missing X-Ultron header"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._json(400, {"ok": False, "error": f"bad json: {exc}"})
            return
        name = str(payload.get("action", "")).strip()
        args = payload.get("args") or {}
        handler = ACTIONS.get(name)
        if handler is None:
            self._json(400, {"ok": False, "error": f"unknown action: {name}"})
            return
        try:
            result = handler(args if isinstance(args, dict) else {})
            log(f"{name} {args} -> {result}")
            self._json(200, {"ok": True, "result": result})
        except Exception as exc:
            log(f"{name} {args} -> ERROR {exc}")
            self._json(200, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:  # silence stderr
        return


def main() -> int:
    log("bridge starting")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Ultron bridge listening on http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log("bridge stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
