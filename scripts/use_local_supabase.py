"""Writes .env.test with the connection details of the local Supabase stack.

Run after `npx supabase start`. The values are the local stack's development
defaults, read from `supabase status`; they are written to the file and not
printed. Tests load .env.test before .env, so with this file present they run
against the local stack and leave the hosted project alone.

    python scripts/use_local_supabase.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WANTED = {"DB_URL": "DATABASE_URL", "API_URL": "SUPABASE_URL"}
KEY_NAMES = ("PUBLISHABLE_KEY", "ANON_KEY")  # newer and older CLI versions


def main() -> int:
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx is None:
        print("npx was not found; install Node.js.")
        return 1
    done = subprocess.run([npx, "--yes", "supabase", "status", "-o", "env"], cwd=ROOT, capture_output=True, text=True)
    values = {}
    for line in done.stdout.splitlines():
        name, sep, value = line.partition("=")
        if sep:
            values[name.strip()] = value.strip().strip('"')
    lines = [f"{target}={values[source]}" for source, target in WANTED.items() if source in values]
    key = next((values[name] for name in KEY_NAMES if values.get(name)), None)
    if key:
        lines.append(f"SUPABASE_PUBLISHABLE_KEY={key}")
    if len(lines) < 3:
        print("The local Supabase stack does not seem to be running (supabase status gave no connection details).")
        return 1
    (ROOT / ".env.test").write_text("# Local Supabase stack (generated; git-ignored)\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote .env.test with DATABASE_URL, SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY for the local stack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
