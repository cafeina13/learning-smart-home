"""Run the Kaggle notebook's steps with phone notifications through ntfy.

    from kaggle_steps import notify, run_step
    run_step("REFIT habit networks", "python -u scripts/eval_habit_networks.py", "logs/refit.log")

The ntfy address (e.g. https://ntfy.sh/<your-topic>) comes from the Kaggle
secret NTFY_URL (Add-ons > Secrets: the easy place to change it); if that is
missing, empty, or Kaggle does not answer within 10 s, from the environment
variable NTFY_URL (the notebook's optional first line). It is never written in
the repository, since anyone who knows a topic can read it. The time limit is
there because one committed run sat with no output at all, likely in the secret
lookup. Without an address everything still runs and nothing is sent. Needs
Internet on.

run_step streams the command's output to the notebook and to a log file, sends
the latest progress line every PROGRESS_EVERY seconds, and on exit a "done"
(with the summary lines) or an "error" (high priority, with the last lines).
"""

import os
import subprocess
import threading
import time
import urllib.request
from collections import deque

PROGRESS_EVERY = 30 * 60


def _secret(timeout: float = 10.0) -> str | None:
    """The Kaggle secret NTFY_URL, or None if it is missing, empty, or Kaggle does not answer in time."""
    found = []

    def lookup():
        try:
            from kaggle_secrets import UserSecretsClient
            found.append(UserSecretsClient().get_secret("NTFY_URL"))
        except Exception:
            pass

    worker = threading.Thread(target=lookup, daemon=True)  # a hung lookup must not hold up the run
    worker.start()
    worker.join(timeout)
    value = (found[0] or "").strip() if found else ""
    return value or None


def _address() -> str | None:
    return _secret() or os.environ.get("NTFY_URL") or None


ADDRESS = _address()


def notify(title: str, message: str, priority: str = "default", tags: str = "") -> None:
    """Send one notification; never fails the run if ntfy cannot be reached."""
    print(f"[ntfy] {title}: {message[:200]}", flush=True)
    if not ADDRESS:
        return
    request = urllib.request.Request(ADDRESS, data=message[-3500:].encode("utf-8"), method="POST",
                                     headers={"Title": title, "Priority": priority, "Tags": tags})
    try:
        urllib.request.urlopen(request, timeout=15).read()
    except Exception as e:  # noqa: BLE001 - a lost notification must not stop the training
        print(f"[ntfy] not sent: {e}", flush=True)


def run_step(name: str, command: str, log_path: str, env: dict | None = None, summary_lines: int = 12) -> bool:
    """Run one shell command, streaming and logging its output; notify progress and the outcome."""
    start = last_sent = time.time()
    tail = deque(maxlen=summary_lines)
    last_progress = ""
    with open(log_path, "w", encoding="utf-8") as log, subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, **(env or {})}) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            tail.append(line.rstrip())
            if line.startswith(("task ", "home ")):
                last_progress = line.strip()
            if time.time() - last_sent >= PROGRESS_EVERY and last_progress:
                notify(f"{name}: running {(time.time() - start) / 60:.0f} min", last_progress, "low", "hourglass")
                last_sent = time.time()
    minutes = (time.time() - start) / 60
    if proc.returncode == 0:
        notify(f"{name}: done in {minutes:.0f} min", "\n".join(tail), "default", "white_check_mark")
        return True
    notify(f"{name}: ERROR after {minutes:.0f} min (exit {proc.returncode})", "\n".join(tail), "high", "x")
    return False
