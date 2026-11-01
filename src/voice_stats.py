import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

_LOCK = threading.Lock()
_START = time.time()
_COUNTS = Counter()
_LATENCIES = []
_LAST_TEXT = ""
_LAST_COMMAND = ""
_LAST_ERROR = ""
_PATH = Path(__file__).resolve().parent.parent / "data" / "voice_stats.json"


def _snapshot():
    values = list(_LATENCIES)
    return {
        "uptime_seconds": round(time.time() - _START, 1),
        "recognized_utterances": _COUNTS["recognized"],
        "commands_executed": _COUNTS["commands"],
        "commands_ignored": _COUNTS["ignored"],
        "dictation_characters": _COUNTS["dictation_chars"],
        "errors": _COUNTS["errors"],
        "average_latency_ms": round(sum(values) / len(values), 1) if values else 0,
        "last_latency_ms": round(values[-1], 1) if values else 0,
        "last_text": _LAST_TEXT,
        "last_command": _LAST_COMMAND,
        "last_error": _LAST_ERROR,
    }


def _save():
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            data = _snapshot()
        temp = _PATH.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(temp), str(_PATH))
    except Exception:
        pass


def record_recognition(text, latency_ms):
    global _LAST_TEXT
    with _LOCK:
        _COUNTS["recognized"] += 1
        _LATENCIES.append(float(latency_ms))
        del _LATENCIES[:-500]
        _LAST_TEXT = str(text)[:300]
    _save()


def record_command(command):
    global _LAST_COMMAND
    with _LOCK:
        _COUNTS["commands"] += 1
        _LAST_COMMAND = str(command)[:200]
    _save()


def record_ignored():
    with _LOCK:
        _COUNTS["ignored"] += 1
    _save()


def record_dictation(text):
    with _LOCK:
        _COUNTS["dictation_chars"] += len(text)
    _save()


def record_error(message):
    global _LAST_ERROR
    with _LOCK:
        _COUNTS["errors"] += 1
        _LAST_ERROR = str(message)[:300]
    _save()


def get_voice_stats():
    with _LOCK:
        return _snapshot()
