"""
agentkit.memory
===============

Two different things are both called "memory". Keeping them separate is one of
the more useful distinctions in agent design.

SHORT-TERM (checkpointers)
    The state of one conversation thread. Written after every superstep so a
    run can be paused, inspected, edited and resumed — possibly in a different
    process, days later. Keyed by `thread_id`.

LONG-TERM (store)
    Facts that outlive any single thread: user preferences, learned
    procedures, summaries of past sessions. Keyed by namespace + key, and
    explicitly loaded into a prompt by a node that decides it is relevant.

Conflating them produces agents that either forget everything between turns or
drag an ever-growing transcript into every request until it overflows.

The implementations here are in-memory and file-backed — enough to develop and
test against. For production, implement the same two tiny interfaces over
Postgres, Redis, or whatever you already run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .types import Message


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _encode(obj: Any) -> Any:
    """Make graph state JSON-safe. Messages get a type tag so we can rebuild
    real objects on the way back rather than handing nodes bare dicts."""
    if isinstance(obj, Message):
        return {"__type__": "Message", **obj.to_dict()}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__type__") == "Message":
            return Message.from_dict(obj)
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Checkpointers (short-term)
# ---------------------------------------------------------------------------
class BaseCheckpointer(ABC):
    """Interface the graph engine calls. Three methods, deliberately.

    A checkpoint payload is `{"state": <graph state>, "next": [node names]}`.
    Storing the pending frontier alongside the state is what lets a resumed
    run continue mid-graph instead of restarting from the entry point.
    """

    @abstractmethod
    def get(self, thread_id: str) -> dict[str, Any] | None:
        """Latest checkpoint for a thread, or None."""

    @abstractmethod
    def put(self, thread_id: str, checkpoint: dict[str, Any]) -> None:
        """Persist a checkpoint. Called after every superstep, so keep it cheap."""

    @abstractmethod
    def delete(self, thread_id: str) -> None:
        """Forget a thread entirely (end of session, GDPR erasure, tests)."""

    # Optional: override to support time-travel debugging.
    def history(self, thread_id: str) -> list[dict[str, Any]]:
        cp = self.get(thread_id)
        return [cp] if cp else []


class InMemoryCheckpointer(BaseCheckpointer):
    """Dict-backed, thread-safe, keeps a bounded history per thread.

    Good for tests, notebooks and single-process apps. Everything vanishes on
    restart, which is exactly what you want in a test and exactly what you do
    not want in production.
    """

    def __init__(self, max_history: int = 50):
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.max_history = max_history

    def get(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            versions = self._data.get(thread_id)
            # Shallow-copy so callers mutating the result cannot corrupt history.
            return dict(versions[-1]) if versions else None

    def put(self, thread_id: str, checkpoint: dict[str, Any]) -> None:
        with self._lock:
            versions = self._data.setdefault(thread_id, [])
            versions.append({**checkpoint, "ts": time.time()})
            if len(versions) > self.max_history:
                del versions[: -self.max_history]

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._data.pop(thread_id, None)

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        """Every checkpoint, oldest first — rewind to any step and branch."""
        with self._lock:
            return list(self._data.get(thread_id, []))


class FileCheckpointer(BaseCheckpointer):
    """One JSON file per thread. Survives restarts; needs no infrastructure.

    Writes are atomic (temp file + rename) so a crash mid-write cannot leave a
    truncated checkpoint that breaks every future resume.
    """

    def __init__(self, directory: str | Path = ".agentkit/checkpoints"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, thread_id: str) -> Path:
        # Sanitise: thread ids often come from user input or URLs.
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in thread_id)
        return self.dir / f"{safe}.json"

    def get(self, thread_id: str) -> dict[str, Any] | None:
        path = self._path(thread_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                raw = json.loads(path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                return None  # corrupt checkpoint: start fresh rather than crash
        return {
            "state": _decode(raw["state"]),
            "next": raw.get("next", []),
            "interrupted": raw.get("interrupted", False),
        }

    def put(self, thread_id: str, checkpoint: dict[str, Any]) -> None:
        path = self._path(thread_id)
        payload = {
            "state": _encode(checkpoint["state"]),
            "next": checkpoint.get("next", []),
            "interrupted": checkpoint.get("interrupted", False),
            "ts": time.time(),
        }
        with self._lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
            os.replace(tmp, path)  # atomic on POSIX and Windows

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._path(thread_id).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Long-term store
# ---------------------------------------------------------------------------
class BaseStore(ABC):
    """Namespaced key-value memory that outlives conversations.

    Namespaces are tuples — `("users", "u_123", "preferences")` — so you get
    natural multi-tenancy and prefix search without inventing a key format.
    """

    @abstractmethod
    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None: ...

    @abstractmethod
    def get(self, namespace: tuple[str, ...], key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def delete(self, namespace: tuple[str, ...], key: str) -> None: ...

    @abstractmethod
    def search(self, namespace: tuple[str, ...], query: str = "", limit: int = 10) -> list[dict[str, Any]]: ...


class InMemoryStore(BaseStore):
    """Reference implementation with naive substring search.

    `search` here is a keyword scan, not embeddings. That is intentional: it
    keeps the package dependency-free and makes the *interface* the thing you
    build against. Swap in pgvector or any vector DB behind the same four
    methods and nothing upstream changes.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault(namespace, {})[key] = {
                "key": key,
                "value": value,
                "updated_at": time.time(),
            }

    def get(self, namespace: tuple[str, ...], key: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._data.get(namespace, {}).get(key)
            return item["value"] if item else None

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        with self._lock:
            self._data.get(namespace, {}).pop(key, None)

    def search(self, namespace: tuple[str, ...], query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """Prefix-match the namespace, then substring-match the serialised value."""
        q = query.lower()
        hits: list[dict[str, Any]] = []
        with self._lock:
            for ns, items in self._data.items():
                if ns[: len(namespace)] != namespace:
                    continue
                for item in items.values():
                    if not q or q in json.dumps(item["value"], default=str).lower():
                        hits.append({"namespace": ns, **item})
        hits.sort(key=lambda i: i["updated_at"], reverse=True)
        return hits[:limit]


# ---------------------------------------------------------------------------
# Context-window management
# ---------------------------------------------------------------------------
def trim_messages(
    messages: list[Message],
    max_messages: int = 40,
    keep_system: bool = True,
) -> list[Message]:
    """Drop the oldest turns to stay inside the context window.

    Two rules that are easy to get wrong:

    1. System messages are always kept — they carry the agent's instructions,
       and silently trimming them makes the agent change personality mid-run.
    2. A `tool` message must never be orphaned from the assistant message that
       requested it. Providers reject a tool result whose tool_use id has no
       matching request, so we walk forward from the cut point until the
       history starts on a clean boundary.
    """
    if len(messages) <= max_messages:
        return list(messages)

    system = [m for m in messages if m.role == "system"] if keep_system else []
    rest = [m for m in messages if m.role != "system"]

    budget = max(0, max_messages - len(system))
    kept = rest[-budget:] if budget else []

    # Repair the boundary: never start with a dangling tool result.
    while kept and kept[0].role == "tool":
        kept.pop(0)

    return system + kept


def summarize_and_trim(
    messages: list[Message],
    summarizer,  # BaseChatModel
    max_messages: int = 40,
    keep_recent: int = 10,
) -> list[Message]:
    """Compress old turns into one system message instead of discarding them.

    Costs an extra model call but preserves the gist of a long session. Use it
    when the agent needs continuity ("as we discussed earlier..."); plain
    `trim_messages` is fine when it does not.
    """
    if len(messages) <= max_messages:
        return list(messages)

    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    old, recent = rest[:-keep_recent], rest[-keep_recent:]
    if not old:
        return list(messages)

    transcript = "\n".join(f"{m.role}: {m.content}" for m in old if m.content)
    prompt = [
        Message.system(
            "Summarise the conversation so far. Preserve decisions, facts "
            "established, user preferences, and open questions. Be terse."
        ),
        Message.user(transcript),
    ]
    summary = summarizer.invoke(prompt).message.content

    while recent and recent[0].role == "tool":
        recent.pop(0)

    return system + [Message.system(f"[Earlier conversation summary]\n{summary}")] + recent
