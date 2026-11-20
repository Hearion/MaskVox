"""Send UI events without blocking audio, inference, or conversion."""

from __future__ import annotations

import json
import logging
from collections import deque
from threading import Condition, Thread
from typing import Any

LOGGER = logging.getLogger(__name__)


class Events:
    def __init__(self) -> None:
        self._condition = Condition()
        self._messages: deque[dict[str, Any]] = deque()
        self._meter: dict[str, Any] | None = None
        self._window: Any = None
        self._closed = False
        self._worker = Thread(target=self._dispatch, name="maskvox-ui-events", daemon=True)
        self._worker.start()

    def ready(self, window: Any) -> None:
        with self._condition:
            self._window = window
            self._condition.notify()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        message = {"type": event, "payload": payload}
        with self._condition:
            if self._closed:
                return
            if event == "meter":
                self._meter = message
            else:
                if event == "status" and payload.get("state") != "running":
                    self._meter = None
                self._messages.append(message)
            self._condition.notify()

    def _dispatch(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed
                    or (self._window is not None and (self._messages or self._meter is not None))
                )
                if self._closed:
                    return
                if self._messages:
                    message = self._messages.popleft()
                else:
                    message, self._meter = self._meter, None
                window = self._window
            try:
                data = json.dumps(message, ensure_ascii=True, allow_nan=False)
                window.evaluate_js(f"window.onEngineEvent?.({data})")
            except Exception:
                if not self._closed:
                    LOGGER.exception("界面事件发送失败")

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._messages.clear()
            self._meter = None
            self._condition.notify_all()
