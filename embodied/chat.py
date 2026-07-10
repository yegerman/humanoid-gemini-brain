"""Terminal chat input on a background thread -> command queue.

Keeps the MuJoCo render loop responsive in the main thread while the user types
commands ("go to the center of the stage", "raise your right hand", "quit").
"""
from __future__ import annotations

import queue
import threading


class Chat:
    def __init__(self) -> None:
        self.q: "queue.Queue[str]" = queue.Queue()
        self.quit = threading.Event()      # set the instant the user asks to quit
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._t.start()

    def _loop(self) -> None:
        print("\nChat ready. Try: 'go to the center of the stage' | 'raise your right hand' | 'quit'")
        while not self._stop.is_set():
            try:
                line = input("you> ").strip()
            except (EOFError, OSError):
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                self.quit.set()      # signal the render loop immediately
                break
            self.q.put(line)

    def poll(self) -> str | None:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
