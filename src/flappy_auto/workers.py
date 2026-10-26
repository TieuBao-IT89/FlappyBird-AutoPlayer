"""Bounded background work: a slow display or disk cannot queue old frames."""
from threading import Condition, Event, Thread

import cv2


class LatestWorker:
    def __init__(self, consume, cleanup=None):
        self._consume, self._cleanup = consume, cleanup
        self._condition = Condition()
        self._pending = None
        self._busy = self._closed = False
        self._error = None
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, item):
        with self._condition:
            self._check()
            if not self._closed:
                self._pending = item
                self._condition.notify()

    def _check(self):
        if self._error is not None:
            raise RuntimeError('Background display/recording failed') from self._error

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending is not None or self._closed)
                    if self._pending is None:
                        break
                    item, self._pending = self._pending, None
                    self._busy = True
                self._consume(item)
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error, self._busy, self._pending = error, False, None
                self._condition.notify_all()
        finally:
            if self._cleanup is not None:
                self._cleanup()

    def flush(self):
        with self._condition:
            self._condition.wait_for(lambda: self._pending is None and not self._busy)
            self._check()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError('Background display/recording did not stop')
        self._check()


class DebugWindow:
    def __init__(self):
        self.stop_requested = Event()
        self._name = 'Flappy AutoPlayer - press Q to quit'
        self._created = False
        self._worker = LatestWorker(self._show, self._destroy)

    def _show(self, frame):
        cv2.imshow(self._name, frame)
        self._created = True
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            self.stop_requested.set()

    def _destroy(self):
        if self._created:
            cv2.destroyWindow(self._name)

    def show(self, frame):
        self._worker.submit(frame)

    def flush(self):
        self._worker.flush()

    def close(self):
        self._worker.close()
