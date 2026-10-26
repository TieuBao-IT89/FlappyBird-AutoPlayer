from threading import Event

import pytest

from flappy_auto.workers import LatestWorker


def test_slow_consumer_keeps_only_latest_pending_frame():
    entered, release = Event(), Event()
    consumed = []

    def consume(item):
        if item == 1:
            entered.set()
            assert release.wait(2)
        consumed.append(item)

    worker = LatestWorker(consume)
    try:
        worker.submit(1)
        assert entered.wait(2)
        # Submission must return while the renderer is still blocked.
        worker.submit(2)
        worker.submit(3)
        worker.submit(4)
        release.set()
        worker.flush()
        assert consumed == [1, 4]
    finally:
        release.set()
        worker.close()


def test_background_failure_surfaces_instead_of_deadlocking():
    def fail(item):
        raise ValueError('disk unavailable')

    worker = LatestWorker(fail)
    worker.submit('frame')
    with pytest.raises(RuntimeError, match='failed'):
        worker.flush()
    with pytest.raises(RuntimeError, match='failed'):
        worker.close()
