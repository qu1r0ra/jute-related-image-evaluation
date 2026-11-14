"""A non-blocking, process-wide lock on one file.

The operating system drops the lock when its holder dies, so a killed queue
never leaves a run directory locked behind it.
"""

import os
import sys
from pathlib import Path
from types import TracebackType


class FitLockedError(RuntimeError):
    """Another writer holds this fit's run directory."""


class FileLock:
    """Exclusive lock on ``path``; acquiring it never waits."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    def acquire(self) -> None:
        """Take the lock.

        Raises:
            FitLockedError: another process or lock object already holds it.
        """
        if self._descriptor is not None:
            raise FitLockedError(f"{self.path} is already locked by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _lock(descriptor)
        except OSError as error:
            os.close(descriptor)
            raise FitLockedError(f"{self.path} is held by another writer") from error
        self._descriptor = descriptor

    def release(self) -> None:
        if self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        try:
            _unlock(descriptor)
        finally:
            os.close(descriptor)

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


if sys.platform == "win32":
    import msvcrt

    def _lock(descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)

    def _unlock(descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
