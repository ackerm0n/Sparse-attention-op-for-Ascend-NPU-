#!/usr/bin/env python3
"""Shared, dependency-light helpers for Triangle v2 NPU validation."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator


DEFAULT_NPU_LOCK = Path(
    os.environ.get(
        "TRIANGLEMIX_NPU_LOCK",
        str(Path(tempfile.gettempdir()) / "trianglemix_npu.lock"),
    )
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def exclusive_npu_lock(
    path: Path = DEFAULT_NPU_LOCK,
    timeout_seconds: float = 300.0,
) -> Iterator[None]:
    """Acquire the project-wide NPU lock without an unbounded wait."""

    if timeout_seconds < 0:
        raise ValueError("lock timeout must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for NPU lock {path}"
                    )
                time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))

        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            f"pid={os.getpid()} acquired={time.time():.6f}\n".encode(),
        )
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
