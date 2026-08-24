# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Write-to-staging-then-publish helpers.

Long-running operations must never write partial results to a user-visible
output path. These helpers give an operation a private working directory on
the same volume as its destination, and move the result into place only after
the operation *and* any validation have succeeded.

See also: temp/hard-cancel-final.md
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


STAGING_PREFIX = ".winml-staging-"


@contextlib.contextmanager
def scratch_dir(directory: str | Path) -> Iterator[Path]:
    """Yield a temporary directory inside *directory*.

    Creating it inside the destination directory guarantees it lives on the
    same volume, which is what makes the final :meth:`Path.replace` atomic.

    The directory is removed on normal exit. If the process is hard-terminated
    (see ``hard_kill_on_ctrl_c``) the cleanup never runs and the directory is
    left behind on purpose -- an identifiable leftover is preferable to a
    half-written model at the real output path.
    """
    parent = Path(directory)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix=STAGING_PREFIX) as path:
        yield Path(path)


@contextlib.contextmanager
def publish_on_success(destination: str | Path) -> Iterator[Path]:
    """Yield a working path to write, publishing it to *destination* on success.

    The caller writes to the yielded path (and may write sidecar files next to
    it). When the block exits normally, everything in the staging directory is
    moved next to *destination*, with the primary file moved last.

    Publishing deliberately does **not** happen in a ``finally``: on an
    exception the staging directory is left as-is and *destination* keeps its
    previous contents. Validation belongs inside the block, so that a model
    that fails its checks is never published::

        with publish_on_success(output) as working:
            run_long_operation(output=working)
            validate(working)

    Args:
        destination: Final path the operation's primary output belongs at.

    Yields:
        Path inside a staging directory to write the primary output to.

    Raises:
        FileNotFoundError: If nothing was written to the yielded path.
        IsADirectoryError: If *destination* exists and is a directory.
    """
    destination = Path(destination).resolve()
    if destination.is_dir():
        msg = f"destination is an existing directory, not a file: {destination}"
        raise IsADirectoryError(msg)

    with scratch_dir(destination.parent) as staging:
        yield staging / destination.name
        publish_staged(staging, destination)


def publish_staged(staging: str | Path, destination: str | Path) -> None:
    """Move everything in *staging* next to *destination*, primary file last.

    Order matters: an ``.onnx`` file references its external-data sidecar by
    filename, so the sidecars must be in place before the ``.onnx`` appears.
    Publishing the primary file first would leave a window in which the model
    at *destination* points at a file that does not exist yet.

    Prefer :func:`publish_on_success`. Call this directly only when producing
    nothing is a legitimate outcome the caller has to report itself, since this
    still treats a missing primary as an error.

    Args:
        staging: Directory holding the operation's output.
        destination: Final path for the primary file; siblings go beside it.

    Raises:
        FileNotFoundError: If no file named after *destination* is in *staging*.
    """
    staging = Path(staging)
    destination = Path(destination)
    primary = staging / destination.name
    if not primary.is_file():
        msg = f"nothing was written to {primary}"
        raise FileNotFoundError(msg)

    for produced in sorted(staging.iterdir()):
        if produced == primary:
            continue
        if not produced.is_file():
            # Directory outputs are out of scope: Windows cannot atomically
            # replace a non-empty directory, so no directory-swap semantics.
            logger.warning("Not publishing non-file staging entry: %s", produced.name)
            continue
        produced.replace(destination.parent / produced.name)

    primary.replace(destination)
