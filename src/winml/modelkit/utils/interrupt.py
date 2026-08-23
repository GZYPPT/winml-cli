# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Hard cancellation for long-running native calls.

Scope: the Windows console + foreground process case that the CLI actually
runs in. Do not reach for this to make unrelated code "more interruptible" --
it trades away every cleanup hook, and it is only worth that on a native call
that provably cannot be interrupted any other way.

See also: temp/hard-cancel-final.md
"""

from __future__ import annotations

import contextlib
import signal
import threading
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def hard_kill_on_ctrl_c() -> Iterator[None]:
    """Let Ctrl+C terminate the process at once for the duration of the block.

    A long native call never returns to the bytecode loop, so no Python handler
    can run -- not even one that only arms a harder one, which is why the usual
    "press Ctrl+C again to force quit" pattern does nothing here. Restoring the
    OS default action is the only thing that still reacts, at the cost of every
    cleanup hook: no ``finally``, no ``atexit``, no ``except``.

    Wrap the single native call, never a whole command, so that the blast
    radius stays minimal and Python code outside the block keeps normal
    cleanup semantics. Anything the operation has written so far is left as-is,
    which is why it must be writing to a staging directory (see
    :func:`~winml.modelkit.utils.staging.publish_on_success`) rather than to a
    user-visible output path.

    Nesting is safe: the inner block restores ``SIG_DFL``, which is what the
    outer block installed.

    Raises:
        RuntimeError: If called from any thread other than the main thread.
            Signal handlers are process-wide and can only be installed from the
            main thread.
    """
    if threading.current_thread() is not threading.main_thread():
        msg = "hard_kill_on_ctrl_c() must be called from the main thread"
        raise RuntimeError(msg)

    previous = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        yield
    finally:
        # A handler installed from C rather than from Python reads back as
        # None, which signal.signal() refuses to take. Nothing better is
        # available in that case, so leave the OS default in place.
        signal.signal(signal.SIGINT, signal.SIG_DFL if previous is None else previous)
