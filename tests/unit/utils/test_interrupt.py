# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for hard cancellation of long-running native calls.

These tests only cover handler installation and restoration. The actual
termination behaviour cannot be tested in-process: the whole point of
``SIG_DFL`` is that the OS kills the process without running any Python, so a
test that really delivered ``CTRL_C_EVENT`` would take pytest down with it.
Verifying that Ctrl+C terminates the CLI within 2s is a manual step on a
machine with the target hardware (see temp/hard-cancel-final.md, slice A).
"""

from __future__ import annotations

import contextlib
import signal
import threading

import pytest

from winml.modelkit.utils import hard_kill_on_ctrl_c


def test_installs_default_handler_inside_block() -> None:
    """Inside the block, SIGINT is the OS default action."""
    with hard_kill_on_ctrl_c():
        assert signal.getsignal(signal.SIGINT) is signal.SIG_DFL


def test_restores_previous_handler_on_normal_exit() -> None:
    """The handler in place before the block is put back afterwards."""
    before = signal.getsignal(signal.SIGINT)

    with hard_kill_on_ctrl_c():
        pass

    assert signal.getsignal(signal.SIGINT) is before


def test_restores_previous_handler_on_exception() -> None:
    """An exception escaping the block still restores the handler."""
    before = signal.getsignal(signal.SIGINT)

    with pytest.raises(RuntimeError, match="native call failed"), hard_kill_on_ctrl_c():
        msg = "native call failed"
        raise RuntimeError(msg)

    assert signal.getsignal(signal.SIGINT) is before


def test_restores_custom_python_handler() -> None:
    """A caller-installed Python handler survives the block."""

    def custom(_signum: int, _frame: object) -> None:  # pragma: no cover - never called
        return None

    previous = signal.signal(signal.SIGINT, custom)
    try:
        with hard_kill_on_ctrl_c():
            assert signal.getsignal(signal.SIGINT) is signal.SIG_DFL
        assert signal.getsignal(signal.SIGINT) is custom
    finally:
        signal.signal(signal.SIGINT, previous)


def test_nesting_leaves_default_active_in_inner_block() -> None:
    """Nested use is safe: the inner block restores what the outer installed."""
    before = signal.getsignal(signal.SIGINT)

    with hard_kill_on_ctrl_c():
        with hard_kill_on_ctrl_c():
            assert signal.getsignal(signal.SIGINT) is signal.SIG_DFL
        # The outer block is still in force here.
        assert signal.getsignal(signal.SIGINT) is signal.SIG_DFL

    assert signal.getsignal(signal.SIGINT) is before


def test_rejects_use_outside_main_thread() -> None:
    """Signal handlers are process-wide; only the main thread may install them."""
    failures: list[BaseException] = []

    def target() -> None:
        try:
            with hard_kill_on_ctrl_c():
                pass
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=target)
    worker.start()
    worker.join()

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "main thread" in str(failures[0])


def test_does_not_disturb_handler_when_rejected_off_main_thread() -> None:
    """The guard must reject before touching process-wide state."""
    before = signal.getsignal(signal.SIGINT)
    observed: list[object] = []

    def target() -> None:
        with contextlib.suppress(RuntimeError), hard_kill_on_ctrl_c():
            pass
        # Whether it raised or not, the process-wide handler must be unchanged.
        observed.append(signal.getsignal(signal.SIGINT))

    worker = threading.Thread(target=target)
    worker.start()
    worker.join()

    assert observed == [before]
    assert signal.getsignal(signal.SIGINT) is before
