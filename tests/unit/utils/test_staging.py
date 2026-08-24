# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for write-to-staging-then-publish helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from winml.modelkit.utils import (
    STAGING_PREFIX,
    publish_on_success,
    publish_staged,
    scratch_dir,
)


def _staging_entries(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p.name.startswith(STAGING_PREFIX)]


def test_scratch_dir_creates_inside_target_and_cleans_up(tmp_path: Path) -> None:
    """The scratch dir lives inside the target (same volume) and is removed."""
    target = tmp_path / "out"

    with scratch_dir(target) as staging:
        assert staging.is_dir()
        assert staging.parent == target
        assert staging.name.startswith(STAGING_PREFIX)
        (staging / "junk.bin").write_text("junk")

    assert target.is_dir()
    assert _staging_entries(target) == []


def test_scratch_dir_creates_missing_parents(tmp_path: Path) -> None:
    """A destination directory that does not exist yet is created."""
    target = tmp_path / "deeply" / "nested" / "out"

    with scratch_dir(target) as staging:
        assert staging.parent == target

    assert target.is_dir()


def test_publish_on_success_publishes_after_clean_exit(tmp_path: Path) -> None:
    """The written file only appears at the destination once the block exits."""
    destination = tmp_path / "model.onnx"

    with publish_on_success(destination) as working:
        working.write_text("new")
        assert not destination.exists()

    assert destination.read_text() == "new"
    assert _staging_entries(tmp_path) == []


def test_publish_on_success_replaces_existing_destination(tmp_path: Path) -> None:
    """An existing output survives until the new one is ready, then is replaced."""
    destination = tmp_path / "model.onnx"
    destination.write_text("old")

    with publish_on_success(destination) as working:
        working.write_text("new")
        assert destination.read_text() == "old"

    assert destination.read_text() == "new"


def test_publish_on_success_publishes_sidecars_alongside(tmp_path: Path) -> None:
    """Every file in staging lands next to the destination."""
    destination = tmp_path / "model.onnx"

    with publish_on_success(destination) as working:
        working.write_text("graph")
        (working.parent / "model.onnx.data").write_text("weights")

    assert destination.read_text() == "graph"
    assert (tmp_path / "model.onnx.data").read_text() == "weights"
    assert _staging_entries(tmp_path) == []


def test_publish_on_success_writes_sidecars_before_primary(tmp_path: Path) -> None:
    """The .onnx must never be visible before the sidecar it references.

    Checked by recording the order of the publishing replace() calls: the
    primary must come last, so the model is never in place while a sidecar it
    points at is still missing.
    """
    destination = tmp_path / "model.onnx"
    sidecar = tmp_path / "model.onnx.data"

    original_replace = Path.replace
    seen: list[str] = []

    def recording_replace(self: Path, target):
        seen.append(Path(target).name)
        return original_replace(self, target)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "replace", recording_replace)
        with publish_on_success(destination) as working:
            working.write_text("graph")
            (working.parent / "model.onnx.data").write_text("weights")

    assert seen == [sidecar.name, destination.name]


def test_publish_on_success_keeps_destination_on_exception(tmp_path: Path) -> None:
    """An exception inside the block leaves the previous output untouched."""
    destination = tmp_path / "model.onnx"
    destination.write_text("old")

    with pytest.raises(RuntimeError, match="boom"), publish_on_success(destination) as working:
        working.write_text("new")
        msg = "boom"
        raise RuntimeError(msg)

    assert destination.read_text() == "old"


def test_publish_on_success_keeps_destination_on_validation_failure(tmp_path: Path) -> None:
    """Validation inside the block gates publication."""
    destination = tmp_path / "model.onnx"
    destination.write_text("old")

    def validate(_path: Path) -> None:
        msg = "model failed validation"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="failed validation"), publish_on_success(
        destination
    ) as working:
        working.write_text("corrupt")
        validate(working)

    assert destination.read_text() == "old"


def test_publish_on_success_raises_when_nothing_written(tmp_path: Path) -> None:
    """A silent no-op operation is an error, not a successful publish."""
    destination = tmp_path / "model.onnx"
    destination.write_text("old")

    with pytest.raises(FileNotFoundError, match="nothing was written"), publish_on_success(
        destination
    ):
        pass

    assert destination.read_text() == "old"


def test_publish_on_success_rejects_directory_destination(tmp_path: Path) -> None:
    """A destination that is an existing directory cannot be replaced."""
    destination = tmp_path / "model.onnx"
    destination.mkdir()

    with pytest.raises(IsADirectoryError), publish_on_success(destination):
        pass


def test_publish_on_success_skips_directories_in_staging(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Directory outputs are out of scope and are reported, not silently dropped."""
    destination = tmp_path / "model.onnx"

    with caplog.at_level("WARNING"), publish_on_success(destination) as working:
        working.write_text("graph")
        (working.parent / "ctx_dir").mkdir()

    assert destination.read_text() == "graph"
    assert not (tmp_path / "ctx_dir").exists()
    assert "ctx_dir" in caplog.text


def test_publish_staged_moves_bundle_next_to_destination(tmp_path: Path) -> None:
    """The compile path publishes a whole directory of artifacts at once."""
    destination = tmp_path / "model_ctx.onnx"
    destination.write_text("previous")

    with scratch_dir(tmp_path) as staging:
        (staging / destination.name).write_text("compiled")
        (staging / "model_ctx.bin").write_text("binary")
        assert destination.read_text() == "previous"

        publish_staged(staging, destination)

    assert destination.read_text() == "compiled"
    assert (tmp_path / "model_ctx.bin").read_text() == "binary"
    assert _staging_entries(tmp_path) == []


def test_publish_staged_raises_when_primary_missing(tmp_path: Path) -> None:
    """Sidecars alone are not a publishable result."""
    destination = tmp_path / "model_ctx.onnx"
    destination.write_text("previous")

    with scratch_dir(tmp_path) as staging:
        (staging / "model_ctx.bin").write_text("binary")

        with pytest.raises(FileNotFoundError, match="nothing was written"):
            publish_staged(staging, destination)

    assert destination.read_text() == "previous"
    assert not (tmp_path / "model_ctx.bin").exists()
