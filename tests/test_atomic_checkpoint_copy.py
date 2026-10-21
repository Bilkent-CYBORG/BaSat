"""Independent atomic byte copies retain authoritative checkpoint recovery."""
import json

import pytest

import basat
from test_advers_camera_ready_lcb_execution import make


def test_checkpoint_serializes_once_and_copies_exact_independent_bytes(tmp_path, monkeypatch):
    experiment = make(directory=tmp_path, gp_backend="fixed_gpy")
    original = basat.Experiment._atomic_torch_save
    calls = []
    def serialize(payload, target):
        calls.append(target)
        return original(payload, target)
    monkeypatch.setattr(basat.Experiment, "_atomic_torch_save", staticmethod(serialize))
    experiment.save_checkpoint(phase="test", archive=True)
    assert len(calls) == 1
    manifest = json.loads((tmp_path / "latest.json").read_text())
    authority = tmp_path / manifest["checkpoint"]
    latest, archive = tmp_path / "checkpoint-latest.pt", tmp_path / "checkpoint-00000000.pt"
    assert authority.read_bytes() == latest.read_bytes() == archive.read_bytes()
    assert len({path.stat().st_ino for path in (authority, latest, archive)}) == 3
    latest.write_bytes(b"corrupt non-authoritative inspection copy")
    restored = make(directory=tmp_path, resume=True, gp_backend="fixed_gpy")
    assert restored.completed_rounds == 0 and authority.read_bytes() == archive.read_bytes()


def test_interrupted_copy_keeps_previous_authoritative_manifest(tmp_path, monkeypatch):
    experiment = make(directory=tmp_path, gp_backend="fixed_gpy")
    experiment.save_checkpoint(phase="first")
    before = (tmp_path / "latest.json").read_bytes()
    latest = (tmp_path / "checkpoint-latest.pt").read_bytes()
    import basat.experiment as module
    def fail(incoming, outgoing):
        outgoing.write(incoming.read(31))
        raise OSError("interrupted independent copy")
    monkeypatch.setattr(module.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="interrupted"):
        experiment.save_checkpoint(phase="second")
    assert (tmp_path / "latest.json").read_bytes() == before
    assert (tmp_path / "checkpoint-latest.pt").read_bytes() == latest
    assert not list(tmp_path.glob(".*.tmp"))
    restored = make(directory=tmp_path, resume=True, gp_backend="fixed_gpy")
    assert restored.completed_rounds == 0

