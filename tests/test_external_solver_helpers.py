"""Tests for external-solver helper functions in adapters/pyscf.py.

No PySCF dependency — these helpers are pure Python/numpy/tarfile.
"""

import types

import numpy as np

from spot_checkpoint.adapters.pyscf import _find_mps_dir, _tar_directory, _untar_directory


class TestTarUntar:
    def test_roundtrip(self, tmp_path):
        """Create a directory tree, tar it, untar into a fresh dir, verify contents."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("hello")
        sub = src / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("world")

        arr = _tar_directory(src)

        dest = tmp_path / "dest"
        dest.mkdir()
        _untar_directory(arr, dest)

        assert (dest / "a.txt").read_text() == "hello"
        assert (dest / "sub" / "b.txt").read_text() == "world"

    def test_empty_directory(self, tmp_path):
        """Tar an empty directory — should produce a valid archive with no errors."""
        src = tmp_path / "empty"
        src.mkdir()

        arr = _tar_directory(src)

        dest = tmp_path / "out"
        dest.mkdir()
        _untar_directory(arr, dest)  # should not raise

    def test_array_is_uint8(self, tmp_path):
        """Output dtype must be np.uint8."""
        src = tmp_path / "d"
        src.mkdir()
        (src / "f.txt").write_text("x")

        arr = _tar_directory(src)
        assert arr.dtype == np.uint8

    def test_data_is_valid_gzip(self, tmp_path):
        """First two bytes of the raw data must be the gzip magic number 0x1f 0x8b."""
        src = tmp_path / "d"
        src.mkdir()
        (src / "f.txt").write_text("y")

        arr = _tar_directory(src)
        raw = arr.tobytes()
        assert raw[:2] == b"\x1f\x8b"


class TestFindMpsDir:
    def test_finds_fcisolver_scratch(self, tmp_path):
        """mc.fcisolver.scratch pointing to a real dir is found."""
        real_dir = tmp_path / "scratch"
        real_dir.mkdir()

        mc = types.SimpleNamespace(
            fcisolver=types.SimpleNamespace(scratch=str(real_dir)),
            ci=types.SimpleNamespace(),
        )
        result = _find_mps_dir(mc)
        assert result == real_dir

    def test_finds_ci_path_attr(self, tmp_path):
        """mc.ci.path pointing to a real dir is found when fcisolver has no attrs."""
        real_dir = tmp_path / "mps"
        real_dir.mkdir()

        mc = types.SimpleNamespace(
            fcisolver=types.SimpleNamespace(),  # no matching attrs
            ci=types.SimpleNamespace(path=str(real_dir)),
        )
        result = _find_mps_dir(mc)
        assert result == real_dir

    def test_returns_none_when_no_attrs(self, tmp_path):
        """Returns None when neither fcisolver nor ci have any matching attributes."""
        mc = types.SimpleNamespace(
            fcisolver=types.SimpleNamespace(),
            ci=types.SimpleNamespace(),
        )
        assert _find_mps_dir(mc) is None

    def test_ignores_nonexistent_path(self, tmp_path):
        """An attribute pointing to a non-existent path is skipped → None."""
        mc = types.SimpleNamespace(
            fcisolver=types.SimpleNamespace(scratch=str(tmp_path / "does_not_exist")),
            ci=None,
        )
        assert _find_mps_dir(mc) is None
