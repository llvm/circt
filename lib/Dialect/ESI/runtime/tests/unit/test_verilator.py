"""Unit tests for the Verilator cosim backend.

These tests exercise the command-generation and CMake-template logic of the
Verilator class *without* requiring the compiled ``esiCppAccel`` C++
extension. We achieve this by inserting a ``MagicMock`` for the extension
module before the real package is imported.
"""

import os
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Provide a comprehensive mock for the native extension so we can import the
# pure-Python cosim modules without a full C++ build.
# ---------------------------------------------------------------------------
_accel_mock = MagicMock()
sys.modules["esiaccel.esiCppAccel"] = _accel_mock

# Now we can safely import the cosim modules.
from esiaccel.cosim.verilator import Verilator  # noqa: E402
from esiaccel.cosim.simulator import SourceFiles  # noqa: E402


def _make_verilator(top="TestTop", debug=False, dpi_so=None, macros=None):
  """Create a Verilator instance with minimal setup."""
  sources = SourceFiles(top)
  if dpi_so is not None:
    sources.dpi_so = dpi_so
  run_dir = Path("/tmp/test_run")
  return Verilator(
      sources=sources,
      run_dir=run_dir,
      debug=debug,
      make_default_logs=False,
      macro_definitions=macros,
  )


class TestCompileCommands:

  def test_uses_verilator_bin(self):
    v = _make_verilator()
    cmds = v.compile_commands()
    assert len(cmds) == 1
    assert cmds[0][0] == "verilator_bin"

  def test_no_exe_or_build_flags(self):
    v = _make_verilator()
    cmd = v.compile_commands()[0]
    assert "--exe" not in cmd
    assert "--build" not in cmd

  def test_no_cflags_or_ldflags(self):
    v = _make_verilator()
    cmd = v.compile_commands()[0]
    assert "-CFLAGS" not in cmd
    assert "-LDFLAGS" not in cmd

  def test_driver_not_in_command(self):
    v = _make_verilator()
    cmd = v.compile_commands()[0]
    assert not any("driver.cpp" in str(c) for c in cmd)

  def test_trace_flags_in_debug(self):
    v = _make_verilator(debug=True)
    cmd = v.compile_commands()[0]
    assert "--trace-fst" in cmd
    assert "--trace-params" in cmd

  def test_respects_verilator_path_env(self):
    with mock.patch.dict(os.environ, {"VERILATOR_PATH": "/custom/verilator"}):
      v = _make_verilator()
      assert v.verilator_bin == "/custom/verilator"

  def test_macro_definitions(self):
    v = _make_verilator(macros={"FOO": "BAR", "BAZ": None})
    cmd = v.compile_commands()[0]
    assert "+define+FOO=BAR" in cmd
    assert "+define+BAZ" in cmd


class TestFindVerilatorRoot:

  def test_from_env(self, tmp_path):
    root = tmp_path / "verilator"
    root.mkdir()
    (root / "include").mkdir()
    (root / "include" / "verilated.h").touch()
    with mock.patch.dict(os.environ, {"VERILATOR_ROOT": str(root)}):
      v = _make_verilator()
      assert v._find_verilator_root() == root

  def test_from_bin_in_path(self, tmp_path):
    root = tmp_path / "verilator"
    (root / "bin").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "include" / "verilated.h").touch()
    fake_bin = root / "bin" / "verilator_bin"
    fake_bin.touch()
    fake_bin.chmod(0o755)
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("VERILATOR_ROOT", None)
      with mock.patch("shutil.which", return_value=str(fake_bin)):
        v = _make_verilator()
        found = v._find_verilator_root()
        assert found == root

  def test_raises_when_not_found(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("VERILATOR_ROOT", None)
      with mock.patch("shutil.which", return_value=None):
        v = _make_verilator()
        with pytest.raises(RuntimeError):
          v._find_verilator_root()


class TestWriteCmake:

  def test_generates_cmake(self, tmp_path):
    obj_dir = tmp_path / "obj_dir"
    obj_dir.mkdir()
    root = tmp_path / "verilator"
    (root / "include").mkdir(parents=True)
    (root / "include" / "verilated.h").touch()
    with mock.patch.dict(os.environ, {"VERILATOR_ROOT": str(root)}):
      v = _make_verilator(dpi_so=[])
      build_dir = v._write_cmake(obj_dir)
      assert (build_dir / "CMakeLists.txt").exists()
      content = (build_dir / "CMakeLists.txt").read_text()
      assert "VTestTop" in content
      assert "verilated.cpp" in content
      assert "verilated_threads.cpp" in content
      assert "driver.cpp" in content

  def test_trace_sources_in_debug(self, tmp_path):
    obj_dir = tmp_path / "obj_dir"
    obj_dir.mkdir()
    root = tmp_path / "verilator"
    (root / "include").mkdir(parents=True)
    (root / "include" / "verilated.h").touch()
    with mock.patch.dict(os.environ, {"VERILATOR_ROOT": str(root)}):
      v = _make_verilator(debug=True, dpi_so=[])
      build_dir = v._write_cmake(obj_dir)
      content = (build_dir / "CMakeLists.txt").read_text()
      assert "verilated_fst_c.cpp" in content
      assert "TRACE" in content


class TestRunCommand:

  def test_exe_path(self, monkeypatch):
    monkeypatch.chdir("/tmp")
    v = _make_verilator(top="MyTop")
    cmd = v.run_command(gui=False)
    assert cmd == [str(Path("/tmp/obj_dir/cmake_build/VMyTop"))]

  def test_gui_raises(self):
    v = _make_verilator()
    with pytest.raises(RuntimeError):
      v.run_command(gui=True)
