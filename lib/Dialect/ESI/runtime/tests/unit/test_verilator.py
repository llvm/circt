"""Unit tests for the Verilator cosim backend.

These tests exercise the command-generation and CMake-template logic of the
Verilator class *without* requiring the compiled ``esiCppAccel`` C++
extension. We achieve this by inserting a ``MagicMock`` for the extension
module before the real package is imported.
"""

import os
import shutil
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


def _make_verilator(run_dir,
                    top="TestTop",
                    debug=False,
                    dpi_so=None,
                    macros=None):
  """Create a Verilator instance with minimal setup."""
  sources = SourceFiles(top)
  if dpi_so is not None:
    sources.dpi_so = dpi_so
  return Verilator(
      sources=sources,
      run_dir=run_dir,
      debug=debug,
      make_default_logs=False,
      macro_definitions=macros,
  )


class TestCompileCommands:

  def test_uses_verilator_bin(self, tmp_path):
    v = _make_verilator(tmp_path)
    cmds = v.compile_commands()
    assert Path(cmds[0][0]).name == "verilator_bin"
    assert cmds[0][0] == v.verilator_bin

  def test_cmake_and_ninja_commands(self, tmp_path):
    v = _make_verilator(tmp_path)
    cmds = v.compile_commands()
    # cmake+ninja present => 4 steps, including a Python callback.
    if v._use_cmake:
      assert len(cmds) == 4
      assert callable(cmds[1])
      assert cmds[2][0] == "cmake"
      assert "-G" in cmds[2] and "Ninja" in cmds[2]
      assert cmds[3][0] == "ninja"

  def test_no_exe_or_build_flags_cmake(self, tmp_path):
    """When using cmake, --exe and --build should not appear."""
    v = _make_verilator(tmp_path)
    if not v._use_cmake:
      pytest.skip("cmake+ninja not available")
    cmd = v.compile_commands()[0]
    assert "--exe" not in cmd
    assert "--build" not in cmd

  def test_no_cflags_or_ldflags_cmake(self, tmp_path):
    """When using cmake, -CFLAGS and -LDFLAGS should not appear."""
    v = _make_verilator(tmp_path)
    if not v._use_cmake:
      pytest.skip("cmake+ninja not available")
    cmd = v.compile_commands()[0]
    assert "-CFLAGS" not in cmd
    assert "-LDFLAGS" not in cmd

  def test_driver_not_in_verilator_cmd_cmake(self, tmp_path):
    """When using cmake, driver.cpp should not be in the verilator command."""
    v = _make_verilator(tmp_path)
    if not v._use_cmake:
      pytest.skip("cmake+ninja not available")
    cmd = v.compile_commands()[0]
    assert not any("driver.cpp" in str(c) for c in cmd)

  def test_trace_flags_in_debug(self, tmp_path):
    v = _make_verilator(tmp_path, debug=True)
    cmd = v.compile_commands()[0]
    assert "--trace-fst" in cmd
    assert "--trace-structs" in cmd
    assert "--trace-underscore" in cmd

  def test_respects_verilator_path_env(self, tmp_path):
    with mock.patch.dict(os.environ,
                         {"VERILATOR_PATH": "/custom/verilator_bin"}):
      v = _make_verilator(tmp_path)
      assert v.verilator_bin == "/custom/verilator_bin"

  def test_verilator_path_redirects_perl_wrapper(self, tmp_path):
    with mock.patch.dict(os.environ, {"VERILATOR_PATH": "/usr/bin/verilator"}):
      v = _make_verilator(tmp_path)
      assert v.verilator_bin == str(Path("/usr/bin/verilator_bin"))

  def test_macro_definitions(self, tmp_path):
    v = _make_verilator(tmp_path, macros={"FOO": "BAR", "BAZ": None})
    cmd = v.compile_commands()[0]
    assert "+define+FOO=BAR" in cmd
    assert "+define+BAZ" in cmd


class TestMakeFallback:
  """Tests for the make fallback when cmake/ninja are not available."""

  def _make_no_cmake(self, tmp_path, **kwargs):
    """Create a Verilator instance that thinks cmake/ninja are missing."""
    v = _make_verilator(tmp_path, **kwargs)
    return v

  @pytest.fixture(autouse=True)
  def _hide_cmake(self):
    """Patch shutil.which so cmake and ninja appear absent."""
    original_which = shutil.which

    def _which_no_cmake(name, *args, **kwargs):
      if name in ("cmake", "ninja"):
        return None
      return original_which(name, *args, **kwargs)

    with mock.patch("shutil.which", side_effect=_which_no_cmake):
      yield

  def test_fallback_uses_make(self, tmp_path):
    v = self._make_no_cmake(tmp_path)
    cmds = v.compile_commands()
    assert len(cmds) == 2
    assert cmds[1][0] == "make"

  def test_fallback_has_exe_flag(self, tmp_path):
    v = self._make_no_cmake(tmp_path)
    cmd = v.compile_commands()[0]
    assert "--exe" in cmd

  def test_fallback_has_cflags(self, tmp_path):
    v = self._make_no_cmake(tmp_path)
    cmd = v.compile_commands()[0]
    assert "-CFLAGS" in cmd
    idx = cmd.index("-CFLAGS")
    assert "-DTOP_MODULE=TestTop" in cmd[idx + 1]

  def test_fallback_has_driver(self, tmp_path):
    v = self._make_no_cmake(tmp_path)
    cmd = v.compile_commands()[0]
    assert any("driver.cpp" in str(c) for c in cmd)

  def test_fallback_has_ldflags_with_dpi(self, tmp_path):
    v = self._make_no_cmake(tmp_path, dpi_so=["EsiCosimDpiServer"])
    cmd = v.compile_commands()[0]
    assert "-LDFLAGS" in cmd
    idx = cmd.index("-LDFLAGS")
    assert "-lEsiCosimDpiServer" in cmd[idx + 1]

  def test_fallback_no_ldflags_without_dpi(self, tmp_path):
    v = self._make_no_cmake(tmp_path, dpi_so=[])
    cmd = v.compile_commands()[0]
    assert "-LDFLAGS" not in cmd

  def test_fallback_trace_cflags_in_debug(self, tmp_path):
    v = self._make_no_cmake(tmp_path, debug=True)
    cmd = v.compile_commands()[0]
    idx = cmd.index("-CFLAGS")
    assert "-DTRACE" in cmd[idx + 1]

  def test_fallback_make_command(self, tmp_path):
    v = self._make_no_cmake(tmp_path, top="MyTop")
    cmds = v.compile_commands()
    make_cmd = cmds[1]
    assert make_cmd[0] == "make"
    assert "-C" in make_cmd
    assert "obj_dir" in make_cmd
    assert "-f" in make_cmd
    assert "VMyTop.mk" in make_cmd

  def test_fallback_exe_path(self, tmp_path):
    v = self._make_no_cmake(tmp_path, top="MyTop")
    with mock.patch.object(Path, "cwd", return_value=tmp_path):
      cmd = v.run_command(gui=False)
      assert cmd == [str(tmp_path / "obj_dir" / "VMyTop")]


class TestFindVerilatorRoot:

  def test_from_env(self, tmp_path):
    root = tmp_path / "verilator"
    root.mkdir()
    (root / "include").mkdir()
    (root / "include" / "verilated.h").touch()
    with mock.patch.dict(os.environ, {"VERILATOR_ROOT": str(root)}):
      v = _make_verilator(tmp_path)
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
        v = _make_verilator(tmp_path)
        found = v._find_verilator_root()
        assert found == root

  def test_raises_when_not_found(self, tmp_path):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("VERILATOR_ROOT", None)
      with mock.patch("shutil.which", return_value=None):
        v = _make_verilator(tmp_path)
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
      v = _make_verilator(tmp_path, dpi_so=[])
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
      v = _make_verilator(tmp_path, debug=True, dpi_so=[])
      build_dir = v._write_cmake(obj_dir)
      content = (build_dir / "CMakeLists.txt").read_text()
      assert "verilated_fst_c.cpp" in content
      assert "TRACE" in content


class TestRunCommand:

  def test_exe_path_cmake(self, tmp_path):
    v = _make_verilator(tmp_path, top="MyTop")
    if not v._use_cmake:
      pytest.skip("cmake+ninja not available")
    with mock.patch.object(Path, "cwd", return_value=tmp_path):
      cmd = v.run_command(gui=False)
      assert cmd == [str(tmp_path / "obj_dir" / "cmake_build" / "VMyTop")]
