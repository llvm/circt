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
from esiaccel.cosim.simulator import (
    available_simulators,  # noqa: E402
    is_simulator_available,
    SourceFiles)


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


requires_verilator_bin = pytest.mark.skipif(
    not is_simulator_available("verilator"), reason="verilator not found")


class TestSimulatorDiscovery:

  def test_unknown_simulator_raises(self):
    with pytest.raises(ValueError):
      is_simulator_available("bogus")

  def test_verilator_unavailable_without_bin(self, monkeypatch):
    monkeypatch.delenv("VERILATOR_PATH", raising=False)
    monkeypatch.delenv("VERILATOR_ROOT", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert not is_simulator_available("verilator")
    assert "verilator" not in available_simulators()

  def test_verilator_available_from_env_path(self, monkeypatch, tmp_path):
    root = tmp_path / "verilator"
    (root / "bin").mkdir(parents=True)
    pkg_root = root / "share" / "verilator"
    (pkg_root / "include").mkdir(parents=True)
    (pkg_root / "include" / "verilated.h").touch()
    fake_bin = root / "bin" / "verilator_bin"
    fake_bin.touch()

    monkeypatch.setenv("VERILATOR_PATH", str(fake_bin))
    monkeypatch.delenv("VERILATOR_ROOT", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert is_simulator_available("verilator")
    assert "verilator" in available_simulators()

  def test_invalid_verilator_path_env_raises(self, monkeypatch, tmp_path):
    monkeypatch.setenv("VERILATOR_PATH",
                       str(tmp_path / "missing" / "verilator_bin"))
    monkeypatch.delenv("VERILATOR_ROOT", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="VERILATOR_PATH"):
      is_simulator_available("verilator")

  def test_invalid_verilator_root_env_raises(self, monkeypatch, tmp_path):
    root = tmp_path / "verilator"
    (root / "bin").mkdir(parents=True)
    pkg_root = root / "share" / "verilator"
    pkg_root.mkdir(parents=True)
    fake_bin = root / "bin" / "verilator_bin"
    fake_bin.touch()

    monkeypatch.setenv("VERILATOR_PATH", str(fake_bin))
    monkeypatch.setenv("VERILATOR_ROOT", str(pkg_root))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="VERILATOR_ROOT"):
      is_simulator_available("verilator")

  def test_questa_unavailable_without_vsim(self, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert not is_simulator_available("questa")

  def test_questa_available_from_path(self, monkeypatch):
    monkeypatch.delenv("VERILATOR_PATH", raising=False)
    monkeypatch.delenv("VERILATOR_ROOT", raising=False)

    def _which(name):
      if name == "vsim":
        return "C:/questa/vsim.exe"
      return None

    monkeypatch.setattr(shutil, "which", _which)
    assert is_simulator_available("questa")
    assert available_simulators() == ["questa"]


class TestCompileCommands:

  @requires_verilator_bin
  def test_uses_verilator_bin(self, tmp_path):
    v = _make_verilator(tmp_path)
    cmds = v.compile_commands()
    assert Path(cmds[0][0]).stem == "verilator_bin"
    assert Path(cmds[0][0]) == v.verilator_bin

  @requires_verilator_bin
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

  @requires_verilator_bin
  def test_no_exe_or_build_flags_cmake(self, tmp_path):
    """When using cmake, --exe and --build should not appear."""
    v = _make_verilator(tmp_path)
    if not v._use_cmake:
      pytest.skip("cmake+ninja not available")
    cmd = v.compile_commands()[0]
    assert "--exe" not in cmd
    assert "--build" not in cmd

  @requires_verilator_bin
  def test_no_cflags_or_ldflags_cmake(self, tmp_path):
    """When using cmake, -CFLAGS and -LDFLAGS should not appear."""
    v = _make_verilator(tmp_path)
    if not v._use_cmake:
      pytest.skip("cmake+ninja not available")
    cmd = v.compile_commands()[0]
    assert "-CFLAGS" not in cmd
    assert "-LDFLAGS" not in cmd

  @requires_verilator_bin
  def test_driver_not_in_verilator_cmd_cmake(self, tmp_path):
    """When using cmake, driver.cpp should not be in the verilator command."""
    v = _make_verilator(tmp_path)
    if not v._use_cmake:
      pytest.skip("cmake+ninja not available")
    cmd = v.compile_commands()[0]
    assert not any("driver.cpp" in str(c) for c in cmd)

  @requires_verilator_bin
  def test_trace_flags_in_debug(self, tmp_path):
    v = _make_verilator(tmp_path, debug=True)
    cmd = v.compile_commands()[0]
    assert "--trace-fst" in cmd
    assert "--trace-structs" in cmd
    assert "--trace-underscore" in cmd

  def test_respects_verilator_path_env(self, tmp_path):
    fake_bin = tmp_path / "custom" / "verilator_bin"
    fake_bin.parent.mkdir()
    fake_bin.touch()
    with mock.patch.dict(os.environ, {"VERILATOR_PATH": str(fake_bin)}):
      v = _make_verilator(tmp_path)
      assert v.verilator_bin == fake_bin.resolve()

  def test_verilator_path_redirects_perl_wrapper(self, tmp_path):
    fake_wrapper = tmp_path / "usr" / "bin" / "verilator"
    fake_bin = fake_wrapper.parent / "verilator_bin"
    fake_wrapper.parent.mkdir(parents=True)
    fake_wrapper.touch()
    fake_bin.touch()
    with mock.patch.dict(os.environ, {"VERILATOR_PATH": str(fake_wrapper)}):
      v = _make_verilator(tmp_path)
      assert v.verilator_bin == fake_bin.resolve()

  def test_verilator_path_overrides_path(self, tmp_path):
    env_root = tmp_path / "env-verilator"
    path_root = tmp_path / "path-verilator"
    (env_root / "bin").mkdir(parents=True)
    (path_root / "bin").mkdir(parents=True)
    env_bin = env_root / "bin" / "verilator_bin"
    path_bin = path_root / "bin" / "verilator_bin"
    env_bin.touch()
    path_bin.touch()

    with mock.patch.dict(os.environ, {"VERILATOR_PATH": str(env_bin)}):
      with mock.patch("shutil.which", return_value=str(path_bin)):
        assert Verilator._find_verilator_bin() == env_bin.resolve()

  def test_compile_commands_requires_verilator_bin(self, tmp_path):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("VERILATOR_ROOT", None)
      os.environ.pop("VERILATOR_PATH", None)
      with mock.patch("shutil.which", return_value=None):
        v = _make_verilator(tmp_path)
        with pytest.raises(RuntimeError, match="Cannot find verilator_bin"):
          v.compile_commands()

  @requires_verilator_bin
  def test_macro_definitions(self, tmp_path):
    v = _make_verilator(tmp_path, macros={"FOO": "BAR", "BAZ": None})
    cmd = v.compile_commands()[0]
    assert "+define+FOO=BAR" in cmd
    assert "+define+BAZ" in cmd


@requires_verilator_bin
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
    exe_name = "VMyTop.exe" if os.name == "nt" else "VMyTop"
    with mock.patch.object(Path, "cwd", return_value=tmp_path):
      cmd = v.run_command(gui=False)
      assert cmd == [str(tmp_path / "obj_dir" / exe_name)]


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
    pkg_root = root / "share" / "verilator"
    (pkg_root / "include").mkdir(parents=True)
    (pkg_root / "include" / "verilated.h").touch()
    fake_bin = root / "bin" / "verilator_bin"
    fake_bin.touch()
    fake_bin.chmod(0o755)
    with mock.patch.dict(os.environ, {}, clear=False):
      # Clear both root and path env vars so the real Verilator install
      # doesn't shadow the fake bin created for this test.
      os.environ.pop("VERILATOR_ROOT", None)
      os.environ.pop("VERILATOR_PATH", None)
      with mock.patch("shutil.which", return_value=str(fake_bin)):
        v = _make_verilator(tmp_path)
        found = v._find_verilator_root()
        assert found == pkg_root

  def test_returns_none_when_not_found(self, tmp_path):
    with mock.patch.dict(os.environ, {}, clear=False):
      # Clear both env vars so the real Verilator install doesn't satisfy
      # root detection before the RuntimeError can be raised.
      os.environ.pop("VERILATOR_ROOT", None)
      os.environ.pop("VERILATOR_PATH", None)
      with mock.patch("shutil.which", return_value=None):
        v = _make_verilator(tmp_path)
        assert v._find_verilator_root() is None

  def test_invalid_env_raises(self, tmp_path):
    root = tmp_path / "verilator"
    root.mkdir()
    with mock.patch.dict(os.environ, {"VERILATOR_ROOT": str(root)}):
      v = _make_verilator(tmp_path)
      with pytest.raises(RuntimeError, match="VERILATOR_ROOT"):
        v._find_verilator_root()


class TestWriteCmake:

  def test_generates_cmake(self, tmp_path):
    obj_dir = tmp_path / "obj_dir"
    obj_dir.mkdir()
    generated_sources = [obj_dir / "VTestTop.cpp"]
    root = tmp_path / "verilator"
    (root / "include").mkdir(parents=True)
    (root / "include" / "verilated.h").touch()
    with mock.patch.dict(os.environ, {"VERILATOR_ROOT": str(root)}):
      v = _make_verilator(tmp_path, dpi_so=[])
      build_dir = v._write_cmake(obj_dir, generated_sources)
      assert (build_dir / "CMakeLists.txt").exists()
      content = (build_dir / "CMakeLists.txt").read_text()
      assert "VTestTop" in content
      assert generated_sources[0].as_posix() in content
      assert "verilated.cpp" in content
      assert "verilated_threads.cpp" in content
      assert "driver.cpp" in content

  def test_trace_sources_in_debug(self, tmp_path):
    obj_dir = tmp_path / "obj_dir"
    obj_dir.mkdir()
    generated_sources = [obj_dir / "VTestTop.cpp"]
    root = tmp_path / "verilator"
    (root / "include").mkdir(parents=True)
    (root / "include" / "verilated.h").touch()
    with mock.patch.dict(os.environ, {"VERILATOR_ROOT": str(root)}):
      v = _make_verilator(tmp_path, debug=True, dpi_so=[])
      build_dir = v._write_cmake(obj_dir, generated_sources)
      content = (build_dir / "CMakeLists.txt").read_text()
      assert "verilated_fst_c.cpp" in content
      assert "TRACE" in content

  def test_enables_pch_when_generated_header_exists(self, tmp_path):
    obj_dir = tmp_path / "obj_dir"
    obj_dir.mkdir()
    generated_sources = [obj_dir / "VTestTop.cpp"]
    pch_header = obj_dir / "VTestTop__pch.h"
    root = tmp_path / "verilator"
    (root / "include").mkdir(parents=True)
    (root / "include" / "verilated.h").touch()
    with mock.patch.dict(os.environ, {"VERILATOR_ROOT": str(root)}):
      v = _make_verilator(tmp_path, dpi_so=[])
      build_dir = v._write_cmake(obj_dir, generated_sources, pch_header)
      content = (build_dir / "CMakeLists.txt").read_text()
      assert "target_precompile_headers(VTestTop PRIVATE" in content
      assert "VTestTop__pch.h" in content
      assert "SKIP_PRECOMPILE_HEADERS ON" in content
      assert "verilated.cpp" in content
      assert "driver.cpp" in content


class TestRunCommand:

  def test_exe_path_cmake(self, tmp_path):
    v = _make_verilator(tmp_path, top="MyTop")
    if not v._use_cmake:
      pytest.skip("cmake+ninja not available")
    exe_name = "VMyTop.exe" if os.name == "nt" else "VMyTop"
    with mock.patch.object(Path, "cwd", return_value=tmp_path):
      cmd = v.run_command(gui=False)
      assert cmd == [str(tmp_path / "obj_dir" / "cmake_build" / exe_name)]
