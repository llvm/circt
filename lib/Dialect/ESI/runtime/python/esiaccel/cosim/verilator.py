#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
from pathlib import Path
from typing import List, Optional, Callable, Dict

from .simulator import CosimCollateralDir, Simulator, SourceFiles


class Verilator(Simulator):
  """Run and compile funcs for Verilator.

  Calls ``verilator_bin`` directly (bypassing the Perl wrapper) to generate
  C++ from RTL, then builds the simulation executable with CMake + Ninja."""

  DefaultDriver = CosimCollateralDir / "driver.cpp"

  def __init__(
      self,
      sources: SourceFiles,
      run_dir: Path,
      debug: bool,
      run_stdout_callback: Optional[Callable[[str], None]] = None,
      run_stderr_callback: Optional[Callable[[str], None]] = None,
      compile_stdout_callback: Optional[Callable[[str], None]] = None,
      compile_stderr_callback: Optional[Callable[[str], None]] = None,
      make_default_logs: bool = True,
      macro_definitions: Optional[Dict[str, str]] = None,
  ):
    super().__init__(
        sources=sources,
        run_dir=run_dir,
        debug=debug,
        run_stdout_callback=run_stdout_callback,
        run_stderr_callback=run_stderr_callback,
        compile_stdout_callback=compile_stdout_callback,
        compile_stderr_callback=compile_stderr_callback,
        make_default_logs=make_default_logs,
        macro_definitions=macro_definitions,
    )
    self.verilator_bin = "verilator_bin"
    if "VERILATOR_PATH" in os.environ:
      self.verilator_bin = os.environ["VERILATOR_PATH"]

  def _find_verilator_root(self) -> Path:
    """Locate VERILATOR_ROOT for runtime includes and sources.

    Checks the ``VERILATOR_ROOT`` environment variable first, then attempts
    to derive the root from the location of ``verilator_bin``."""
    if "VERILATOR_ROOT" in os.environ:
      root = Path(os.environ["VERILATOR_ROOT"])
      if root.is_dir():
        return root
    # verilator_bin is typically in $VERILATOR_ROOT/bin/
    verilator_bin_path = shutil.which(self.verilator_bin)
    if verilator_bin_path:
      root = Path(verilator_bin_path).resolve().parent.parent
      if (root / "include" / "verilated.h").exists():
        return root
    raise RuntimeError(
        "Cannot find VERILATOR_ROOT. Set the VERILATOR_ROOT environment "
        "variable or ensure verilator_bin is in PATH.")

  def compile_commands(self) -> List[List[str]]:
    """Return the verilator_bin command for generating C++ from RTL.

    This only produces the Verilated C++ source files; the actual C++ build
    is handled separately by :meth:`compile` via CMake + Ninja."""
    cmd: List[str] = [
        self.verilator_bin,
        "--cc",
    ]

    if self.macro_definitions:
      cmd += [
          f"+define+{k}={v}" if v is not None else f"+define+{k}"
          for k, v in self.macro_definitions.items()
      ]

    cmd += [
        "--top-module",
        self.sources.top,
        "-DSIMULATION",
        "-Wno-TIMESCALEMOD",
        "-Wno-fatal",
        "-sv",
        "-j",
        "0",
        "--output-split",
        "--autoflush",
        "--assert",
    ]
    if self.debug:
      cmd += [
          "--trace-fst", "--trace-params", "--trace-structs",
          "--trace-underscore"
      ]
    cmd += [str(p) for p in self.sources.rtl_sources]
    return [cmd]

  def _write_cmake(self, obj_dir: Path) -> Path:
    """Write a CMakeLists.txt for building the verilated simulation.

    Returns the path to the CMake build directory."""
    verilator_root = self._find_verilator_root()
    include_dir = verilator_root / "include"
    exe_name = "V" + self.sources.top

    runtime_sources = [
        include_dir / "verilated.cpp",
        include_dir / "verilated_threads.cpp",
    ]
    if self.debug:
      runtime_sources.append(include_dir / "verilated_fst_c.cpp")

    rt_src = "\n  ".join(str(s) for s in runtime_sources)
    driver = str(Verilator.DefaultDriver)
    inc = str(include_dir)
    vltstd = str(include_dir / "vltstd")

    defs = f"TOP_MODULE={self.sources.top}"
    if self.debug:
      defs += "\n  TRACE"

    # Link DPI shared objects by full path.
    dpi_link = ""
    if self.sources.dpi_so:
      dpi_paths = self.sources.dpi_so_paths()
      dpi_link = "\n  ".join(str(p) for p in dpi_paths)

    content = f"""\
cmake_minimum_required(VERSION 3.20)
project({exe_name} CXX)

file(GLOB GENERATED_SOURCES "${{CMAKE_CURRENT_SOURCE_DIR}}/../*.cpp")

add_executable({exe_name}
  ${{GENERATED_SOURCES}}
  {rt_src}
  {driver}
)

target_include_directories({exe_name} PRIVATE
  {inc}
  {vltstd}
  ${{CMAKE_CURRENT_SOURCE_DIR}}/..
)

target_compile_definitions({exe_name} PRIVATE
  {defs}
)

find_package(Threads REQUIRED)
target_link_libraries({exe_name} PRIVATE
  Threads::Threads
  {dpi_link}
)
"""
    build_dir = obj_dir / "cmake_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeLists.txt").write_text(content)
    return build_dir

  def compile(self) -> int:
    """Compile RTL sources with verilator_bin, then build with CMake + Ninja."""
    self.run_dir.mkdir(parents=True, exist_ok=True)
    env = Simulator.get_env()

    # Ensure VERILATOR_ROOT is in the environment for verilator_bin.
    verilator_root = self._find_verilator_root()
    env["VERILATOR_ROOT"] = str(verilator_root)

    # Step 1: Run verilator_bin to generate C++ from Verilog/SystemVerilog.
    for cmd in self.compile_commands():
      ret = self._start_process_with_callbacks(
          cmd,
          env=env,
          cwd=None,
          stdout_cb=self._compile_stdout_cb,
          stderr_cb=self._compile_stderr_cb,
          wait=True)
      if isinstance(ret, int) and ret != 0:
        print("====== Verilator compilation failure")
        if self.UsesStderr:
          if self._compile_stderr_log is not None:
            self._compile_stderr_log.seek(0)
            print(self._compile_stderr_log.read())
        else:
          if self._compile_stdout_log is not None:
            self._compile_stdout_log.seek(0)
            print(self._compile_stdout_log.read())
        return ret

    # Step 2: Write a CMakeLists.txt and build the simulation executable.
    obj_dir = Path.cwd() / "obj_dir"
    build_dir = self._write_cmake(obj_dir)

    # Step 3: Configure with CMake (Ninja generator).
    cmake_cmd = ["cmake", "-G", "Ninja", "."]
    ret = self._start_process_with_callbacks(
        cmake_cmd,
        env=env,
        cwd=build_dir,
        stdout_cb=self._compile_stdout_cb,
        stderr_cb=self._compile_stderr_cb,
        wait=True)
    if isinstance(ret, int) and ret != 0:
      print("====== CMake configuration failure")
      if self._compile_stderr_log is not None:
        self._compile_stderr_log.seek(0)
        print(self._compile_stderr_log.read())
      return ret

    # Step 4: Build with Ninja.
    ninja_cmd = ["ninja"]
    ret = self._start_process_with_callbacks(
        ninja_cmd,
        env=env,
        cwd=build_dir,
        stdout_cb=self._compile_stdout_cb,
        stderr_cb=self._compile_stderr_cb,
        wait=True)
    if isinstance(ret, int) and ret != 0:
      print("====== Ninja build failure")
      if self._compile_stderr_log is not None:
        self._compile_stderr_log.seek(0)
        print(self._compile_stderr_log.read())
      return ret

    return 0

  def run_command(self, gui: bool):
    if gui:
      raise RuntimeError("Verilator does not support GUI mode.")
    exe = Path.cwd() / "obj_dir" / "cmake_build" / ("V" + self.sources.top)
    return [str(exe)]
