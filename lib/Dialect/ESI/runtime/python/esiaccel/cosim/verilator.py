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
  C++ from RTL, then builds the simulation executable with CMake + Ninja.
  Falls back to ``make`` when cmake/ninja are not available."""

  DefaultDriver = CosimCollateralDir / "driver.cpp"

  def __init__(
      self,
      sources: SourceFiles,
      run_dir: Path,
      debug: bool,
      save_waveform: bool = False,
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
        save_waveform=save_waveform,
        run_stdout_callback=run_stdout_callback,
        run_stderr_callback=run_stderr_callback,
        compile_stdout_callback=compile_stdout_callback,
        compile_stderr_callback=compile_stderr_callback,
        make_default_logs=make_default_logs,
        macro_definitions=macro_definitions,
    )
    self.verilator_bin = "verilator_bin"
    if "VERILATOR_PATH" in os.environ:
      vpath = os.environ["VERILATOR_PATH"]
      # Backwards compatibility: if the env var points to the Perl wrapper,
      # redirect to verilator_bin.
      basename = Path(vpath).stem
      if basename == "verilator":
        self.verilator_bin = str(Path(vpath).parent / "verilator_bin")
      else:
        self.verilator_bin = vpath

  def _find_verilator_root(self) -> Path:
    """Locate VERILATOR_ROOT for runtime includes and sources.

    Checks the ``VERILATOR_ROOT`` environment variable first, then attempts
    to derive the root from the location of ``verilator_bin``.  Supports both
    source-tree layouts (``$ROOT/include/verilated.h``) and system package
    layouts (``$PREFIX/share/verilator/include/verilated.h``)."""
    if "VERILATOR_ROOT" in os.environ:
      root = Path(os.environ["VERILATOR_ROOT"])
      if root.is_dir():
        return root
    # verilator_bin is typically in $PREFIX/bin/
    verilator_bin_path = shutil.which(self.verilator_bin)
    if verilator_bin_path:
      prefix = Path(verilator_bin_path).resolve().parent.parent
      # Source-tree layout: $VERILATOR_ROOT/bin/verilator_bin
      if (prefix / "include" / "verilated.h").exists():
        return prefix
      # System package layout: $PREFIX/share/verilator/include/verilated.h
      pkg_root = prefix / "share" / "verilator"
      if (pkg_root / "include" / "verilated.h").exists():
        return pkg_root
    raise RuntimeError(
        "Cannot find VERILATOR_ROOT. Set the VERILATOR_ROOT environment "
        "variable or ensure verilator_bin is in PATH.")

  @property
  def _use_cmake(self) -> bool:
    """True when both cmake and ninja are available on PATH."""
    return shutil.which("cmake") is not None and \
        shutil.which("ninja") is not None

  def compile_commands(self) -> List[List[str]]:
    """Return the commands for the full compile flow.

    When cmake and ninja are available the returned list contains three
    commands run sequentially:
      1. ``verilator_bin`` – generates C++ from RTL.
      2. ``cmake`` – configures the C++ build (Ninja generator).
      3. ``ninja`` – builds the simulation executable.

    Otherwise falls back to two commands:
      1. ``verilator_bin --exe`` – generates C++ and a Makefile.
      2. ``make`` – builds via the generated Makefile.
    """
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

    if self._use_cmake:
      cmd += [str(p) for p in self.sources.rtl_sources]
      build_dir = str(Path.cwd() / "obj_dir" / "cmake_build")
      cmake_cmd = ["cmake", "-G", "Ninja", "-S", build_dir, "-B", build_dir]
      ninja_cmd = ["ninja", "-C", build_dir]
      return [cmd, cmake_cmd, ninja_cmd]

    # -- make fallback --
    # Let verilator generate a Makefile with --exe so it includes the
    # driver, CFLAGS, and LDFLAGS directly.
    cmd += ["--exe", str(Verilator.DefaultDriver)]
    cflags = ["-DTOP_MODULE=" + self.sources.top]
    if self.debug:
      cflags.append("-DTRACE")
    cmd += ["-CFLAGS", " ".join(cflags)]
    if self.sources.dpi_so:
      cmd += ["-LDFLAGS", " ".join(["-l" + so for so in self.sources.dpi_so])]
    cmd += [str(p) for p in self.sources.rtl_sources]
    top = self.sources.top
    make_cmd = ["make", "-C", "obj_dir", "-f", f"V{top}.mk", "-j"]
    return [cmd, make_cmd]

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
    if self.sources.dpi_so:
      runtime_sources.append(include_dir / "verilated_dpi.cpp")
    if self.debug:
      runtime_sources.append(include_dir / "verilated_fst_c.cpp")
    # Include constrained-randomization runtime when available (Verilator 5.x+).
    random_cpp = include_dir / "verilated_random.cpp"
    if random_cpp.exists():
      runtime_sources.append(random_cpp)

    rt_src = "\n  ".join(str(s) for s in runtime_sources)
    driver = str(Verilator.DefaultDriver)
    inc = str(include_dir)
    vltstd = str(include_dir / "vltstd")

    defs = [f"TOP_MODULE={self.sources.top}"]
    if self.debug:
      defs.append("TRACE")
    defs_str = "\n  ".join(defs)

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
  {defs_str}
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
    """Set VERILATOR_ROOT, write the CMakeLists.txt (if using cmake), then
    delegate to the base class which runs all commands from
    :meth:`compile_commands`."""
    verilator_root = self._find_verilator_root()
    os.environ["VERILATOR_ROOT"] = str(verilator_root)
    if self._use_cmake:
      self._write_cmake(Path.cwd() / "obj_dir")
    return super().compile()

  @property
  def waveform_extension(self) -> str:
    """Verilator's C++ driver uses ``VerilatedFstC`` — FST format."""
    return ".fst"

  def run_command(self, gui: bool):
    if gui:
      raise RuntimeError("Verilator does not support GUI mode.")
    exe_name = "V" + self.sources.top
    if self._use_cmake:
      exe = Path.cwd() / "obj_dir" / "cmake_build" / exe_name
    else:
      exe = Path.cwd() / "obj_dir" / exe_name
    return [str(exe)]
