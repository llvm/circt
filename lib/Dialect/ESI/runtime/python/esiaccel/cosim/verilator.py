#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
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
  VerilatorBinNotFound = (
      "Cannot find verilator_bin. Set VERILATOR_PATH to an absolute path "
      "or ensure verilator_bin is in PATH.")
  VerilatorRootNotFound = (
      "Cannot find VERILATOR_ROOT. Set the VERILATOR_ROOT environment "
      "variable or ensure verilator_bin is in PATH.")
  VerilatorPathInvalid = (
      "VERILATOR_PATH does not point to a valid verilator_bin executable.")
  VerilatorRootInvalid = (
      "VERILATOR_ROOT does not point to a Verilator root containing "
      "include/verilated.h.")

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

  @property
  def verilator_bin(self) -> Path:
    vpath = Verilator._find_verilator_bin()
    if vpath is None:
      raise RuntimeError(Verilator.VerilatorBinNotFound)
    return vpath

  @staticmethod
  def _find_verilator_bin() -> Optional[Path]:
    """Locate the ``verilator_bin`` executable.

    When ``VERILATOR_PATH`` is set it must point to a valid executable;
    otherwise a ``RuntimeError`` is raised. Without it, ``verilator_bin`` is
    looked up on ``PATH``. Returns ``None`` when nothing is found."""

    def check_path(path: Path | str | None) -> Optional[Path]:
      if isinstance(path, str):
        path = Path(path)
      if path is not None and path.exists() and path.is_file():
        return path.resolve()
      return None

    if "VERILATOR_PATH" in os.environ:
      vpath = Path(os.environ["VERILATOR_PATH"])
      if vpath.stem == "verilator":
        vpath = vpath.parent / "verilator_bin"
      checked = check_path(vpath)
      if checked is None:
        raise RuntimeError(Verilator.VerilatorPathInvalid)
      return checked
    return check_path(shutil.which("verilator_bin"))

  @staticmethod
  def _find_verilator_root() -> Optional[Path]:
    """Locate the Verilator root containing ``include/verilated.h``.

    When ``VERILATOR_ROOT`` is set it must contain ``include/verilated.h``;
    otherwise a ``RuntimeError`` is raised. Without it, the packaged root
    (``$PREFIX/share/verilator``) is derived from the ``verilator_bin``
    location. Returns ``None`` when nothing is found."""
    if "VERILATOR_ROOT" in os.environ:
      root = Path(os.environ["VERILATOR_ROOT"])
      if (root / "include" / "verilated.h").exists():
        return root
      raise RuntimeError(Verilator.VerilatorRootInvalid)

    verilator_bin = Verilator._find_verilator_bin()
    if verilator_bin is None:
      return None

    # Packaged installations put Verilator's support files under
    # $PREFIX/share/verilator, where $PREFIX is the bin directory's parent.
    pkg_root = verilator_bin.parent.parent / "share" / "verilator"
    if (pkg_root / "include" / "verilated.h").exists():
      return pkg_root

    return None

  @property
  def _use_cmake(self) -> bool:
    """True when both cmake and ninja are available on PATH."""
    return shutil.which("cmake") is not None and \
        shutil.which("ninja") is not None

  def compile_commands(self) -> List[Simulator.CompileStep]:
    """Return the compile steps for the full compile flow.

    When cmake and ninja are available the returned list contains four
    sequential steps:
      1. ``verilator_bin`` – generates C++ from RTL.
      2. Python callback – generates the CMakeLists.txt from the depfile.
      3. ``cmake`` – configures the C++ build (Ninja generator).
      4. ``ninja`` – builds the simulation executable.

    Otherwise falls back to two commands:
      1. ``verilator_bin --exe`` – generates C++ and a Makefile.
      2. ``make`` – builds via the generated Makefile.
    """
    verilator_bin = self._find_verilator_bin()
    if verilator_bin is None:
      raise RuntimeError(Verilator.VerilatorBinNotFound)
    verilator_root = self._find_verilator_root()
    if verilator_root is None:
      raise RuntimeError(Verilator.VerilatorRootNotFound)
    os.environ["VERILATOR_ROOT"] = str(verilator_root)

    verilator_cmd: List[str] = [
        str(verilator_bin),
        "--cc",
    ]

    if self.macro_definitions:
      verilator_cmd += [
          f"+define+{k}={v}" if v is not None else f"+define+{k}"
          for k, v in self.macro_definitions.items()
      ]

    verilator_cmd += [
        "--top-module",
        self.sources.top,
        "-DSIMULATION",
        "-Wno-TIMESCALEMOD",
        "-Wno-fatal",
        "-sv",
        "--verilate-jobs",
        "0",
        "--output-split",
        "2500",
    ]
    if self.debug:
      verilator_cmd += [
          "--assert",
          "--trace-fst",
          "--trace-structs",
          "--trace-underscore",
      ]

    if self._use_cmake:
      verilator_cmd += [str(p) for p in self.sources.rtl_sources]
      build_dir = str(Path.cwd() / "obj_dir" / "cmake_build")
      # ``CMAKE_BUILD_TYPE=Release`` is important on Windows: the prebuilt
      # ``EsiCosimDpiServer.dll`` ships with the Release MSVC runtime, and
      # mixing it with a Debug-runtime executable causes silent failures
      # (e.g. transport/control connections come up but requests stall).
      cmake_cmd = [
          "cmake", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-S", build_dir,
          "-B", build_dir
      ]
      # If vcpkg is available, use its toolchain file so that
      # ``find_package(ZLIB)`` (and other transitive deps) can pick up vcpkg
      # installations. This is the standard story on Windows.
      vcpkg_root = os.environ.get("VCPKG_ROOT") or os.environ.get(
          "VCPKG_INSTALLATION_ROOT")
      if vcpkg_root:
        toolchain = Path(
            vcpkg_root) / "scripts" / "buildsystems" / "vcpkg.cmake"
        if toolchain.exists():
          cmake_cmd.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain}")
      ninja_cmd = ["ninja", "-C", build_dir]
      return [
          verilator_cmd, self._write_cmake_from_depfile, cmake_cmd, ninja_cmd
      ]

    # -- make fallback --
    # Let verilator generate a Makefile with --exe so it includes the
    # driver, CFLAGS, and LDFLAGS directly.
    verilator_cmd += ["--exe", str(Verilator.DefaultDriver)]
    cflags = ["-DTOP_MODULE=" + self.sources.top]
    if self.debug:
      cflags.append("-DTRACE")
    verilator_cmd += ["-CFLAGS", " ".join(cflags)]
    if self.sources.dpi_so:
      dpi_so_paths = self.sources.dpi_so_paths()
      verilator_cmd += [
          "-LDFLAGS",
          " ".join(["-l" + so for so in self.sources.dpi_so]) + " " +
          " ".join(["-L" + so.parent.as_posix() for so in dpi_so_paths]),
      ]
    verilator_cmd += [str(p) for p in self.sources.rtl_sources]
    top = self.sources.top
    make_cmd = ["make", "-C", "obj_dir", "-f", f"V{top}.mk", "-j"]
    return [verilator_cmd, make_cmd]

  def _depfile_path(self, obj_dir: Path) -> Path:
    return obj_dir / f"V{self.sources.top}__ver.d"

  def _generated_targets(self, depfile: Path) -> List[Path]:
    depfile_contents = depfile.read_text().replace("\\\n", " ")
    separator = re.search(r":\s", depfile_contents)
    if separator is None:
      raise RuntimeError(f"Malformed Verilator depfile: {depfile}")
    return [(Path.cwd() / path).resolve()
            for path in depfile_contents[:separator.start()].split()]

  def _write_cmake_from_depfile(self) -> int:
    obj_dir = Path.cwd() / "obj_dir"
    depfile = self._depfile_path(obj_dir)
    generated_targets = self._generated_targets(depfile)
    generated_sources = [
        path for path in generated_targets if path.suffix == ".cpp"
    ]
    pch_header = next(
        (path for path in generated_targets if path.name.endswith("__pch.h")),
        None)
    self._write_cmake(obj_dir, generated_sources, pch_header)
    return 0

  def _generated_cpp_sources(self, depfile: Path) -> List[Path]:
    generated_sources = [
        path for path in self._generated_targets(depfile)
        if path.suffix == ".cpp"
    ]
    if not generated_sources:
      raise RuntimeError(
          f"No generated C++ sources found in depfile: {depfile}")
    return generated_sources

  def _write_cmake(self,
                   obj_dir: Path,
                   generated_sources: List[Path],
                   pch_header: Optional[Path] = None) -> Path:
    """Write a CMakeLists.txt for building the verilated simulation.

    Returns the path to the CMake build directory."""

    verilator_root = self._find_verilator_root()
    if verilator_root is None:
      raise RuntimeError(Verilator.VerilatorRootNotFound)
    include_dir = verilator_root / "include"
    exe_name = "V" + self.sources.top

    if os.name == "nt" and all(source.exists() for source in generated_sources):
      # Verilator can emit deeply descriptive source filenames. CMake uses the
      # source basename in MSVC's /Fo object path, which can overflow Windows'
      # practical object path limits even after CMake hashes directories.
      # Short local copies keep the build graph stable without changing the
      # generated code or its includes.
      short_source_dir = obj_dir / "cmake_src"
      if short_source_dir.exists():
        shutil.rmtree(short_source_dir)
      short_source_dir.mkdir(parents=True)
      shortened_sources = []
      for index, source in enumerate(generated_sources):
        shortened_source = short_source_dir / f"vsrc_{index}.cpp"
        shutil.copy2(source, shortened_source)
        shortened_sources.append(shortened_source)
      generated_sources = shortened_sources

    runtime_sources = [
        include_dir / "verilated.cpp",
        include_dir / "verilated_threads.cpp",
    ]
    # Include Verilator's DPI helpers when DPI shared objects are enabled.
    if self.sources.dpi_so:
      runtime_sources.append(include_dir / "verilated_dpi.cpp")
    if self.debug:
      runtime_sources.append(include_dir / "verilated_fst_c.cpp")
    # Include constrained-randomization runtime when available (Verilator 5.x+).
    random_cpp = include_dir / "verilated_random.cpp"
    if random_cpp.exists():
      runtime_sources.append(random_cpp)

    generated_src = "\n  ".join(
        source.as_posix() for source in generated_sources)
    rt_src = "\n  ".join(s.as_posix() for s in runtime_sources)
    driver = Path(Verilator.DefaultDriver).as_posix()
    inc = include_dir.as_posix()
    vltstd = (include_dir / "vltstd").as_posix()

    defs = [f"TOP_MODULE={self.sources.top}"]
    if self.debug:
      defs.append("TRACE")
    defs_str = "\n  ".join(defs)

    # Link DPI shared objects by full path. On Windows, link against the
    # ``.lib`` import library; the matching ``.dll`` is found at runtime via
    # ``PATH`` (see ``Simulator.get_env``).
    dpi_link = ""
    if self.sources.dpi_so:
      dpi_paths = self.sources.dpi_link_paths()
      dpi_link = "\n  ".join(p.as_posix() for p in dpi_paths)

    pch_setup = ""
    if pch_header is not None:
      runtime_and_driver = "\n  ".join(
          [source.as_posix() for source in runtime_sources] + [driver])
      pch_setup = f"""
target_precompile_headers({exe_name} PRIVATE
  {pch_header.as_posix()}
)

set_source_files_properties(
  {runtime_and_driver}
  PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)
"""

    # zlib is only needed when FST tracing (debug builds) is enabled.
    if self.debug:
      zlib_find = "find_package(ZLIB REQUIRED)"
      zlib_link = "ZLIB::ZLIB"
    else:
      zlib_find = ""
      zlib_link = ""

    content = f"""\
cmake_minimum_required(VERSION 3.20)
project({exe_name} CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/EHsc /bigobj)
endif()

find_package(Threads REQUIRED)
{zlib_find}
add_executable({exe_name}
  {generated_src}
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
{pch_setup}

target_link_libraries({exe_name} PRIVATE
  Threads::Threads
  {zlib_link}
  {dpi_link}
)
"""
    build_dir = obj_dir / "cmake_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeLists.txt").write_text(content)
    return build_dir

  @property
  def waveform_extension(self) -> str:
    """Verilator's C++ driver uses ``VerilatedFstC`` — FST format."""
    return ".fst"

  def run_command(self, gui: bool):
    if gui:
      raise RuntimeError("Verilator does not support GUI mode.")
    exe_name = "V" + self.sources.top
    if os.name == "nt":
      exe_name += ".exe"
    if self._use_cmake:
      exe = Path.cwd() / "obj_dir" / "cmake_build" / exe_name
    else:
      exe = Path.cwd() / "obj_dir" / exe_name
    return [str(exe)]
