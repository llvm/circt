#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
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
  _CMakeSignatureFilename = ".esi-cosim-cmake-config.json"
  _CMakeSignatureEnv = (
      "CMAKE_PREFIX_PATH",
      "CMAKE_TOOLCHAIN_FILE",
      "CXX",
      "CXXFLAGS",
      "LDFLAGS",
      "PATH",
  )
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
    # Set by _write_cmake when the generated CMakeLists.txt actually changed.
    self._cmake_dirty = True

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

  @staticmethod
  def _raise_stack_limit() -> None:
    """Lift the stack soft limit to the hard limit for the verilator process.

    Verilator recurses over the design AST and segfaults on large designs with
    the usual 8MB stack. Its ``verilator`` wrapper script normally runs
    ``ulimit -s unlimited`` first; we invoke ``verilator_bin`` directly and so
    have to do it ourselves. Subprocesses inherit the raised limit.
    """
    if os.name == "nt":
      return
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    if soft == hard:
      return
    try:
      resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except (ValueError, OSError):
      pass

  @staticmethod
  def _toolchain_args() -> List[str]:
    """Prefer clang and lld when they are available.

    Verilated code compiles about twice as fast with clang as with gcc, and
    the model is a throwaway simulation binary, so the toolchain only affects
    build time. Setting ``CXX`` or ``LDFLAGS`` opts back out.
    """
    if os.name == "nt":
      return []
    args = []
    if not os.environ.get("CXX"):
      clangxx = shutil.which("clang++")
      if clangxx is not None:
        args.append(f"-DCMAKE_CXX_COMPILER={clangxx}")
    if not os.environ.get("LDFLAGS") and shutil.which("ld.lld") is not None:
      args.append("-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld")
    return args

  @staticmethod
  def _cmake_signature(cmake_cmd: List[str]) -> str:
    """Serialize inputs that can affect CMake configuration."""
    signature = {
        "command": cmake_cmd,
        "environment": {
            name: os.environ.get(name) for name in Verilator._CMakeSignatureEnv
        },
    }
    return json.dumps(signature, indent=2, sort_keys=True) + "\n"

  def compile_commands(self) -> List[Simulator.CompileStep]:
    """Return the compile steps for the full compile flow.

    When cmake and ninja are available the returned list contains four
    sequential steps:
      1. ``verilator_bin`` – generates C++ from RTL.
      2. Python callback – generates the CMakeLists.txt from the depfile.
      3. Python callback – configures the C++ build when inputs changed.
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
    self._raise_stack_limit()

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
        # Every generated .cpp re-parses the model headers, so file count sets
        # the floor for the C++ build; 5000 balances that against parallelism.
        "--output-split",
        "5000",
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
      build_dir = Path.cwd() / "obj_dir" / "cmake_build"
      # ``CMAKE_BUILD_TYPE=Release`` is important on Windows: the prebuilt
      # ``EsiCosimDpiServer.dll`` ships with the Release MSVC runtime, and
      # mixing it with a Debug-runtime executable causes silent failures
      # (e.g. transport/control connections come up but requests stall).
      cmake_cmd = [
          "cmake", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-S",
          str(build_dir), "-B",
          str(build_dir)
      ]
      cmake_cmd += self._toolchain_args()
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
      ninja_cmd = ["ninja", "-C", str(build_dir)]
      cmake_signature = self._cmake_signature(cmake_cmd)
      signature_file = build_dir / self._CMakeSignatureFilename

      def configure(cmake_cmd=cmake_cmd,
                    cmake_signature=cmake_signature,
                    signature_file=signature_file) -> int:
        # Ninja regenerates an existing build graph when CMakeLists.txt changes.
        # Run CMake explicitly only for a new tree or changed configure inputs.
        signature_matches = signature_file.exists() and \
            signature_file.read_text() == cmake_signature
        if (build_dir / "build.ninja").exists() and signature_matches:
          return 0
        result = self._run_compile_command(cmake_cmd)
        if result == 0:
          signature_file.write_text(cmake_signature)
        return result

      return [
          verilator_cmd, self._write_cmake_from_depfile, configure, ninja_cmd
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

  @staticmethod
  def _is_slow(source: Path) -> bool:
    """Verilator suffixes cold-path TUs, including Syms/ConstPool, with
    ``__Slow``."""
    return source.stem.endswith("__Slow")

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

    slow_sources = [s for s in generated_sources if self._is_slow(s)]
    fast_sources = [s for s in generated_sources if not self._is_slow(s)]

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

      def shorten(sources: List[Path], prefix: str) -> List[Path]:
        shortened = []
        for index, source in enumerate(sources):
          destination = short_source_dir / f"{prefix}{index}.cpp"
          shutil.copy2(source, destination)
          shortened.append(destination)
        return shortened

      fast_sources = shorten(fast_sources, "vfast_")
      slow_sources = shorten(slow_sources, "vslow_")

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

    rt_src = "\n  ".join(s.as_posix() for s in runtime_sources)
    driver = Path(Verilator.DefaultDriver).as_posix()
    rt_and_driver = "\n  ".join([s.as_posix() for s in runtime_sources] +
                                [driver])
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

    # Separate object libraries so each optimization group gets its own
    # precompiled header; one PCH cannot serve two different -O levels.
    groups = []
    obj_refs = []
    for name, group_sources in (("vl_fast", fast_sources), ("vl_slow",
                                                            slow_sources)):
      if not group_sources:
        continue
      listing = "\n  ".join(s.as_posix() for s in group_sources)
      opts = ("\ntarget_compile_options(vl_slow PRIVATE ${VL_OPT_SLOW})"
              if name == "vl_slow" else "")
      pch = ("" if pch_header is None else
             f"\ntarget_precompile_headers({name} PRIVATE "
             f"{pch_header.as_posix()})")
      groups.append(f"""
add_library({name} OBJECT
  {listing}
)
target_link_libraries({name} PRIVATE vl_common){opts}{pch}""")
      obj_refs.append(f"$<TARGET_OBJECTS:{name}>")
    groups_str = "\n".join(groups)
    obj_str = "\n  ".join(obj_refs)

    # Verilator's FST writer (debug builds) pulls in both zlib and lz4.
    if self.debug:
      zlib_find = ("find_package(ZLIB REQUIRED)\n"
                   "find_library(LZ4_LIBRARY NAMES lz4 REQUIRED)")
      zlib_link = "ZLIB::ZLIB\n  ${LZ4_LIBRARY}"
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
  set(VL_OPT_SLOW /Od)
  set(VL_OPT_GLOBAL /O1)
else()
  set(VL_OPT_SLOW -O0)
  set(VL_OPT_GLOBAL -Os)
endif()

find_package(Threads REQUIRED)
{zlib_find}
add_library(vl_common INTERFACE)

target_include_directories(vl_common INTERFACE
  {inc}
  {vltstd}
  ${{CMAKE_CURRENT_SOURCE_DIR}}/..
)

target_compile_definitions(vl_common INTERFACE
  {defs_str}
)
{groups_str}

add_executable({exe_name}
  {obj_str}
  {rt_src}
  {driver}
)

set_source_files_properties(
  {rt_and_driver}
  PROPERTIES COMPILE_OPTIONS "${{VL_OPT_GLOBAL}}"
)

target_link_libraries({exe_name} PRIVATE
  vl_common
  Threads::Threads
  {zlib_link}
  {dpi_link}
)
"""
    build_dir = obj_dir / "cmake_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_file = build_dir / "CMakeLists.txt"
    existing = cmake_file.read_text() if cmake_file.exists() else None
    self._cmake_dirty = existing != content
    if self._cmake_dirty:
      cmake_file.write_text(content)
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
