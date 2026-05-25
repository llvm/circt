"""End-to-end tests for `esiaccel.codegen`.

The bulk of the verification is delegated to `codegen_harness.cpp` in this
directory: the Python side generates a `types.h` from a representative
manifest, compiles the harness against it, and runs the result. The harness
exercises every accessor path (Path A / B / C, bool, nested struct, array,
union, window) and checks both the user-visible round-trip and the
underlying wire bytes. Reviewers can read the harness directly to see what
the codegen is contracted to do.

A small number of Python-level tests cover behaviours the harness can't
exercise — ordering, name collisions, type aliases, and the "skip with a
comment" path for unsupported / window-containing / fully-collapsed
structs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import pytest

import esiaccel.types as types
from esiaccel.codegen import CppTypePlanner, CppTypeEmitter

_HARNESS_DIR = Path(__file__).parent

# Small set of pre-built ESI scalar types shared by the manifest builders.
_uint1 = types.UIntType("ui1", 1)
_uint3 = types.UIntType("ui3", 3)
_uint7 = types.UIntType("ui7", 7)
_uint8 = types.UIntType("ui8", 8)
_uint12 = types.UIntType("ui12", 12)
_uint16 = types.UIntType("ui16", 16)
_uint24 = types.UIntType("ui24", 24)
_uint32 = types.UIntType("ui32", 32)
_uint64 = types.UIntType("ui64", 64)
_sint5 = types.SIntType("si5", 5)
_sint7 = types.SIntType("si7", 7)
_sint8 = types.SIntType("si8", 8)
_sint16 = types.SIntType("si16", 16)
_sint24 = types.SIntType("si24", 24)
_sint32 = types.SIntType("si32", 32)
_sint64 = types.SIntType("si64", 64)

# ---------------------------------------------------------------------------
# Harness Manifest Builder
# ---------------------------------------------------------------------------


def _build_harness_manifest():
  """Build the type table the `codegen_harness.cpp` test program references.

  The aliases below give every emitted struct a stable, hand-written C++
  name (e.g. `StdU`) so the harness can use plain identifiers rather than
  spelling the auto-generated mangled names.
  """
  # Path A: standard widths.
  std_u_inner = types.StructType(
      "@StdU::inner",
      [("u8", _uint8), ("u16", _uint16), ("u32", _uint32), ("u64", _uint64)],
  )
  std_u = types.TypeAlias("@StdU", "StdU", std_u_inner)

  std_s_inner = types.StructType(
      "@StdS::inner",
      [("s8", _sint8), ("s16", _sint16), ("s32", _sint32), ("s64", _sint64)],
  )
  std_s = types.TypeAlias("@StdS", "StdS", std_s_inner)

  # Path B: byte-aligned but non-standard width.
  odd_u_inner = types.StructType("@OddU::inner", [("u24", _uint24)])
  odd_u = types.TypeAlias("@OddU", "OddU", odd_u_inner)
  odd_s_inner = types.StructType("@OddS::inner", [("s24", _sint24)])
  odd_s = types.TypeAlias("@OddS", "OddS", odd_s_inner)

  # Path C: sub-byte alignment.
  sub_u_inner = types.StructType("@SubU::inner", [("u3", _uint3),
                                                  ("u12", _uint12)])
  sub_u = types.TypeAlias("@SubU", "SubU", sub_u_inner)
  sub_s_inner = types.StructType("@SubS::inner", [("s5", _sint5),
                                                  ("s7", _sint7)])
  sub_s = types.TypeAlias("@SubS", "SubS", sub_s_inner)

  # 1-bit bool field.
  bool_inner = types.StructType("@BoolField::inner", [("flag", _uint1),
                                                      ("pad", _uint7)])
  bool_field = types.TypeAlias("@BoolField", "BoolField", bool_inner)

  # Nested struct field.
  inner_inner = types.StructType("@Inner::inner", [("x", _uint8),
                                                   ("y", _uint8)])
  inner = types.TypeAlias("@Inner", "Inner", inner_inner)
  outer_inner = types.StructType("@Outer::inner", [("label", _uint8),
                                                   ("inner", inner)])
  outer = types.TypeAlias("@Outer", "Outer", outer_inner)

  # Nested struct embedded at a sub-byte bit offset. With the default
  # `cpp_type.reverse=True`, the LAST manifest field ends up at wire bit
  # 0, so listing `inner` first and `tag` (ui3) second puts `tag` in
  # bits 0..2 and the 16-bit inner in bits 3..18 — i.e. the inner
  # aggregate starts at bit 3, not byte-aligned. Exercises the
  # `copyBitsIn`/`copyBitsOut` paths.
  mis_inner_inner = types.StructType("@MisInner::inner", [("x", _uint8),
                                                          ("y", _uint8)])
  mis_inner = types.TypeAlias("@MisInner", "MisInner", mis_inner_inner)
  misaligned_inner = types.StructType("@Misaligned::inner",
                                      [("inner", mis_inner), ("tag", _uint3)])
  misaligned = types.TypeAlias("@Misaligned", "Misaligned", misaligned_inner)

  # Array-of-integers field with the indexed accessor pair.
  arr4_type = types.ArrayType("!hw.array<4xui8>", _uint8, 4)
  arr4_inner = types.StructType("@Arr4::inner", [("r", arr4_type)])
  arr4 = types.TypeAlias("@Arr4", "Arr4", arr4_inner)

  # Union with one narrow and one wide variant.
  union_inner = types.UnionType("@UnionTwo::inner", [("small", _uint8),
                                                     ("big", _uint16)])
  union_two = types.TypeAlias("@UnionTwo", "UnionTwo", union_inner)

  # Window helper with one static `tag` header field and a list of ui32.
  list_id = "!esi.list<ui32>"
  list_type = types.ListType(list_id, _uint32)
  win_arg_inner = types.StructType(
      "@ListWindow::arg",
      [("tag", _uint16), ("items", list_type)],
  )
  win_header_inner = types.StructType(
      "@ListWindow::header",
      [("tag", _uint16), ("items_count", _uint16)],
  )
  win_data_inner = types.StructType(
      "@ListWindow::data",
      [("items", types.ArrayType("!hw.array<1xui32>", _uint32, 1))],
  )
  win_lowered = types.UnionType(
      "@ListWindow::lowered",
      [("header", win_header_inner), ("data", win_data_inner)],
  )
  window_id = ('!esi.window<"ListWindow", @ListWindow::arg, '
               '[<"header", [<"tag">, <"items" countWidth 16>]>, '
               '<"data", [<"items", 1>]>]>')
  list_window_inner = types.WindowType(
      window_id,
      "ListWindow",
      win_arg_inner,
      win_lowered,
      [
          types.WindowType.Frame(
              "header",
              [
                  types.WindowType.Field("tag", 0, 0),
                  types.WindowType.Field("items", 0, 16),
              ],
          ),
          types.WindowType.Frame(
              "data",
              [types.WindowType.Field("items", 1, 0)],
          ),
      ],
  )
  # Hand-name the window so the harness can spell `ListWindow` directly.
  list_window = types.TypeAlias("@ListWindow", "ListWindow", list_window_inner)

  return [
      std_u, std_s, odd_u, odd_s, sub_u, sub_s, bool_field, outer, misaligned,
      arr4, union_two, list_window
  ]


# ---------------------------------------------------------------------------
# Harness build + run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _harness_env():
  """Resolve `cmake` and the runtime shared library, or skip.

  The runtime library always ships next to the `esiaccel` Python package
  at `<esiaccel>/lib/...`, which is the same binary the rest of the Python
  test suite already loads. Using that as the anchor avoids needing to
  plumb in a separate `ESI_RUNTIME_ROOT`, parse `esiaccelConfig.cmake`,
  or walk the build tree.

  On Windows, MSVC needs the `.lib` import library to link against and
  the `.dll` to load at runtime, so we look up both separately and feed
  them through to CMake as `ESI_RUNTIME_LIB` (link target) and
  `ESI_RUNTIME_DLL` (runtime artifact to stage next to the executable).
  On Unix the single shared object covers both jobs.
  """
  if shutil.which("cmake") is None:
    pytest.skip("cmake not available")

  try:
    import esiaccel
  except ImportError:
    pytest.skip("esiaccel not importable")
  esiaccel_lib_dir = Path(esiaccel.__file__).parent / "lib"

  runtime_lib = None
  runtime_dll = None
  if sys.platform == "win32":
    # Prefer the MSVC import library for linking; fall back to the DLL
    # itself (which works with MinGW) if no `.lib` was shipped.
    lib = esiaccel_lib_dir / "ESICppRuntime.lib"
    dll = esiaccel_lib_dir / "ESICppRuntime.dll"
    if lib.exists():
      runtime_lib = lib
    if dll.exists():
      runtime_dll = dll
      if runtime_lib is None:
        runtime_lib = dll
  else:
    for name in ("libESICppRuntime.so", "libESICppRuntime.dylib"):
      candidate = esiaccel_lib_dir / name
      if candidate.exists():
        runtime_lib = candidate
        break

  if runtime_lib is None:
    pytest.skip(f"ESICppRuntime link artifact not found under "
                f"{esiaccel_lib_dir}")

  return {"runtime_lib": runtime_lib, "runtime_dll": runtime_dll}


def test_codegen_round_trip(_harness_env, tmp_path):
  """Compile `codegen_harness.cpp` against a freshly-generated `types.h`
  and run it. The harness asserts every wire-format and accessor invariant
  end-to-end; this Python test just drives the cmake build and reports
  failures.
  """
  # Generate the header into `<generated>/codegen_harness/types.h` so the
  # harness's `#include "codegen_harness/types.h"` resolves under
  # `<generated>`, which we feed to CMake as the include root.
  generated_dir = tmp_path / "generated"
  (generated_dir / "codegen_harness").mkdir(parents=True)
  planner = CppTypePlanner(_build_harness_manifest())
  emitter = CppTypeEmitter(planner)
  emitter.write_header(generated_dir / "codegen_harness", "esi_system")

  build_dir = tmp_path / "build"
  configure_cmd = [
      "cmake",
      "-S",
      str(_HARNESS_DIR),
      "-B",
      str(build_dir),
      f"-DCODEGEN_HARNESS_GENERATED_DIR={generated_dir}",
      f"-DESI_RUNTIME_LIB={_harness_env['runtime_lib']}",
  ]
  if _harness_env["runtime_dll"] is not None:
    configure_cmd.append(f"-DESI_RUNTIME_DLL={_harness_env['runtime_dll']}")
  configure_proc = subprocess.run(configure_cmd, capture_output=True, text=True)
  if configure_proc.returncode != 0:
    pytest.fail(
        "cmake configure failed for the codegen harness "
        f"(rc={configure_proc.returncode}):\n"
        f"--- cmd ---\n{' '.join(configure_cmd)}\n"
        f"--- stdout ---\n{configure_proc.stdout}\n"
        f"--- stderr ---\n{configure_proc.stderr}",
        pytrace=False,
    )

  build_cmd = [
      "cmake", "--build",
      str(build_dir), "--target", "codegen_harness"
  ]
  build_proc = subprocess.run(build_cmd, capture_output=True, text=True)
  if build_proc.returncode != 0:
    # Print the generated header alongside the compile failure so a
    # reviewer can diff the codegen output against what the harness
    # expects to see.
    pytest.fail(
        "codegen_harness.cpp failed to compile against the generated "
        f"types.h (rc={build_proc.returncode}):\n"
        f"--- cmd ---\n{' '.join(build_cmd)}\n"
        f"--- stdout ---\n{build_proc.stdout}\n"
        f"--- stderr ---\n{build_proc.stderr}\n"
        "--- generated types.h ---\n" +
        (generated_dir / "codegen_harness" / "types.h").read_text(),
        pytrace=False,
    )

  # CMake places the executable under the build directory; on multi-config
  # generators (e.g. Visual Studio) the output lives under
  # `build_dir/<Config>/`. Walk the build dir to find whatever was built.
  binary_name = "codegen_harness" + sysconfig.get_config_var("EXE")
  candidates = list(build_dir.rglob(binary_name))
  if not candidates:
    pytest.fail(
        f"codegen_harness binary not found under {build_dir}; "
        "the cmake build reported success but produced no executable.",
        pytrace=False,
    )
  binary = candidates[0]

  run_proc = subprocess.run([str(binary)], capture_output=True, text=True)
  if run_proc.returncode != 0 or run_proc.stdout.strip() != "OK":
    pytest.fail(
        f"codegen_harness reported failure (rc={run_proc.returncode}):\n"
        f"--- stdout ---\n{run_proc.stdout}\n"
        f"--- stderr ---\n{run_proc.stderr}",
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# Lightweight Python-level checks for behaviours the harness can't drive.
# ---------------------------------------------------------------------------


def _emit(type_table, system_name: str = "test_ns") -> str:
  """Run the planner + emitter against `type_table` and return `types.h`."""
  planner = CppTypePlanner(type_table)
  emitter = CppTypeEmitter(planner)
  with tempfile.TemporaryDirectory() as tmpdir:
    emitter.write_header(Path(tmpdir), system_name)
    return (Path(tmpdir) / "types.h").read_text()


def test_unbounded_field_skips_struct_with_comment():
  """A struct whose payload type has no bounded width (e.g. `!esi.any`)
  cannot be expressed in the raw-bytes layout; the codegen drops the
  whole struct and leaves an `Unsupported type` comment behind so callers
  see why the symbol they expected is missing."""
  any_t = types.AnyType("!esi.any")
  s = types.StructType("@any_struct", [("tag", _uint8), ("data", any_t)])

  hdr = _emit([s])
  assert "// Unsupported type '<@any_struct>'" in hdr
