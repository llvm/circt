"""End-to-end tests for `esiaccel.codegen`.

The bulk of the verification is delegated to `codegen_harness.cpp` in this
directory: the Python side generates a `types.h` from a representative
manifest, compiles the harness against it, and runs the result. The harness
exercises every accessor category (standard / odd / sub-byte integer widths,
bool, view-class, nested struct, array, union, window) and checks both the
user-visible round-trip and the underlying wire bytes. Reviewers can read the
harness directly to see what the codegen is contracted to do.

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

requires_cmake = pytest.mark.skipif(shutil.which("cmake") is None,
                                    reason="cmake not available")

# Small set of pre-built ESI scalar types shared by the manifest builders.
_uint1 = types.UIntType("ui1", 1)
_uint2 = types.UIntType("ui2", 2)
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

# View-class fields backed by `esi::IntView` / `esi::UIntView` /
# `esi::BitVector` (any width supported; only wider-than-64 cases route
# through the view classes -- narrower Bits/Int/UInt stay on the native
# int accessors). Tested below with 96- and 128-bit widths.
_bits128 = types.BitsType("bits128", 128)
_uint96 = types.UIntType("ui96", 96)
_uint128 = types.UIntType("ui128", 128)
_sint96 = types.SIntType("si96", 96)
_sint128 = types.SIntType("si128", 128)

# ---------------------------------------------------------------------------
# Harness Manifest Builder
# ---------------------------------------------------------------------------


def _build_harness_manifest():
  """Build the type table the `codegen_harness.cpp` test program references.

  The aliases below give every emitted struct a stable, hand-written C++
  name (e.g. `StdU`) so the harness can use plain identifiers rather than
  spelling the auto-generated mangled names.
  """
  # Standard 8/16/32/64-bit widths.
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

  # Byte-aligned but non-standard width (e.g. ui24).
  odd_u_inner = types.StructType("@OddU::inner", [("u24", _uint24)])
  odd_u = types.TypeAlias("@OddU", "OddU", odd_u_inner)
  odd_s_inner = types.StructType("@OddS::inner", [("s24", _sint24)])
  odd_s = types.TypeAlias("@OddS", "OddS", odd_s_inner)

  # Sub-byte alignment.
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

  # Arrays whose element storage size differs from the on-wire element
  # width, so the whole-array accessor must unpack each element from its
  # own wire bit offset instead of flat-copying. `std::array<uint8_t, 8>`
  # (ui3), `std::array<bool, 8>` (ui1), `std::array<int8_t, 4>` (si5), and
  # `std::array<uint32_t, 2>` (ui24) are all wider in memory than the
  # bit-packed wire layout they represent.
  u3_arr_inner = types.StructType(
      "@U3Arr::inner",
      [("vals", types.ArrayType("!hw.array<8xui3>", _uint3, 8))])
  u3_arr = types.TypeAlias("@U3Arr", "U3Arr", u3_arr_inner)

  bits1_arr_inner = types.StructType(
      "@Bits1Arr::inner",
      [("flags", types.ArrayType("!hw.array<8xui1>", _uint1, 8))])
  bits1_arr = types.TypeAlias("@Bits1Arr", "Bits1Arr", bits1_arr_inner)

  s5_arr_inner = types.StructType(
      "@S5Arr::inner",
      [("vals", types.ArrayType("!hw.array<4xsi5>", _sint5, 4))])
  s5_arr = types.TypeAlias("@S5Arr", "S5Arr", s5_arr_inner)

  u24_arr_inner = types.StructType(
      "@U24Arr::inner",
      [("vals", types.ArrayType("!hw.array<2xui24>", _uint24, 2))])
  u24_arr = types.TypeAlias("@U24Arr", "U24Arr", u24_arr_inner)

  # Array of sub-byte STRUCT elements. Each `{ui3 hi, ui2 lo}` cell is 5
  # wire bits, so successive cells pack at a 5-bit stride on the wire but a
  # padded 1-byte stride in the C++ `std::array<SbCell, 4>`. Exercises the
  # aggregate arm of the packed-array accessor, which copies each element's
  # bits into its own `_bytes` buffer.
  sb_cell_inner = types.StructType("@SbCell::inner", [("hi", _uint3),
                                                      ("lo", _uint2)])
  sb_cell = types.TypeAlias("@SbCell", "SbCell", sb_cell_inner)
  sb_cell_arr_inner = types.StructType(
      "@SbCellArr::inner",
      [("cells", types.ArrayType("!hw.array<4xSbCell>", sb_cell, 4))])
  sb_cell_arr = types.TypeAlias("@SbCellArr", "SbCellArr", sb_cell_arr_inner)

  # Nested array of sub-byte ints: `2 x 4 x ui3`. The element type is itself
  # the non-byte-packable array `4 x ui3` (12 wire bits but 4 bytes in the
  # C++ `std::array<uint8_t, 4>`). The old flat-copy path handled this case
  # incorrectly -- it emitted a corrupt whole-array copy and no indexed
  # accessor at all -- because `is_packed_array` only matched scalar/struct
  # elements, not array elements. The recursive (un)packer places each `ui3`
  # leaf at its true wire offset `i * 12 + j * 3`.
  nested3_inner = types.StructType(
      "@Nested3::inner",
      [("rows",
        types.ArrayType("!hw.array<2x!hw.array<4xui3>>",
                        types.ArrayType("!hw.array<4xui3>", _uint3, 4), 2))])
  nested3 = types.TypeAlias("@Nested3", "Nested3", nested3_inner)

  # Nested array of sub-byte STRUCT elements: `2 x 3 x SbCell` (each cell 5
  # wire bits). Array-of-array-of-aggregate: the recursive packer copies each
  # cell's bits at its true wire offset `i * 15 + j * 5`.
  nested_cell_inner = types.StructType("@NestedCell::inner", [
      ("grid",
       types.ArrayType("!hw.array<2x!hw.array<3xSbCell>>",
                       types.ArrayType("!hw.array<3xSbCell>", sb_cell, 3), 2))
  ])
  nested_cell = types.TypeAlias("@NestedCell", "NestedCell", nested_cell_inner)

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

  # View-class fields backed by `esi::MutableBitVector` /
  # `esi::Int` / `esi::UInt` from `esi/Values.h`. Covers BitsType at
  # both narrow and wide widths, plus signed/unsigned integers above
  # the 64-bit native ceiling. Layout fields cover byte-aligned and
  # bit-misaligned wide-int offsets so both the aligned and the
  # bit-shifted view accessors get exercised.
  wide_u_inner = types.StructType(
      "@WideU::inner",
      [("u96", _uint96), ("u128", _uint128)],
  )
  wide_u = types.TypeAlias("@WideU", "WideU", wide_u_inner)
  wide_s_inner = types.StructType(
      "@WideS::inner",
      [("s96", _sint96), ("s128", _sint128)],
  )
  wide_s = types.TypeAlias("@WideS", "WideS", wide_s_inner)
  # Bits-typed fields > 64 bits route through the value-class path
  # (`esi::BitVector` view); narrower Bits stay on the native int paths
  # and don't need a dedicated harness here.
  bits_inner = types.StructType(
      "@BitsField::inner",
      [("wide", _bits128)],
  )
  bits_field = types.TypeAlias("@BitsField", "BitsField", bits_inner)
  # Place the wide UInt field *before* a 3-bit tag in the manifest.
  # Field order is reversed on the wire (`cpp_type.reverse=True`), so the
  # LAST manifest field lands at wire bit 0. Listing `payload` first and
  # `tag` last puts `tag` at bits 0..2 (byte-aligned) and the 128-bit
  # `payload` at bits 3..130 — exercising the bit-shifted view
  # accessor for a wide field at a non-byte-aligned offset rather
  # than only the byte-aligned case.
  wide_mis_inner = types.StructType(
      "@WideMisaligned::inner",
      [("payload", _uint128), ("tag", _uint3)],
  )
  wide_mis = types.TypeAlias("@WideMisaligned", "WideMisaligned",
                             wide_mis_inner)

  # Array of view-class elements. The planner used to skip parent types
  # that reached this construct; now the emitter handles them by emitting
  # per-element indexed accessors that build fresh views into `_bytes`.
  # Use ui128 (a view-class type) at multiple element widths to exercise
  # both byte-aligned and bit-misaligned per-element offsets.
  arr3_u128_type = types.ArrayType("!hw.array<3xui128>", _uint128, 3)
  arr_views_inner = types.StructType(
      "@ArrViews::inner",
      [("items", arr3_u128_type)],
  )
  arr_views = types.TypeAlias("@ArrViews", "ArrViews", arr_views_inner)
  # Same array placed after a 3-bit tag so the array starts at bit 3
  # and successive elements land at non-byte-aligned per-element
  # offsets (3, 131, 259, ...).
  arr_views_mis_inner = types.StructType(
      "@ArrViewsMis::inner",
      [("items", arr3_u128_type), ("tag", _uint3)],
  )
  arr_views_mis = types.TypeAlias("@ArrViewsMis", "ArrViewsMis",
                                  arr_views_mis_inner)

  return [
      std_u, std_s, odd_u, odd_s, sub_u, sub_s, bool_field, outer, misaligned,
      arr4, u3_arr, bits1_arr, s5_arr, u24_arr, sb_cell, sb_cell_arr, nested3,
      nested_cell, union_two, list_window, wide_u, wide_s, bits_field, wide_mis,
      arr_views, arr_views_mis
  ]


# ---------------------------------------------------------------------------
# Harness build + run
# ---------------------------------------------------------------------------


@requires_cmake
def test_codegen_round_trip(tmp_path):
  """Compile `codegen_harness.cpp` against a freshly-generated `types.h`
  and run it. The harness asserts every wire-format and accessor invariant
  end-to-end; this Python test just drives the cmake build and reports
  failures.
  """
  from esiaccel.utils import get_dll_dir
  esi_dll_path = get_dll_dir()
  if sys.platform == "win32":
    runtime_lib = esi_dll_path / "ESICppRuntime.lib"
    runtime_dll = esi_dll_path / "ESICppRuntime.dll"
  else:
    runtime_lib = esi_dll_path / "libESICppRuntime.so"
    runtime_dll = None

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
      "-DCMAKE_BUILD_TYPE=Release",
      f"-DCODEGEN_HARNESS_GENERATED_DIR={generated_dir}",
      f"-DESI_RUNTIME_LIB={runtime_lib}",
  ]
  if sys.platform == "win32":
    configure_cmd.append(f"-DESI_RUNTIME_DLL={runtime_dll}")
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
      str(build_dir), "--target", "codegen_harness", "--config", "Release"
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


def test_port_find_code_golden():
  """Golden check of the per-port-kind `connect()` snippets for ALL kinds.

  The round-trip harness only exercises the types; the live port-kind paths
  are covered by the (heavyweight) cosim integration suite. This test pins
  the generated resolution code for every kind -- including the ones the
  sample manifest doesn't contain (Callback / FromHost / MMIO / Metric /
  Bundle) -- so a typo in the `_PortKind` table can't slip through.
  """
  from esiaccel.codegen import (_PORT_KINDS, _BUNDLE_KIND, _render_scalar_find,
                                _render_indexed_find)

  kinds = {cls.__name__: kind for cls, kind in _PORT_KINDS}
  kinds["BundlePort"] = _BUNDLE_KIND
  ae = 'esi::AppID("p")'
  aei = 'esi::AppID("p", idx)'

  def scalar(name: str) -> str:
    k = kinds[name]
    return _render_scalar_find(k, f"p{k.param_suffix}", ae)

  def indexed(name: str) -> str:
    return _render_indexed_find(kinds[name], "p_backing", aei)

  # --- scalar resolution snippets ---
  assert scalar("FunctionPort") == (
      "auto *p_port =\n"
      "    esi::findPortAsOrThrow<esi::services::FuncService::Function>(\n"
      '        rawModule, esi::AppID("p"));')
  assert scalar("CallbackPort") == (
      "auto *p_port =\n"
      "    esi::findPortAsOrThrow<esi::services::CallService::Callback>(\n"
      '        rawModule, esi::AppID("p"));')
  assert scalar("ToHostPort") == (
      "auto &p_chan =\n"
      "    esi::findPortAsOrThrow<esi::services::ChannelService::ToHost>(\n"
      '        rawModule, esi::AppID("p"))->getRawRead("data");')
  assert scalar("FromHostPort") == (
      "auto &p_chan =\n"
      "    esi::findPortAsOrThrow<esi::services::ChannelService::FromHost>(\n"
      '        rawModule, esi::AppID("p"))->getRawWrite("data");')
  assert scalar("MMIORegion") == (
      "auto &p_svc =\n"
      "    *esi::findPortAsOrThrow<esi::services::MMIO::MMIORegion>(\n"
      '        rawModule, esi::AppID("p"));')
  assert scalar("MetricPort") == (
      "auto &p_svc =\n"
      "    *esi::findPortAsOrThrow<esi::services::TelemetryService::Metric>(\n"
      '        rawModule, esi::AppID("p"));')
  assert scalar("BundlePort") == (
      'auto &p_port = esi::findPortOrThrow(rawModule, esi::AppID("p"));')

  # --- indexed (IndexedPorts<T>) try_emplace bodies ---
  assert indexed("FunctionPort") == (
      "  p_backing.try_emplace(\n"
      "      static_cast<int>(idx),\n"
      "      esi::findPortAsOrThrow<esi::services::FuncService::Function>(\n"
      '          rawModule, esi::AppID("p", idx)));')
  assert indexed("ToHostPort") == (
      "  auto *svc =\n"
      "      esi::findPortAsOrThrow<esi::services::ChannelService::ToHost>(\n"
      '          rawModule, esi::AppID("p", idx));\n'
      "  p_backing.try_emplace(\n"
      "      static_cast<int>(idx),\n"
      '      svc->getRawRead("data"));')
  assert indexed("FromHostPort") == (
      "  auto *svc =\n"
      "      esi::findPortAsOrThrow<esi::services::ChannelService::FromHost>(\n"
      '          rawModule, esi::AppID("p", idx));\n'
      "  p_backing.try_emplace(\n"
      "      static_cast<int>(idx),\n"
      '      svc->getRawWrite("data"));')
  assert indexed("MMIORegion") == (
      "  p_backing.try_emplace(\n"
      "      static_cast<int>(idx),\n"
      "      esi::findPortAsOrThrow<esi::services::MMIO::MMIORegion>(\n"
      '          rawModule, esi::AppID("p", idx)));')
  assert indexed("BundlePort") == (
      "  p_backing.try_emplace(\n"
      "      static_cast<int>(idx),\n"
      "      &esi::findPortOrThrow(rawModule, esi::AppID(\"p\", idx)));")

  # --- per-kind field invariants ---
  assert [
      k.connectable
      for k in (kinds["FunctionPort"], kinds["CallbackPort"],
                kinds["ToHostPort"], kinds["FromHostPort"], kinds["MMIORegion"],
                kinds["MetricPort"], kinds["BundlePort"])
  ] == [True, False, True, True, False, True, False]
  assert kinds["FunctionPort"].param_suffix == "_port"
  assert kinds["ToHostPort"].param_suffix == "_chan"
  assert kinds["MMIORegion"].param_suffix == "_svc"
  assert (kinds["FunctionPort"].member_template ==
          "esi::TypedFunction<{p}Args, {p}Result>")
  assert kinds["MMIORegion"].member_template is None
  assert (
      kinds["MMIORegion"].indexed_elem == "esi::services::MMIO::MMIORegion *")
  assert kinds["FunctionPort"].alias_kind == "func"
  assert kinds["ToHostPort"].alias_kind == "chan"
  assert kinds["MMIORegion"].alias_kind is None
