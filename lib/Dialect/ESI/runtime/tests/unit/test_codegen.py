"""Tests for UnionType support in codegen (CppTypePlanner + CppTypeEmitter)."""

import tempfile
from pathlib import Path

import esiaccel.types as types
from esiaccel.codegen import CppTypePlanner, CppTypeEmitter


def _generate_header(type_table, system_name="test_ns"):
  """Helper: run the planner + emitter on a type table and return the header."""
  planner = CppTypePlanner(type_table)
  emitter = CppTypeEmitter(planner)
  with tempfile.TemporaryDirectory() as tmpdir:
    emitter.write_header(Path(tmpdir), system_name)
    return (Path(tmpdir) / "types.h").read_text()


def test_union_basic():
  """A simple union with two scalar fields produces a C++ union."""
  uint8 = types.UIntType("ui8", 8)
  uint16 = types.UIntType("ui16", 16)
  union_t = types.UnionType("!hw.union<a: ui8, b: ui16>", [("a", uint8),
                                                           ("b", uint16)])

  hdr = _generate_header([union_t])
  assert "union " in hdr
  assert '_ESI_ID = "!hw.union<a: ui8, b: ui16>"' in hdr
  # Field "a" (8 bits) is narrower than the 16-bit union → wrapper struct
  # with _pad before the data field.
  assert "struct _union_a_ui8_b_ui16_a" in hdr
  assert "uint8_t _pad[1]" in hdr
  assert "uint8_t a;" in hdr
  # In the wrapper struct, _pad appears before the field.
  pad_pos = hdr.index("uint8_t _pad[1]")
  field_a_pos = hdr.index("uint8_t a;")
  assert pad_pos < field_a_pos
  # Field "b" (16 bits) is full width → no wrapper, appears directly in union.
  assert "uint16_t b;" in hdr
  # Wrapper struct for "a" is emitted before the union keyword.
  wrapper_pos = hdr.index("struct _union_a_ui8_b_ui16_a")
  union_pos = hdr.index("union ")
  assert wrapper_pos < union_pos
  # Inside the union, "a" comes before "b" (declaration order preserved).
  union_body = hdr[union_pos:]
  assert union_body.index("a;") < union_body.index("b;")
  # The union member for "a" uses the wrapper struct type, not a raw int.
  assert "_union_a_ui8_b_ui16_a a;" in union_body


def test_union_with_struct_field():
  """A union containing a struct field emits the struct before the union."""
  uint8 = types.UIntType("ui8", 8)
  uint16 = types.UIntType("ui16", 16)
  inner = types.StructType("!hw.struct<x: ui8, y: ui8>", [("x", uint8),
                                                          ("y", uint8)])
  union_t = types.UnionType("!hw.union<header: ui16, data: !s>",
                            [("header", uint16), ("data", inner)])

  hdr = _generate_header([union_t])
  # The inner struct must appear before the wrapper structs and union.
  struct_pos = hdr.index("struct _struct")
  union_pos = hdr.index("union ")
  assert struct_pos < union_pos
  assert "data;" in hdr
  # "header" is 16 bits in a 16-bit union (both fields are 16 bits),
  # so no padding wrapper is needed for either field.
  assert "_pad" not in hdr


def test_union_ordering_among_structs():
  """Unions are properly ordered with respect to struct dependencies."""
  uint8 = types.UIntType("ui8", 8)
  s1 = types.StructType("!hw.struct<p: ui8>", [("p", uint8)])
  s2 = types.StructType("!hw.struct<q: ui8>", [("q", uint8)])
  union_t = types.UnionType("!hw.union<a: !s1, b: !s2>", [("a", s1), ("b", s2)])

  hdr = _generate_header([union_t])
  union_pos = hdr.index("union ")
  # Both structs should appear before the union.
  for keyword in ["struct _struct_p_ui8", "struct _struct_q_ui8"]:
    assert keyword in hdr
    assert hdr.index(keyword) < union_pos


def test_union_in_struct():
  """A struct with a union field emits the union before the struct."""
  uint8 = types.UIntType("ui8", 8)
  uint16 = types.UIntType("ui16", 16)
  union_t = types.UnionType("!hw.union<a: ui8, b: ui16>", [("a", uint8),
                                                           ("b", uint16)])
  outer = types.StructType("!hw.struct<tag: ui8, data: !u>",
                           [("tag", uint8), ("data", union_t)])

  hdr = _generate_header([outer])
  assert "union " in hdr
  # Wrapper struct for padded field "a" and the union both precede the
  # outer struct that references the union.
  wrapper_pos = hdr.index("struct _union")
  union_pos = hdr.index("union ")
  struct_pos = hdr.index("struct _struct")
  assert wrapper_pos < union_pos < struct_pos


def test_union_planner_naming():
  """The planner auto-generates deterministic names for unions."""
  uint8 = types.UIntType("ui8", 8)
  union_t = types.UnionType("!hw.union<x: ui8>", [("x", uint8)])

  planner = CppTypePlanner([union_t])
  assert union_t in planner.type_id_map
  name = planner.type_id_map[union_t]
  assert name.startswith("_union")


def test_union_alias():
  """A TypeAlias wrapping a union emits the union then a using alias."""
  uint8 = types.UIntType("ui8", 8)
  uint16 = types.UIntType("ui16", 16)
  union_t = types.UnionType("!hw.union<a: ui8, b: ui16>", [("a", uint8),
                                                           ("b", uint16)])
  alias = types.TypeAlias("!hw.typealias<MyUnion>", "MyUnion", union_t)

  hdr = _generate_header([alias])
  assert "union " in hdr
  assert "using MyUnion" in hdr


def test_union_same_width_integrals():
  """Integrals of the same width don't need padding wrappers."""
  uint16 = types.UIntType("ui16", 16)
  sint16 = types.SIntType("si16", 16)
  union_t = types.UnionType("!hw.union<a: ui16, b: si16>", [("a", uint16),
                                                            ("b", sint16)])

  hdr = _generate_header([union_t])
  assert "union " in hdr
  assert "_pad" not in hdr
  # Both fields appear directly in the union as raw types.
  union_body = hdr[hdr.index("union "):]
  assert "uint16_t a;" in union_body
  assert "int16_t b;" in union_body


def test_union_field_order_preserved():
  """Union fields are emitted in declaration order, not reversed."""
  uint8 = types.UIntType("ui8", 8)
  sint16 = types.SIntType("si16", 16)
  uint32 = types.UIntType("ui32", 32)
  union_t = types.UnionType("!hw.union<z: ui8, m: si16, a: ui32>",
                            [("z", uint8), ("m", sint16), ("a", uint32)])

  hdr = _generate_header([union_t])
  # Fields z (8 bit) and m (16 bit) need padding wrappers; a (32 bit) doesn't.
  assert "_pad[3]" in hdr  # z wrapper: 4 - 1 = 3 bytes padding
  assert "_pad[2]" in hdr  # m wrapper: 4 - 2 = 2 bytes padding
  # In each wrapper struct, _pad appears before the data field.
  z_pad = hdr.index("_pad[3]")
  z_field = hdr.index("uint8_t z;")
  assert z_pad < z_field
  m_pad = hdr.index("_pad[2]")
  m_field = hdr.index("int16_t m;")
  assert m_pad < m_field
  # Inside the union body, field order is preserved (z, m, a).
  union_body = hdr[hdr.index("union "):]
  z_pos = union_body.index(" z;")
  m_pos = union_body.index(" m;")
  a_pos = union_body.index(" a;")
  assert z_pos < m_pos < a_pos
  # Field a (full width) has no wrapper.
  assert "uint32_t a;" in union_body
  # Wrapped fields use wrapper struct types as union members.
  assert "_union_z_ui8_m_si16_a_ui32_z z;" in union_body
  assert "_union_z_ui8_m_si16_a_ui32_m m;" in union_body


def test_windowed_list_bulk_message_wrapper():
  """Bulk-encoded list windows emit a SegmentedMessageData helper."""
  uint16 = types.UIntType("ui16", 16)
  uint32 = types.UIntType("ui32", 32)
  coord_struct_id = "!hw.struct<x: ui32, y: ui32>"
  coord_alias_id = (
    f"!hw.typealias<@esi_runtime_codegen::@Coord, {coord_struct_id}>")
  coord_list_id = f"!esi.list<{coord_alias_id}>"
  arg_struct_id = (
    f"!hw.struct<x_translation: ui32, y_translation: ui32, coords: "
    f"{coord_list_id}>")
  header_struct_id = (
    "!hw.struct<x_translation: ui32, y_translation: ui32, coords_count: "
    "ui16>")
  data_struct_id = f"!hw.struct<coords: !hw.array<1x{coord_alias_id}>>"
  lowered_id = f"!hw.union<header: {header_struct_id}, data: {data_struct_id}>"
  serial_args_id = (
    f'!esi.window<"serial_coord_args", {arg_struct_id}, '
    '[<"header", [<"x_translation">, <"y_translation">, '
    '<"coords" countWidth 16>]>, <"data", [<"coords", 1>]>]>')

  coord_inner = types.StructType(coord_struct_id, [("x", uint32), ("y", uint32)])
  coord = types.TypeAlias(coord_alias_id, "Coord", coord_inner)
  coord_list = types.ListType(coord_list_id, coord)
  arg_struct = types.StructType(arg_struct_id, [("x_translation", uint32),
                        ("y_translation", uint32),
                        ("coords", coord_list)])
  header_struct = types.StructType(header_struct_id,
                   [("x_translation", uint32),
                  ("y_translation", uint32),
                  ("coords_count", uint16)])
  data_struct = types.StructType(
    data_struct_id,
    [("coords", types.ArrayType(f"!hw.array<1x{coord_alias_id}>", coord, 1))],
  )
  lowered = types.UnionType(lowered_id, [("header", header_struct),
                     ("data", data_struct)])
  serial_args = types.WindowType(serial_args_id, "serial_coord_args", arg_struct,
                 lowered, [
                   types.WindowType.Frame(
                     "header",
                     [
                       types.WindowType.Field(
                         "x_translation", 0, 0),
                       types.WindowType.Field(
                         "y_translation", 0, 0),
                       types.WindowType.Field(
                         "coords", 0, 16),
                     ],
                   ),
                   types.WindowType.Frame(
                     "data",
                     [types.WindowType.Field("coords", 1, 0)],
                   ),
                 ])

  hdr = _generate_header([coord, serial_args])
  assert "Unsupported type" not in hdr
  assert "struct serial_coord_args : public esi::SegmentedMessageData" in hdr
  assert "using value_type = Coord;" in hdr
  assert "using count_type = uint16_t;" in hdr
  assert "uint16_t coords_count;" in hdr
  assert "uint8_t _pad[2];" in hdr
  assert "Coord coords;" in hdr
  assert "std::vector<data_frame> data_frames;" in hdr
  assert "esi::Segment segment(size_t idx) const override" in hdr
  assert "footer.coords_count = 0;" in hdr
  assert "frame.coords = element;" in hdr
  assert '!esi.window<\\"serial_coord_args\\"' in hdr
  assert 'throw std::out_of_range("serial_coord_args: invalid segment index")' in hdr


def test_windowed_list_header_padding_matches_frame_width():
  """Headers pad out to the data frame width for count-only windows."""
  uint16 = types.UIntType("ui16", 16)
  uint32 = types.UIntType("ui32", 32)
  element_id = "!hw.struct<x: ui32, y: ui32>"
  list_id = f"!esi.list<{element_id}>"
  arg_struct_id = f"!hw.struct<coords: {list_id}>"
  header_struct_id = "!hw.struct<coords_count: ui16>"
  data_struct_id = f"!hw.struct<coords: !hw.array<1x{element_id}>>"
  lowered_id = f"!hw.union<header: {header_struct_id}, data: {data_struct_id}>"
  window_id = (
      f'!esi.window<"coords_only", {arg_struct_id}, '
      '[<"header", [<"coords" countWidth 16>]>, '
      '<"data", [<"coords", 1>]>]>')

  element = types.StructType(element_id, [("x", uint32), ("y", uint32)])
  coord_list = types.ListType(list_id, element)
  arg_struct = types.StructType(arg_struct_id, [("coords", coord_list)])
  header_struct = types.StructType(header_struct_id, [("coords_count", uint16)])
  data_struct = types.StructType(
      data_struct_id,
      [("coords", types.ArrayType(f"!hw.array<1x{element_id}>", element, 1))],
  )
  lowered = types.UnionType(lowered_id, [("header", header_struct),
                                         ("data", data_struct)])
  window = types.WindowType(window_id, "coords_only", arg_struct, lowered, [
      types.WindowType.Frame(
          "header", [types.WindowType.Field("coords", 0, 16)]),
      types.WindowType.Frame(
          "data", [types.WindowType.Field("coords", 1, 0)]),
  ])

  hdr = _generate_header([window])
  assert "struct coords_only : public esi::SegmentedMessageData" in hdr
  assert "struct header_frame {\n    uint8_t _pad[6];\n    uint16_t coords_count;\n  };" in hdr
  assert "header_frame footer{};" in hdr
