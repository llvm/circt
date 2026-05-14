"""Tests for UnionType support in codegen (CppTypePlanner + CppTypeEmitter)."""

import re
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


def _window_struct_name(hdr, window_name_suffix):
  """Find the generated SegmentedMessageData subclass name for a window.

  Auto-names compose `{into_name}_{window_name}`, so locate the helper by its
  trailing window-name suffix instead of hard-coding the full identifier.
  """
  match = re.search(
      rf"struct (\S*{re.escape(window_name_suffix)}) "
      r": public esi::SegmentedMessageData", hdr)
  assert match, f"Window helper for '*{window_name_suffix}' not found in:\n{hdr}"
  return match.group(1)


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
  serial_args_id = (f'!esi.window<"serial_coord_args", {arg_struct_id}, '
                    '[<"header", [<"x_translation">, <"y_translation">, '
                    '<"coords" countWidth 16>]>, <"data", [<"coords", 1>]>]>')

  coord_inner = types.StructType(coord_struct_id, [("x", uint32),
                                                   ("y", uint32)])
  coord = types.TypeAlias(coord_alias_id, "Coord", coord_inner)
  coord_list = types.ListType(coord_list_id, coord)
  arg_struct = types.StructType(arg_struct_id, [("x_translation", uint32),
                                                ("y_translation", uint32),
                                                ("coords", coord_list)])
  header_struct = types.StructType(header_struct_id, [("x_translation", uint32),
                                                      ("y_translation", uint32),
                                                      ("coords_count", uint16)])
  data_struct = types.StructType(
      data_struct_id,
      [("coords", types.ArrayType(f"!hw.array<1x{coord_alias_id}>", coord, 1))],
  )
  lowered = types.UnionType(lowered_id, [("header", header_struct),
                                         ("data", data_struct)])
  serial_args = types.WindowType(
      serial_args_id, "serial_coord_args", arg_struct, lowered, [
          types.WindowType.Frame(
              "header",
              [
                  types.WindowType.Field("x_translation", 0, 0),
                  types.WindowType.Field("y_translation", 0, 0),
                  types.WindowType.Field("coords", 0, 16),
              ],
          ),
          types.WindowType.Frame(
              "data",
              [types.WindowType.Field("coords", 1, 0)],
          ),
      ])

  hdr = _generate_header([coord, serial_args])
  assert "Unsupported type" not in hdr
  win_name = _window_struct_name(hdr, "_serial_coord_args")
  assert f"struct {win_name} : public esi::SegmentedMessageData" in hdr
  assert "using value_type = Coord;" in hdr
  assert "using count_type = uint16_t;" in hdr
  assert "count_type coords_count;" in hdr
  assert "uint8_t _pad[2];" in hdr
  assert "Coord coords;" in hdr
  assert hdr.index("struct data_frame {") < hdr.index(
      "private:\n#pragma pack(push, 1)\n  struct header_frame {")
  assert "std::vector<data_frame> data_frames;" in hdr
  assert "esi::Segment segment(size_t idx) const override" in hdr
  assert "footer.coords_count = 0;" in hdr
  assert "const std::vector<value_type> &coords" in hdr
  assert "void construct(uint32_t x_translation, uint32_t y_translation, std::vector<data_frame> frames)" in hdr
  assert "construct(x_translation, y_translation, std::move(frames));" in hdr
  assert "auto &frame = frames.emplace_back();" in hdr
  assert "for (const auto &element : coords) {" in hdr
  assert "frame.coords = element;" in hdr
  # Inner struct id is _ESI_ID; window id (with escaped quotes) is
  # _ESI_WINDOW_ID so the runtime can verify the wire format too.
  assert f'_ESI_ID = "{arg_struct_id}"' in hdr
  escaped_serial_args_id = serial_args_id.replace('"', '\\"')
  assert f'_ESI_WINDOW_ID = "{escaped_serial_args_id}"' in hdr
  assert f'throw std::out_of_range("{win_name}: invalid segment index")' in hdr
  # Accessor methods for header fields and data fields.
  assert "uint32_t x_translation() const { return header.x_translation; }" in hdr
  assert "uint32_t y_translation() const { return header.y_translation; }" in hdr
  assert "size_t coords_count() const { return data_frames.size(); }" in hdr
  # Byte-aligned data field: pointer-to-member projection (zero-copy view).
  assert "return std::views::transform(data_frames, &data_frame::coords);" in hdr
  assert "std::vector<value_type> coords_vector() const" in hdr
  assert "out.push_back(frame.coords);" in hdr


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
  window_id = (f'!esi.window<"coords_only", {arg_struct_id}, '
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
      types.WindowType.Frame("header",
                             [types.WindowType.Field("coords", 0, 16)]),
      types.WindowType.Frame("data", [types.WindowType.Field("coords", 1, 0)]),
  ])

  hdr = _generate_header([window])
  win_name = _window_struct_name(hdr, "_coords_only")
  assert f"struct {win_name} : public esi::SegmentedMessageData" in hdr
  assert "struct header_frame {\n    uint8_t _pad[6];\n    count_type coords_count;\n  };" in hdr
  assert "header_frame footer{};" in hdr
  assert "void construct(std::vector<data_frame> frames)" in hdr


def test_windowed_list_arrays_in_header_and_value_type():
  """Window helpers copy array header fields and array-valued elements."""
  uint8 = types.UIntType("ui8", 8)
  uint16 = types.UIntType("ui16", 16)
  header_array_id = "!hw.array<2xui16>"
  value_array_id = "!hw.array<4xui8>"
  list_id = f"!esi.list<{value_array_id}>"
  arg_struct_id = (
      f"!hw.struct<header_words: {header_array_id}, payloads: {list_id}>")
  header_struct_id = (
      f"!hw.struct<header_words: {header_array_id}, payloads_count: ui16>")
  data_struct_id = f"!hw.struct<payloads: !hw.array<1x{value_array_id}>>"
  lowered_id = f"!hw.union<header: {header_struct_id}, data: {data_struct_id}>"
  window_id = (f'!esi.window<"array_payloads", {arg_struct_id}, '
               '[<"header", [<"header_words">, <"payloads" countWidth 16>]>, '
               '<"data", [<"payloads", 1>]>]>')

  header_array = types.ArrayType(header_array_id, uint16, 2)
  value_array = types.ArrayType(value_array_id, uint8, 4)
  payload_list = types.ListType(list_id, value_array)
  arg_struct = types.StructType(arg_struct_id, [("header_words", header_array),
                                                ("payloads", payload_list)])
  header_struct = types.StructType(header_struct_id,
                                   [("header_words", header_array),
                                    ("payloads_count", uint16)])
  data_struct = types.StructType(
      data_struct_id,
      [("payloads",
        types.ArrayType(f"!hw.array<1x{value_array_id}>", value_array, 1))],
  )
  lowered = types.UnionType(lowered_id, [("header", header_struct),
                                         ("data", data_struct)])
  window = types.WindowType(window_id, "array_payloads", arg_struct, lowered, [
      types.WindowType.Frame(
          "header",
          [
              types.WindowType.Field("header_words", 0, 0),
              types.WindowType.Field("payloads", 0, 16),
          ],
      ),
      types.WindowType.Frame(
          "data",
          [types.WindowType.Field("payloads", 1, 0)],
      ),
  ])

  hdr = _generate_header([window])
  assert "#include <array>" in hdr
  win_name = _window_struct_name(hdr, "_array_payloads")
  assert f"struct {win_name} : public esi::SegmentedMessageData" in hdr
  # `value_type` is exposed as `std::array` so it is storable in `std::vector`.
  assert "using value_type = std::array<uint8_t, 4>;" in hdr
  assert "using count_type = uint16_t;" in hdr
  # All array-typed fields (header and data) are emitted as `std::array`.
  assert "std::array<uint16_t, 2> header_words;" in hdr
  assert "std::array<uint8_t, 4> payloads;" in hdr
  # Constructor params for arrays are simple const-refs to `std::array`.
  assert f"{win_name}(const std::array<uint16_t, 2> &header_words, const std::vector<value_type> &payloads)" in hdr
  assert "void construct(const std::array<uint16_t, 2> &header_words, std::vector<data_frame> frames)" in hdr
  # Plain `=` assignment for both header and data array fields.
  assert "header.header_words = header_words;" in hdr
  assert "frame.payloads = element;" in hdr
  assert f'throw std::out_of_range("{win_name}: invalid segment index")' in hdr
  # Array-typed header field: const-ref accessor with std::array return type.
  assert "const std::array<uint16_t, 2> &header_words() const { return header.header_words; }" in hdr
  # Array-typed data field: pointer-to-member projection (zero-copy view) and
  # std::array-valued vector helper. The list field uses the `value_type`
  # alias (which equals `std::array<uint8_t, 4>`).
  assert "return std::views::transform(data_frames, &data_frame::payloads);" in hdr
  assert "std::vector<value_type> payloads_vector() const" in hdr
  assert "out.push_back(frame.payloads);" in hdr


def test_windowed_list_struct_element_data_uses_pointer_to_member():
  """Struct-typed data fields use a pointer-to-member projection, not a lambda.

  Even if the struct contains a non-byte-aligned (bit-field) member, the data
  field itself is a struct type, which is byte-aligned and supports
  pointer-to-member projection.
  """
  sint3 = types.SIntType("si3", 3)
  uint16 = types.UIntType("ui16", 16)
  list_id = f"!esi.list<!hw.struct<v: si3>>"
  elem_id = "!hw.struct<v: si3>"
  elem_struct = types.StructType(elem_id, [("v", sint3)])
  elem_list = types.ListType(list_id, elem_struct)
  arg_struct_id = f"!hw.struct<items: {list_id}>"
  arg_struct = types.StructType(arg_struct_id, [("items", elem_list)])
  header_struct_id = "!hw.struct<items_count: ui16>"
  header_struct = types.StructType(header_struct_id, [("items_count", uint16)])
  data_struct_id = f"!hw.struct<items: !hw.array<1x{elem_id}>>"
  data_struct = types.StructType(
      data_struct_id,
      [("items", types.ArrayType(f"!hw.array<1x{elem_id}>", elem_struct, 1))],
  )
  lowered_id = (
      f"!hw.union<header: {header_struct_id}, data: {data_struct_id}>")
  lowered = types.UnionType(lowered_id, [("header", header_struct),
                                         ("data", data_struct)])
  window_id = (f'!esi.window<"bitfield_items", {arg_struct_id}, '
               '[<"header", [<"items" countWidth 16>]>, '
               '<"data", [<"items", 1>]>]>')
  window = types.WindowType(window_id, "bitfield_items", arg_struct, lowered, [
      types.WindowType.Frame("header",
                             [types.WindowType.Field("items", 0, 16)]),
      types.WindowType.Frame("data", [types.WindowType.Field("items", 1, 0)]),
  ])

  hdr = _generate_header([elem_struct, window])
  win_name = _window_struct_name(hdr, "_bitfield_items")
  assert f"struct {win_name} : public esi::SegmentedMessageData" in hdr
  # The data field "items" is a struct type (byte-aligned), so pointer-to-member
  # IS valid.  The generated accessor must use &data_frame::items, not a lambda.
  assert "return std::views::transform(data_frames, &data_frame::items);" in hdr
  assert "[](const data_frame &f) { return f.items; }" not in hdr
  assert "std::vector<value_type> items_vector() const" in hdr


def test_windowed_list_bitfield_scalar_data_uses_lambda():
  """A window data field that is itself a non-byte-aligned int uses a lambda projection."""
  uint3 = types.UIntType("ui3", 3)
  uint16 = types.UIntType("ui16", 16)
  # Build a window where the data field is directly a 3-bit uint.
  list_id = "!esi.list<ui3>"
  elem_list = types.ListType(list_id, uint3)
  arg_struct_id = f"!hw.struct<vals: {list_id}>"
  arg_struct = types.StructType(arg_struct_id, [("vals", elem_list)])
  header_struct_id = "!hw.struct<vals_count: ui16>"
  header_struct = types.StructType(header_struct_id, [("vals_count", uint16)])
  data_struct_id = "!hw.struct<vals: !hw.array<1xui3>>"
  data_struct = types.StructType(
      data_struct_id,
      [("vals", types.ArrayType("!hw.array<1xui3>", uint3, 1))],
  )
  lowered_id = (
      f"!hw.union<header: {header_struct_id}, data: {data_struct_id}>")
  lowered = types.UnionType(lowered_id, [("header", header_struct),
                                         ("data", data_struct)])
  window_id = (f'!esi.window<"bitval_window", {arg_struct_id}, '
               '[<"header", [<"vals" countWidth 16>]>, '
               '<"data", [<"vals", 1>]>]>')
  window = types.WindowType(window_id, "bitval_window", arg_struct, lowered, [
      types.WindowType.Frame("header", [types.WindowType.Field("vals", 0, 16)]),
      types.WindowType.Frame("data", [types.WindowType.Field("vals", 1, 0)]),
  ])

  hdr = _generate_header([window])
  win_name = _window_struct_name(hdr, "_bitval_window")
  assert f"struct {win_name} : public esi::SegmentedMessageData" in hdr
  # The data field "vals" stores a 3-bit uint, which becomes a bit-field.
  # The range accessor must use a lambda, not &data_frame::vals.
  assert "[](const data_frame &f) { return f.vals; }" in hdr
  assert "&data_frame::vals" not in hdr
  assert "std::vector<value_type> vals_vector() const" in hdr


def test_size_assert_emitted_for_struct():
  """Each packed struct gets a `static_assert` pinning its `sizeof`."""
  uint16 = types.UIntType("ui16", 16)
  sint8 = types.SIntType("si8", 8)
  s = types.StructType("!hw.struct<a: ui16, b: si8>", [("a", uint16),
                                                       ("b", sint8)])

  hdr = _generate_header([s])
  # Total: 16 + 8 = 24 bits = 3 bytes.
  assert "static_assert(sizeof(_struct_a_ui16_b_si8) == 3," in hdr
  assert "packed layout does not match manifest size" in hdr


def test_size_assert_emitted_for_union_and_wrappers():
  """Unions and their padding wrapper structs each get a size assert."""
  uint8 = types.UIntType("ui8", 8)
  uint16 = types.UIntType("ui16", 16)
  union_t = types.UnionType("!hw.union<a: ui8, b: ui16>", [("a", uint8),
                                                           ("b", uint16)])

  hdr = _generate_header([union_t])
  # The union itself is 2 bytes (max(8, 16) = 16 bits).
  assert "static_assert(sizeof(_union_a_ui8_b_ui16) == 2," in hdr
  # The wrapper around the narrow `a` field is also 2 bytes.
  assert "static_assert(sizeof(_union_a_ui8_b_ui16_a) == 2," in hdr


def test_size_assert_emitted_for_window_frames():
  """Both `data_frame` and `header_frame` get size asserts inside the window."""
  uint16 = types.UIntType("ui16", 16)
  uint32 = types.UIntType("ui32", 32)
  element_id = "!hw.struct<x: ui32, y: ui32>"
  list_id = f"!esi.list<{element_id}>"
  arg_struct_id = f"!hw.struct<coords: {list_id}>"
  header_struct_id = "!hw.struct<coords_count: ui16>"
  data_struct_id = f"!hw.struct<coords: !hw.array<1x{element_id}>>"
  lowered_id = f"!hw.union<header: {header_struct_id}, data: {data_struct_id}>"
  window_id = (f'!esi.window<"coords_only", {arg_struct_id}, '
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
      types.WindowType.Frame("header",
                             [types.WindowType.Field("coords", 0, 16)]),
      types.WindowType.Frame("data", [types.WindowType.Field("coords", 1, 0)]),
  ])

  hdr = _generate_header([window])
  # Both inner frames are sized to the wider data frame: 8 bytes per coord.
  assert "static_assert(sizeof(data_frame) == 8," in hdr
  assert "static_assert(sizeof(header_frame) == 8," in hdr


def test_size_assert_skipped_for_unbounded_struct():
  """Structs containing an `!esi.any` field have no static size, so no assert."""
  uint8 = types.UIntType("ui8", 8)
  any_t = types.AnyType("!esi.any")
  s = types.StructType("!hw.struct<tag: ui8, data: !esi.any>",
                       [("tag", uint8), ("data", any_t)])

  hdr = _generate_header([s])
  # The struct is still emitted, but no size assert (its size is unbounded).
  assert "struct _struct_tag_ui8_data__esi_any" in hdr
  assert "static_assert(sizeof(_struct_tag_ui8_data__esi_any)" not in hdr


def test_struct_with_void_field_commented_out():
  """Void-typed struct fields are commented out so the header stays valid C++.

  `void x;` is not a valid C++ field declaration. The codegen must instead
  emit `// void x;` and exclude the field from the generated constructor.
  """
  uint8 = types.UIntType("ui8", 8)
  void_t = types.VoidType("!esi.void")
  s = types.StructType(
      "!hw.struct<valid: ui8, client_data: void>",
      [("valid", uint8), ("client_data", void_t)],
  )

  hdr = _generate_header([s])
  # No uncommented `void <name>;` declaration may appear in the header.
  assert re.search(r"^\s*void\s+\w+;", hdr, re.M) is None, hdr
  # The void field must instead show up as a commented-out placeholder.
  assert "// void client_data;" in hdr
  # The non-void field is still emitted normally.
  assert "uint8_t valid" in hdr
  # The constructor must not take a `void` parameter (which would be
  # invalid C++); only the non-void field is in the parameter list.
  ctor_match = re.search(r"_struct_[^\s(]*\(([^)]*)\) :", hdr)
  assert ctor_match, f"Constructor not found in:\n{hdr}"
  ctor_params = ctor_match.group(1)
  assert "client_data" not in ctor_params
  assert "valid" in ctor_params


def test_union_with_void_field_commented_out():
  """Void-typed union members are commented out and skip wrapper generation."""
  uint16 = types.UIntType("ui16", 16)
  void_t = types.VoidType("!esi.void")
  u = types.UnionType(
      "!hw.union<a: ui16, b: void>",
      [("a", uint16), ("b", void_t)],
  )

  hdr = _generate_header([u])
  # No uncommented `void <name>;` declaration may appear in the header;
  # the void member must show up as a commented-out placeholder instead.
  assert re.search(r"^\s*void\s+\w+;", hdr, re.M) is None, hdr
  assert "// void b;" in hdr
  # No padding wrapper struct is generated for the void field.
  assert "_union_a_ui16_b__esi_void_b" not in hdr
  # The non-void member is still present in the union body.
  union_body = hdr[hdr.index("union "):]
  assert "uint16_t a;" in union_body
