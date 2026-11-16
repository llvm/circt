import json

import pytest

import esiaccel.types as types


def test_manifest_version():
  ctxt = types.cpp.Context()
  manifest = types.cpp.Manifest(
      ctxt, json.dumps({
          "apiVersion": 1,
          "types": [],
          "modules": []
      }))
  assert manifest.api_version == 1


@pytest.mark.parametrize("version", [0, 2, -1, 1 << 32 | 1, 1.5, "1", None])
def test_manifest_rejects_incompatible_version(version):
  ctxt = types.cpp.Context()
  # Reject the version before parsing any types or design metadata.
  with pytest.raises(RuntimeError, match="Unsupported ESI ABI version:"):
    types.cpp.Manifest(ctxt, json.dumps({"apiVersion": version}))


def test_types():
  void_type = types.VoidType("void")
  assert void_type is not None
  assert isinstance(void_type, types.VoidType)

  bits_type = types.BitsType("bits8", 8)
  assert bits_type is not None
  assert isinstance(bits_type, types.BitsType)

  uint_type = types.UIntType("uint32", 32)
  assert uint_type is not None
  assert isinstance(uint_type, types.UIntType)

  sint_type = types.SIntType("sint8", 8)
  assert sint_type is not None
  assert isinstance(sint_type, types.SIntType)
  assert sint_type.bit_width == 8

  struct_type = types.StructType(
      "mystruct",
      [("field1", types.UIntType("uint8", 8)),
       ("field2", types.UIntType("uint16", 16))],
  )
  assert struct_type is not None
  assert isinstance(struct_type, types.StructType)
  field_map = {name: field_type for name, field_type in struct_type.fields}
  assert isinstance(field_map["field1"], types.UIntType)
  assert isinstance(field_map["field2"], types.UIntType)

  array_type = types.ArrayType("uint8_array", types.UIntType("uint8", 8), 10)
  assert array_type is not None
  assert isinstance(array_type, types.ArrayType)
  assert hasattr(array_type, "element_type")
  assert isinstance(array_type.element_type, types.UIntType)
  assert hasattr(array_type, "size")
  assert array_type.size == 10

  any_type = types.AnyType("any")
  assert any_type is not None
  assert isinstance(any_type, types.AnyType)
  valid, reason = any_type.is_valid(0)
  assert not valid
  assert "any type" in reason
  assert any_type.bit_width == -1
  try:
    any_type.serialize(0)
  except ValueError as exc:
    assert "any type" in str(exc)
  else:
    assert False, "AnyType.serialize should raise"

  alias_inner = types.UIntType("alias_inner", 16)
  type_alias = types.TypeAlias("alias_scope", "aliasName", alias_inner)
  assert type_alias is not None
  assert isinstance(type_alias, types.TypeAlias)
  assert type_alias.name == "aliasName"
  assert isinstance(type_alias.inner_type, types.UIntType)
  assert type_alias.bit_width == alias_inner.bit_width
  alias_valid, alias_reason = type_alias.is_valid(42)
  inner_valid, inner_reason = alias_inner.is_valid(42)
  assert alias_valid == inner_valid
  assert alias_reason == inner_reason
  serialized = type_alias.serialize(42)
  inner_serialized = alias_inner.serialize(42)
  assert serialized == inner_serialized
  assert type_alias.deserialize(serialized) == alias_inner.deserialize(
      serialized)
  assert str(type_alias) == "aliasName"


def test_union_type():
  uint8 = types.UIntType("uint8", 8)
  uint16 = types.UIntType("uint16", 16)

  union_type = types.UnionType("myunion", [("a", uint8), ("b", uint16)])
  assert union_type is not None
  assert isinstance(union_type, types.UnionType)
  assert union_type.bit_width == 16

  field_map = {name: ty for name, ty in union_type.fields}
  assert isinstance(field_map["a"], types.UIntType)
  assert isinstance(field_map["b"], types.UIntType)

  # is_valid: single active field
  valid, reason = union_type.is_valid({"a": 42})
  assert valid, reason
  valid, reason = union_type.is_valid({"b": 1000})
  assert valid, reason

  # is_valid: wrong number of fields
  valid, reason = union_type.is_valid({"a": 1, "b": 2})
  assert not valid
  assert "exactly 1" in reason

  # is_valid: unknown field
  valid, reason = union_type.is_valid({"c": 1})
  assert not valid
  assert "unknown" in reason

  # is_valid: not a dict
  valid, reason = union_type.is_valid(42)
  assert not valid

  # serialize / deserialize round-trip through field "a"
  serialized_a = union_type.serialize({"a": 42})
  assert serialized_a == bytearray([42, 0])
  (deserialized, remaining) = union_type.deserialize(serialized_a)
  assert remaining == bytearray()
  assert "a" in deserialized
  assert "b" in deserialized
  assert deserialized["a"] == 42
  assert deserialized["b"] == 42

  # serialize / deserialize round-trip through field "b"
  # Field "b" is 2 bytes (full width), no padding needed.
  serialized_b = union_type.serialize({"b": 0x1234})
  assert len(serialized_b) == 2
  assert serialized_b == bytearray([0x34, 0x12])  # little-endian, no padding
  (deserialized_b,
   remaining_b) = union_type.deserialize(serialized_b + bytearray([0xDE, 0xAD]))
  assert remaining_b == bytearray([0xDE, 0xAD])
  assert deserialized_b["a"] == 0x34
  assert deserialized_b["b"] == 0x1234


def test_union_padding_with_struct():
  """Union members share the LSB without changing their internal layout."""
  uint8 = types.UIntType("uint8", 8)
  uint32 = types.UIntType("uint32", 32)
  small_struct = types.StructType("!hw.struct<x: ui8, y: ui8>", [("x", uint8),
                                                                 ("y", uint8)])
  union_type = types.UnionType("myunion2", [("wide", uint32),
                                            ("narrow", small_struct)])
  assert union_type.bit_width == 32

  serialized = union_type.serialize({"narrow": {"x": 0xAA, "y": 0xBB}})
  assert serialized == bytearray([0xBB, 0xAA, 0, 0])

  (result, leftover) = union_type.deserialize(serialized)
  assert leftover == bytearray()
  assert result["narrow"] == {"x": 0xAA, "y": 0xBB}
  assert result["wide"] == 0xAABB


def test_union_signed_variant():
  union_type = types.UnionType("signed_union",
                               [("small", types.SIntType("si8", 8)),
                                ("wide", types.UIntType("ui16", 16))])
  assert union_type.serialize({"small": -2}) == bytearray([0xFE, 0])
  result, remaining = union_type.deserialize(bytearray([0x80, 0xAB]))
  assert result == {"small": -128, "wide": 0xAB80}
  assert remaining == bytearray()


@pytest.mark.parametrize("use_alias", [False, True])
@pytest.mark.parametrize("union_width", [8, 12, 16])
def test_union_subbyte_members(use_alias, union_width):
  signed = types.SIntType("si5", 5)
  if use_alias:
    signed = types.TypeAlias("signed_alias", "Signed", signed)
  union_type = types.UnionType(
      "subbyte_union",
      [("signed", signed), ("unsigned", types.UIntType("ui3", 3)),
       ("bits", types.BitsType("i5", 5)),
       ("wide", types.UIntType(f"ui{union_width}", union_width))])
  union_bytes = (union_width + 7) // 8
  for value in [-16, -7, -1, 0, 7, 15]:
    serialized = union_type.serialize({"signed": value})
    assert serialized == bytearray([value & 0x1F]) + bytearray(union_bytes - 1)
    decoded, remaining = union_type.deserialize(serialized)
    assert decoded["signed"] == value
    assert decoded["unsigned"] == value & 7
    assert decoded["bits"] == bytearray([value & 0x1F])
    assert remaining == bytearray()

  raw = bytearray([0xBC]) + bytearray([0x0A] * (union_bytes - 1))
  decoded, remaining = union_type.deserialize(raw + bytearray([0xDE, 0xAD]))
  assert decoded["signed"] == -4
  assert decoded["unsigned"] == 4
  assert decoded["bits"] == bytearray([0x1C])
  assert remaining == bytearray([0xDE, 0xAD])

  bits = bytearray([0xF9])
  assert union_type.serialize({"bits": bits}) == (bytearray([0x19]) +
                                                  bytearray(union_bytes - 1))
  assert bits == bytearray([0xF9])

  with pytest.raises(ValueError, match="insufficient data for union"):
    union_type.deserialize(bytearray(union_bytes - 1))


def test_union_in_struct():
  uint8 = types.UIntType("ui8", 8)
  union_type = types.UnionType("nested_union",
                               [("small", uint8),
                                ("wide", types.UIntType("ui16", 16))])
  struct_type = types.StructType("union_struct", [("tag", uint8),
                                                  ("payload", union_type),
                                                  ("tail", uint8)])
  serialized = struct_type.serialize({
      "tag": 0x12,
      "payload": {
          "small": 0xA5
      },
      "tail": 0x87
  })
  assert serialized == bytearray([0x87, 0xA5, 0, 0x12])
  result, remaining = struct_type.deserialize(serialized)
  assert result == {
      "tag": 0x12,
      "payload": {
          "small": 0xA5,
          "wide": 0xA5
      },
      "tail": 0x87
  }
  assert remaining == bytearray()


def test_list_type_not_supported_for_host():
  element_type = types.UIntType("ui8", 8)
  list_type = types._get_esi_type(
      types.cpp.ListType("!esi.list<ui8>", element_type.cpp_type))

  assert isinstance(list_type, types.ListType)
  assert isinstance(list_type.element_type, types.UIntType)
  assert list_type.element_type.id == element_type.id
  assert list_type.bit_width == -1

  supports_host, reason = list_type.supports_host
  assert not supports_host
  assert reason == "list types require an enclosing window encoding"

  valid, invalid_reason = list_type.is_valid([0x12, 0x34])
  assert valid
  assert invalid_reason is None

  valid, invalid_reason = list_type.is_valid("not a list")
  assert not valid
  assert "must be a list" in invalid_reason

  valid, invalid_reason = list_type.is_valid([0x12, 0x1FF])
  assert not valid
  assert invalid_reason == "invalid element 1: out of range: 511"

  with pytest.raises(ValueError, match="cannot be serialized without a window"):
    list_type.serialize([0x12, 0x34])

  with pytest.raises(ValueError,
                     match="cannot be deserialized without a window"):
    list_type.deserialize(bytearray([0x12, 0x34]))


def test_window_type_not_supported_for_host():
  uint8 = types.UIntType("ui8", 8)
  into_type = types.StructType("!hw.struct<header: ui8, data: ui8>",
                               [("header", uint8), ("data", uint8)])
  header_type = types.StructType("!hw.struct<header: ui8>", [("header", uint8)])
  data_type = types.StructType("!hw.struct<data: ui8>", [("data", uint8)])
  lowered_type = types.UnionType(
      "!hw.union<header: !hw.struct<header: ui8>, data: !hw.struct<data: ui8>>",
      [("header", header_type), ("data", data_type)])
  window_type = types.WindowType(
      '!esi.window<"test_window", !hw.struct<header: ui8, data: ui8>, '
      '[<"header", [<"header">]>, <"data", [<"data">]>]>',
      "test_window",
      into_type,
      lowered_type,
      [
          types.WindowType.Frame("header",
                                 [types.WindowType.Field("header", 0, 0)]),
          types.WindowType.Frame("data",
                                 [types.WindowType.Field("data", 0, 0)]),
      ],
  )

  supports_host, reason = window_type.supports_host
  assert not supports_host
  assert reason is not None
  assert "not yet supported" in reason

  valid, invalid_reason = window_type.is_valid({"header": 1, "data": 2})
  assert not valid
  assert invalid_reason == reason

  with pytest.raises(ValueError, match="not yet supported"):
    window_type.serialize({"header": 1, "data": 2})

  with pytest.raises(ValueError, match="not yet supported"):
    window_type.deserialize(bytearray([1, 2]))
