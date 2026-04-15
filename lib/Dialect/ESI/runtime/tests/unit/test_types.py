import pytest

import esiaccel.types as types


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
