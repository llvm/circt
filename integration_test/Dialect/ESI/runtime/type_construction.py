# REQUIRES: esi-runtime
# RUN: %PYTHON% %s

import esiaccel.types as types

# Test VoidType construction
void_type = types.VoidType("void")
assert void_type is not None
assert isinstance(void_type, types.VoidType)

# Test BitsType construction
bits_type = types.BitsType("bits8", 8)
assert bits_type is not None
assert isinstance(bits_type, types.BitsType)

# Test UIntType construction
uint_type = types.UIntType("uint32", 32)
assert uint_type is not None
assert isinstance(uint_type, types.UIntType)

# Test SIntType construction
sint_type = types.SIntType("sint8", 8)
assert sint_type is not None
assert isinstance(sint_type, types.SIntType)
assert sint_type.bit_width == 8

# Test StructType construction
struct_type = types.StructType("mystruct",
                               [("field1", types.UIntType("uint8", 8)),
                                ("field2", types.UIntType("uint16", 16))])
assert struct_type is not None
assert isinstance(struct_type, types.StructType)
assert isinstance(struct_type.field("field1"), types.UIntType)
assert isinstance(struct_type.field("field2"), types.UIntType)

# Test ArrayType construction
array_type = types.ArrayType("uint8_array", types.UIntType("uint8", 8), 10)
assert array_type is not None
assert isinstance(array_type, types.ArrayType)
assert hasattr(array_type, "element_type")
assert isinstance(array_type.element_type, types.UIntType)
assert hasattr(array_type, "size")
assert array_type.size == 10

# Test AnyType construction
any_type = types.AnyType("any")
assert any_type is not None
assert isinstance(any_type, types.AnyType)
supports, reason = any_type.supports_host
assert not supports
assert "any type" in reason
assert any_type.bit_width == -1
try:
  any_type.serialize(0)
except ValueError as exc:
  assert "any type" in str(exc)
else:
  assert False, "AnyType.serialize should raise"

# Test TypeAliasType construction
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
assert type_alias.deserialize(serialized) == alias_inner.deserialize(serialized)
assert str(type_alias) == "aliasName"

print("SUCCESS!")
