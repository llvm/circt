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

# Test ArrayType construction
array_type = types.ArrayType("uint8_array", types.UIntType("uint8", 8), 10)
assert array_type is not None
assert isinstance(array_type, types.ArrayType)

print("SUCCESS!")
