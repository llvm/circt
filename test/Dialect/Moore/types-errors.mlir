// RUN: circt-opt --verify-diagnostics --split-input-file %s

// -----
// expected-error @below {{invalid kind of type specified}}
// expected-error @below {{parameter 'elementType' which is to be a `PackedType`}}
unrealized_conversion_cast to !moore.array<4 x string>

// -----
// expected-error @below {{invalid kind of type specified}}
// expected-error @below {{parameter 'elementType' which is to be a `PackedType`}}
unrealized_conversion_cast to !moore.open_array<string>

// -----
// expected-error @below {{StructType/UnionType members must be packed types}}
unrealized_conversion_cast to !moore.struct<{foo: string}>

// -----
// expected-error @below {{StructType/UnionType members must be packed types}}
unrealized_conversion_cast to !moore.union<{foo: string}>
