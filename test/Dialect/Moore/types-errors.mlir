// RUN: circt-opt --verify-diagnostics --split-input-file %s

// -----
// expected-error @+1 {{ambiguous packing; wrap `struct` in `packed<...>` or `unpacked<...>` to disambiguate}}
func.func @Foo(%arg0: !moore.struct<{}, loc(unknown)>) { return }

// -----
// expected-error @below {{invalid kind of type specified}}
// expected-error @below {{parameter 'elementType' which is to be a `PackedType`}}
unrealized_conversion_cast to !moore.array<4 x !moore.string>

// -----
// expected-error @below {{invalid kind of type specified}}
// expected-error @below {{parameter 'elementType' which is to be a `PackedType`}}
unrealized_conversion_cast to !moore.open_array<!moore.string>
