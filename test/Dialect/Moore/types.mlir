// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: !moore.void
unrealized_conversion_cast to !moore.void
// CHECK: !moore.string
unrealized_conversion_cast to !moore.string
// CHECK: !moore.chandle
unrealized_conversion_cast to !moore.chandle
// CHECK: !moore.event
unrealized_conversion_cast to !moore.event

// CHECK: !moore.i42
unrealized_conversion_cast to !moore.i42
// CHECK: !moore.l42
unrealized_conversion_cast to !moore.l42

// CHECK: !moore.real
unrealized_conversion_cast to !moore.real

// Packed Arrays
// CHECK: !moore.array<4 x !moore.i42>
unrealized_conversion_cast to !moore.array<4 x !moore.i42>
// CHECK: !moore.open_array<!moore.i42>
unrealized_conversion_cast to !moore.open_array<!moore.i42>

// Unpacked arrays
// CHECK: !moore.uarray<4 x !moore.i42>
unrealized_conversion_cast to !moore.uarray<4 x !moore.i42>
// CHECK: !moore.uarray<4 x !moore.string>
unrealized_conversion_cast to !moore.uarray<4 x !moore.string>
// CHECK: !moore.open_uarray<!moore.string>
unrealized_conversion_cast to !moore.open_uarray<!moore.string>
// CHECK: !moore.assoc_array<!moore.string, !moore.chandle>
unrealized_conversion_cast to !moore.assoc_array<!moore.string, !moore.chandle>
// CHECK: !moore.queue<!moore.string, 42>
unrealized_conversion_cast to !moore.queue<!moore.string, 42>

// CHECK-LABEL: func @StructTypes(
func.func @StructTypes(
  // CHECK-SAME: %arg0: !moore.packed<struct<{}>>
  // CHECK-SAME: %arg1: !moore.packed<struct<{foo: i1, bar: i32}>>
  %arg0: !moore.packed<struct<{}>>,
  %arg1: !moore.packed<struct<{foo: i1, bar: i32}>>,
  // CHECK-SAME: %arg2: !moore.unpacked<struct<{}>>
  // CHECK-SAME: %arg3: !moore.unpacked<struct<{foo: string, bar: event}>>
  %arg2: !moore.unpacked<struct<{}>>,
  %arg3: !moore.unpacked<struct<{foo: string, bar: event}>>
) { return }
