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

// CHECK: !moore.f64
unrealized_conversion_cast to !moore.f64
// CHECK: !moore.f32
unrealized_conversion_cast to !moore.f32

// Packed Arrays
// CHECK: !moore.array<4 x i42>
unrealized_conversion_cast to !moore.array<4 x i42>
// CHECK: !moore.open_array<i42>
unrealized_conversion_cast to !moore.open_array<i42>

// Unpacked arrays
// CHECK: !moore.uarray<4 x i42>
unrealized_conversion_cast to !moore.uarray<4 x i42>
// CHECK: !moore.uarray<4 x string>
unrealized_conversion_cast to !moore.uarray<4 x string>
// CHECK: !moore.open_uarray<string>
unrealized_conversion_cast to !moore.open_uarray<string>
// CHECK: !moore.assoc_array<string, chandle>
unrealized_conversion_cast to !moore.assoc_array<string, chandle>
// CHECK: !moore.queue<string, 42>
unrealized_conversion_cast to !moore.queue<string, 42>

// Packed structs
// CHECK: !moore.struct<{}>
unrealized_conversion_cast to !moore.struct<{}>
// CHECK: !moore.struct<{foo: i42}>
unrealized_conversion_cast to !moore.struct<{foo: i42}>
// CHECK: !moore.struct<{foo: i42, bar: i1337}>
unrealized_conversion_cast to !moore.struct<{foo: i42, bar: i1337}>
// CHECK: !moore.struct<{"a b": i42}>
unrealized_conversion_cast to !moore.struct<{"a b": i42}>

// Unpacked structs
// CHECK: !moore.ustruct<{}>
unrealized_conversion_cast to !moore.ustruct<{}>
// CHECK: !moore.ustruct<{foo: string}>
unrealized_conversion_cast to !moore.ustruct<{foo: string}>
// CHECK: !moore.ustruct<{foo: i42, bar: string}>
unrealized_conversion_cast to !moore.ustruct<{foo: i42, bar: string}>
// CHECK: !moore.ustruct<{"a b": string}>
unrealized_conversion_cast to !moore.ustruct<{"a b": string}>
