// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @UnitTypes(
func.func @UnitTypes(
  // CHECK-SAME: %arg0: !moore.void
  // CHECK-SAME: %arg1: !moore.string
  // CHECK-SAME: %arg2: !moore.chandle
  // CHECK-SAME: %arg3: !moore.event
  %arg0: !moore.void,
  %arg1: !moore.string,
  %arg2: !moore.chandle,
  %arg3: !moore.event
) { return }

// CHECK-LABEL: func @IntTypes(
func.func @IntTypes(
  // CHECK-SAME: %arg0: !moore.bit
  // CHECK-SAME: %arg1: !moore.logic
  // CHECK-SAME: %arg2: !moore.reg
  // CHECK-SAME: %arg3: !moore.byte
  // CHECK-SAME: %arg4: !moore.shortint
  // CHECK-SAME: %arg5: !moore.int
  // CHECK-SAME: %arg6: !moore.longint
  // CHECK-SAME: %arg7: !moore.integer
  // CHECK-SAME: %arg8: !moore.time
  %arg0: !moore.bit,
  %arg1: !moore.logic,
  %arg2: !moore.reg,
  %arg3: !moore.byte,
  %arg4: !moore.shortint,
  %arg5: !moore.int,
  %arg6: !moore.longint,
  %arg7: !moore.integer,
  %arg8: !moore.time
) { return }

// CHECK-LABEL: func @RealTypes(
func.func @RealTypes(
  // CHECK-SAME: %arg0: !moore.shortreal
  // CHECK-SAME: %arg1: !moore.real
  // CHECK-SAME: %arg2: !moore.realtime
  %arg0: !moore.shortreal,
  %arg1: !moore.real,
  %arg2: !moore.realtime
) { return }

// CHECK-LABEL: func @DimTypes(
func.func @DimTypes(
  // CHECK-SAME: %arg0: !moore.packed<unsized<bit>>,
  // CHECK-SAME: %arg1: !moore.packed<range<bit, 4:-5>>,
  %arg0: !moore.packed<unsized<bit>>,
  %arg1: !moore.packed<range<bit, 4:-5>>,
  // CHECK-SAME: %arg2: !moore.unpacked<unsized<bit>>,
  // CHECK-SAME: %arg3: !moore.unpacked<array<bit, 42>>,
  // CHECK-SAME: %arg4: !moore.unpacked<range<bit, 4:-5>>,
  // CHECK-SAME: %arg5: !moore.unpacked<assoc<bit>>,
  // CHECK-SAME: %arg6: !moore.unpacked<assoc<bit, string>>,
  // CHECK-SAME: %arg7: !moore.unpacked<queue<bit>>,
  // CHECK-SAME: %arg8: !moore.unpacked<queue<bit, 9001>>
  %arg2: !moore.unpacked<unsized<bit>>,
  %arg3: !moore.unpacked<array<bit, 42>>,
  %arg4: !moore.unpacked<range<bit, 4:-5>>,
  %arg5: !moore.unpacked<assoc<bit>>,
  %arg6: !moore.unpacked<assoc<bit, string>>,
  %arg7: !moore.unpacked<queue<bit>>,
  %arg8: !moore.unpacked<queue<bit, 9001>>
) {
  return
}

// CHECK-LABEL: func @StructTypes(
func.func @StructTypes(
  // CHECK-SAME: %arg0: !moore.packed<struct<{}>>
  // CHECK-SAME: %arg1: !moore.packed<struct<{foo: bit, bar: int}>>
  %arg0: !moore.packed<struct<{}>>,
  %arg1: !moore.packed<struct<{foo: bit, bar: int}>>,
  // CHECK-SAME: %arg2: !moore.unpacked<struct<{}>>
  // CHECK-SAME: %arg3: !moore.unpacked<struct<{foo: string, bar: event}>>
  %arg2: !moore.unpacked<struct<{}>>,
  %arg3: !moore.unpacked<struct<{foo: string, bar: event}>>
) { return }
