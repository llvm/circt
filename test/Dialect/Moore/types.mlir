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
  // CHECK-SAME: %arg0: !moore.i42
  // CHECK-SAME: %arg1: !moore.l42
  %arg0: !moore.i42,
  %arg1: !moore.l42
) { return }

// CHECK-LABEL: func @RealTypes(
func.func @RealTypes(
  // CHECK-SAME: %arg0: !moore.real
  %arg0: !moore.real
) { return }

// CHECK-LABEL: func @DimTypes(
func.func @DimTypes(
  // CHECK-SAME: %arg0: !moore.packed<unsized<i1>>,
  // CHECK-SAME: %arg1: !moore.packed<range<i1, 4:-5>>,
  %arg0: !moore.packed<unsized<i1>>,
  %arg1: !moore.packed<range<i1, 4:-5>>,
  // CHECK-SAME: %arg2: !moore.unpacked<unsized<i1>>,
  // CHECK-SAME: %arg3: !moore.unpacked<array<i1, 42>>,
  // CHECK-SAME: %arg4: !moore.unpacked<range<i1, 4:-5>>,
  // CHECK-SAME: %arg5: !moore.unpacked<assoc<i1>>,
  // CHECK-SAME: %arg6: !moore.unpacked<assoc<i1, string>>,
  // CHECK-SAME: %arg7: !moore.unpacked<queue<i1>>,
  // CHECK-SAME: %arg8: !moore.unpacked<queue<i1, 9001>>
  %arg2: !moore.unpacked<unsized<i1>>,
  %arg3: !moore.unpacked<array<i1, 42>>,
  %arg4: !moore.unpacked<range<i1, 4:-5>>,
  %arg5: !moore.unpacked<assoc<i1>>,
  %arg6: !moore.unpacked<assoc<i1, string>>,
  %arg7: !moore.unpacked<queue<i1>>,
  %arg8: !moore.unpacked<queue<i1, 9001>>
) {
  return
}

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
