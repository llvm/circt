// RUN: circt-opt %s --hw-convert-bitcasts --canonicalize | FileCheck %s
// RUN: circt-opt %s --canonicalize                       | FileCheck %s

// CHECK-LABEL: hw.module @intToArray
hw.module @intToArray(out o : !hw.array<3xi4>) {
  %c = hw.constant 0x123 : i12
  %o = hw.bitcast %c : (i12) -> !hw.array<3xi4>
  // CHECK-NEXT: %[[OUT:.+]] = hw.aggregate_constant [1 : i4, 2 : i4, 3 : i4] : !hw.array<3xi4>
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %o : !hw.array<3xi4>
}

// CHECK-LABEL: hw.module @arrayToInt
hw.module @arrayToInt(out o : i24) {
  %c = hw.aggregate_constant [0x1 : i8, 0x0 : i8, 0x2 : i8] : !hw.array<3xi8>
  %o = hw.bitcast %c : (!hw.array<3xi8>) -> i24
  // CHECK-NEXT: %[[OUT:.+]] = hw.constant 65538 : i24
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %o : i24
}

// CHECK-LABEL: hw.module @intToStruct
hw.module @intToStruct(out o : !hw.struct<a: i4, b: i4, c: i8>) {
  %c = hw.constant 0x1234 : i16
  %o = hw.bitcast %c : (i16) -> !hw.struct<a: i4, b: i4, c: i8>
  // CHECK-NEXT: %[[OUT:.+]] = hw.aggregate_constant [1 : i4, 2 : i4, 52 : i8] : !hw.struct<a: i4, b: i4, c: i8>
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %o : !hw.struct<a: i4, b: i4, c: i8>
}

// CHECK-LABEL: hw.module @structToInt
hw.module @structToInt(out o : i16) {
  %c = hw.aggregate_constant [0x1 : i4, 0x0 : i4, 0x2 : i8] : !hw.struct<a: i4, b: i4, c: i8>
  %o = hw.bitcast %c : (!hw.struct<a: i4, b: i4, c: i8>) -> i16
  // CHECK-NEXT: %[[OUT:.+]] = hw.constant 4098 : i16
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %o : i16
}

// CHECK-LABEL: hw.module @structToArray
hw.module @structToArray(out o : !hw.array<4xi8>) {
  %c = hw.aggregate_constant [0x1 : i8, 0x2 : i8, 0x3 : i16] : !hw.struct<a: i8, b: i8, c: i16>
  %o = hw.bitcast %c : (!hw.struct<a: i8, b: i8, c: i16>) -> !hw.array<4xi8>
  // CHECK-NEXT: %[[OUT:.+]] = hw.aggregate_constant [1 : i8, 2 : i8, 0 : i8, 3 : i8] : !hw.array<4xi8>
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %o : !hw.array<4xi8>
}

// CHECK-LABEL: hw.module @nop
hw.module @nop(out o : i8) {
  %c = hw.constant 5 : i8
  %o = hw.bitcast %c : (i8) -> i8
  // CHECK-NEXT: %[[OUT:.+]] = hw.constant 5 : i8
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %o : i8
}

// Don't crash on zero width
// CHECK-LABEL: hw.module @zeroWidth
hw.module @zeroWidth(out o : i0) {
  %c = hw.constant 0 : i0
  %0 = hw.bitcast %c : (i0) -> !hw.struct<a: i0, b: i0, c: i0>
  %1 = hw.bitcast %0 : (!hw.struct<a: i0, b: i0, c: i0>) -> !hw.array<8xi0>
  %2 = hw.bitcast %1 : (!hw.array<8xi0>) -> !hw.struct<a: i0>
  %3 = hw.bitcast %2 : (!hw.struct<a: i0>) -> i0
  // CHECK-NEXT: %[[OUT:.+]] = hw.constant 0 : i0
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %3 : i0
}

// CHECK-LABEL: hw.module @arrayRoundtrip
hw.module @arrayRoundtrip(out o : i32) {
  // CHECK-NOT: hw.bitcast
  %c = hw.constant 0x12345678 : i32
  %a = hw.bitcast %c : (i32) -> !hw.array<4x!hw.array<2xi4>>
  %o = hw.bitcast %a : (!hw.array<4x!hw.array<2xi4>>) -> i32
  // CHECK-NEXT: %[[OUT:.+]] = hw.constant 305419896 : i32
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %o : i32
}

// CHECK-LABEL: hw.module @structRoundtrip
hw.module @structRoundtrip(out o : i32) {
  // CHECK-NOT: hw.bitcast
  %c = hw.constant 0x00010002 : i32
  %a = hw.bitcast %c : (i32) -> !hw.struct<a: i8, b: i8, c: i16>
  %o = hw.bitcast %a : (!hw.struct<a: i8, b: i8, c: i16>) -> i32
  // CHECK-NEXT: %[[OUT:.+]] = hw.constant 65538 : i32
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %o : i32
}

// CHECK-LABEL: hw.module @mixedRoundtrip
hw.module @mixedRoundtrip(out o : i32) {
  // CHECK-NOT: hw.bitcast
  %c = hw.constant 0x01020003 : i32
  %0 = hw.bitcast %c : (i32) -> !hw.struct<a: !hw.array<2xi8>, b: i0, c: i16>
  %1 = hw.bitcast %0 : (!hw.struct<a: !hw.array<2xi8>, b: i0, c: i16>) -> !hw.array<32x!hw.struct<a: i1>>
  %2 = hw.bitcast %1 : (!hw.array<32x!hw.struct<a: i1>>) -> i32
  // CHECK-NEXT: %[[OUT:.+]] = hw.constant 16908291 : i32
  // CHECK-NEXT: hw.output %[[OUT]]
  hw.output %2 : i32
}
