// RUN: circt-opt %s --hw-convert-bitcasts=allow-partial-conversion=true                | FileCheck %s --check-prefixes=CHECK,NOCANON
// RUN: circt-opt %s --hw-convert-bitcasts=allow-partial-conversion=true --canonicalize | FileCheck %s --check-prefixes=CHECK,CANON

// NOCANON-LABEL: hw.module @intToArray
hw.module @intToArray(in %i: i18, out o : !hw.array<3xi6>) {
  // NOCANON-DAG: %[[B12:.+]] = comb.extract %i from 12 : (i18) -> i6
  // NOCANON-DAG: %[[B6:.+]]  = comb.extract %i from 6 : (i18) -> i6
  // NOCANON-DAG: %[[B0:.+]]  = comb.extract %i from 0 : (i18) -> i6
  // NOCANON: %[[OUT:.+]] = hw.array_create %[[B12]], %[[B6]], %[[B0]] : i6
  %o = hw.bitcast %i : (i18) -> !hw.array<3xi6>
  // NOCANON: hw.output %[[OUT]]
  hw.output %o : !hw.array<3xi6>
}

// NOCANON-LABEL: hw.module @arrayToInt
hw.module @arrayToInt(in %i: !hw.array<3xi6>, out o : i18) {
  // NOCANON-DAG: %[[C2:.+]] = hw.constant -2 : i2
  // NOCANON-DAG: %[[C1:.+]] = hw.constant 1 : i2
  // NOCANON-DAG: %[[C0:.+]] = hw.constant 0 : i2
  // NOCANON-DAG: %[[E2:.+]] = hw.array_get %i[%[[C2]]] : !hw.array<3xi6>, i2
  // NOCANON-DAG: %[[E1:.+]] = hw.array_get %i[%[[C1]]] : !hw.array<3xi6>, i2
  // NOCANON-DAG: %[[E0:.+]] = hw.array_get %i[%[[C0]]] : !hw.array<3xi6>, i2
  // NOCANON: %[[OUT:.+]] = comb.concat %[[E2]], %[[E1]], %[[E0]]
  %o = hw.bitcast %i : (!hw.array<3xi6>) -> i18
  // NOCANON: hw.output %[[OUT]]
  hw.output %o : i18
}

// NOCANON-LABEL: hw.module @intToStruct
hw.module @intToStruct(in %i: i18, out o : !hw.struct<a: i1, b: i10, c: i7>) {
  // NOCANON-DAG: %[[B17:.+]] = comb.extract %i from 17 : (i18) -> i1
  // NOCANON-DAG: %[[B7:.+]] = comb.extract %i from 7 : (i18) -> i10
  // NOCANON-DAG: %[[B0:.+]] = comb.extract %i from 0 : (i18) -> i7
  // NOCANON: %[[OUT:.+]] = hw.struct_create (%[[B17]], %[[B7]], %[[B0]]) : !hw.struct<a: i1, b: i10, c: i7>
  %o = hw.bitcast %i : (i18) -> !hw.struct<a: i1, b: i10, c: i7>
  // NOCANON: hw.output %[[OUT]]
  hw.output %o : !hw.struct<a: i1, b: i10, c: i7>
}

// NOCANON-LABEL: hw.module @structToInt
hw.module @structToInt(in %i: !hw.struct<a: i1, b: i10, c: i7>, out o : i18) {
  // NOCANON: %[[EA:.+]], %[[EB:.+]], %[[EC:.+]] = hw.struct_explode %i : !hw.struct<a: i1, b: i10, c: i7>
  // NOCANON: %[[OUT:.+]] = comb.concat %[[EA]], %[[EB]], %[[EC]] : i1, i10, i7
  %o = hw.bitcast %i : (!hw.struct<a: i1, b: i10, c: i7>) -> i18
  // NOCANON: hw.output %[[OUT]]
  hw.output %o : i18
}

// Don't crash on unsupported types
// NOCANON-LABEL: hw.module @unsupported
hw.module @unsupported(in %i: i8, out o : i8) {
  // NOCANON: hw.bitcast
  // NOCANON: hw.bitcast
  %a = hw.bitcast %i : (i8) -> !hw.union<foo: i8, bar: i8>
  %o = hw.bitcast %a : (!hw.union<foo: i8, bar: i8>) -> i8
  hw.output %o : i8
}

// NOCANON-LABEL: hw.module @nop
hw.module @nop(in %i: i8, out o : i8) {
  %o = hw.bitcast %i : (i8) -> i8
  // NOCANON: hw.output %i : i8
  hw.output %o : i8
}

// Don't crash on zero width
// NOCANON-LABEL: hw.module @zeroWidth
hw.module @zeroWidth(in %i: i0, out o : i0) {
  %0 = hw.bitcast %i : (i0) -> !hw.struct<a: i0, b: i0, c: i0>
  %1 = hw.bitcast %0 : (!hw.struct<a: i0, b: i0, c: i0>) -> !hw.array<8xi0>
  %2 = hw.bitcast %1 : (!hw.array<8xi0>) -> !hw.struct<a: i0>
  %3 = hw.bitcast %2 : (!hw.struct<a: i0>) -> i0
  // NOCANON: hw.output %{{.+}} : i0
  hw.output %3 : i0
}

// CHECK-LABEL: hw.module @arrayRoundtrip
hw.module @arrayRoundtrip(in %raw: i32, out o : i32) {
  // CHECK-NOT: hw.bitcast
  %a = hw.bitcast %raw : (i32) -> !hw.array<4x!hw.array<2xi4>>
  %o = hw.bitcast %a : (!hw.array<4x!hw.array<2xi4>>) -> i32
  // CANON: hw.output %raw : i32
  hw.output %o : i32
}

// CHECK-LABEL: hw.module @structRoundtrip
hw.module @structRoundtrip(in %raw: i32, out o : i32) {
  // CHECK-NOT: hw.bitcast
  %a = hw.bitcast %raw : (i32) -> !hw.struct<a: i8, b: i8, c: i16>
  %o = hw.bitcast %a : (!hw.struct<a: i8, b: i8, c: i16>) -> i32
  // CANON: hw.output %raw : i32
  hw.output %o : i32
}

// CHECK-LABEL: hw.module @mixedRoundtrip
hw.module @mixedRoundtrip(in %raw: i32, out o : i32) {
  // CHECK-NOT: hw.bitcast
  %0 = hw.bitcast %raw : (i32) -> !hw.struct<a: !hw.array<2xi8>, b: i0, c: i16>
  %1 = hw.bitcast %0 : (!hw.struct<a: !hw.array<2xi8>, b: i0, c: i16>) -> !hw.array<32x!hw.struct<a: i1>>
  %2 = hw.bitcast %1 : (!hw.array<32x!hw.struct<a: i1>>) -> i32
  // CANON: hw.output %raw : i32
  hw.output %2 : i32
}
