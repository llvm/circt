// RUN: circt-opt %s --split-input-file --hw-specialize | FileCheck %s

// Test two different ways of instantiating a generic module.

module {

// CHECK-LABEL:  hw.module @addToFirst_N_4_X_8(%vec: !hw.array<4xi8>, %a: i8) -> (out: i8) {
// CHECK:          %c0_i64 = arith.constant 0 : i64
// CHECK:          %0 = comb.extract %c0_i64 from 0 : (i64) -> i2
// CHECK:          %1 = hw.array_get %vec[%0] : !hw.array<4xi8>
// CHECK:          %2 = comb.add %1, %a : i8
// CHECK:          hw.output %2 : i8
// CHECK:        }
// CHECK-LABEL:  hw.module @addToFirst_N_5_X_9(%vec: !hw.array<5xi9>, %a: i9) -> (out: i9) {
// CHECK:          %c0_i64 = arith.constant 0 : i64
// CHECK:          %0 = comb.extract %c0_i64 from 0 : (i64) -> i3
// CHECK:          %1 = hw.array_get %vec[%0] : !hw.array<5xi9>
// CHECK:          %2 = comb.add %1, %a : i9
// CHECK:          hw.output %2 : i9
// CHECK:        }
  hw.module @addToFirst<N: i32, X: i32>(
      %vec : !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"X">>>,
      %a : !hw.int<#hw.param.decl.ref<"X">>) -> (out: !hw.int<#hw.param.decl.ref<"X">>) {
    %c0 = arith.constant 0 : i64
    %first = hw.array_get %vec[%c0] : !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"X">>>
    %0 = comb.add %first, %a : !hw.int<#hw.param.decl.ref<"X">>
    hw.output %0 : !hw.int<#hw.param.decl.ref<"X">>
  }

// CHECK-LABEL:  hw.module @top(%vec1: !hw.array<4xi8>, %a1: i8, %vec2: !hw.array<5xi9>, %a2: i9) -> (out1: i8, out2: i9) {
// CHECK:          %inst1.out = hw.instance "inst1" @addToFirst_N_4_X_8(vec: %vec1: !hw.array<4xi8>, a: %a1: i8) -> (out: i8)
// CHECK:          %inst2.out = hw.instance "inst2" @addToFirst_N_5_X_9(vec: %vec2: !hw.array<5xi9>, a: %a2: i9) -> (out: i9)
// CHECK:          hw.output %inst1.out, %inst2.out : i8, i9
  hw.module @top(
      %vec1 : !hw.array<4 x !hw.int<8>>, %a1 : !hw.int<8>,
      %vec2 : !hw.array<5 x !hw.int<9>>, %a2 : !hw.int<9>) ->
      (out1: !hw.int<8>, out2: !hw.int<9>) {
    %0 = hw.instance "inst1" @addToFirst<N: i32 = 4, X: i32 = 8>
      (vec: %vec1 : !hw.array<4 x !hw.int<8>>, a: %a1 : !hw.int<8>) -> (out: !hw.int<8>)
    %1 = hw.instance "inst2" @addToFirst<N: i32 = 5, X: i32 = 9>
      (vec: %vec2 : !hw.array<5 x !hw.int<9>>, a: %a2 : !hw.int<9>) -> (out: !hw.int<9>)
    hw.output %0, %1 : i8, i9
  }
}

// -----

// Test hw.param.value.

module {

// CHECK-LABEL:  hw.module @constantGen_V_8() -> (out: i64) {
// CHECK:          %c8_i64 = hw.constant 8 : i64
// CHECK:          hw.output %c8_i64 : i64
// CHECK:        }
// CHECK-LABEL:  hw.module @constantGen_V_9() -> (out: i64) {
// CHECK:          %c9_i64 = hw.constant 9 : i64
// CHECK:          hw.output %c9_i64 : i64
// CHECK:        }
  hw.module @constantGen<V: i64>() -> (out: i64) {
    %0 = hw.param.value i64 = #hw.param.decl.ref<"V">
    hw.output %0 :i64
  }

// CHECK-LABEL:  hw.module @top() -> (out1: i64, out2: i64) {
// CHECK:          %inst1.out = hw.instance "inst1" @constantGen_V_8() -> (out: i64)
// CHECK:          %inst2.out = hw.instance "inst2" @constantGen_V_9() -> (out: i64)
// CHECK:          hw.output %inst1.out, %inst2.out : i64, i64
// CHECK:        }
  hw.module @top() -> (out1: i64, out2: i64) {
    %0 = hw.instance "inst1" @constantGen<V: i64 = 8> () -> (out: i64)
    %1 = hw.instance "inst2" @constantGen<V: i64 = 9> () -> (out: i64)
    hw.output %0, %1 : i64, i64
  }
}

// -----

// Test two identical instances of the same module.

module {

  hw.module @constantGen<V: i64>() -> (out: i64) {
    %0 = hw.param.value i64 = #hw.param.decl.ref<"V">
    hw.output %0 :i64
  }

// CHECK-LABEL:  hw.module @top() -> (out1: i64, out2: i64) {
// CHECK:          %inst1.out = hw.instance "inst1" @constantGen_V_8() -> (out: i64)
// CHECK:          %inst2.out = hw.instance "inst2" @constantGen_V_8() -> (out: i64)
// CHECK:          hw.output %inst1.out, %inst2.out : i64, i64
// CHECK:        }
  hw.module @top() -> (out1: i64, out2: i64) {
    %0 = hw.instance "inst1" @constantGen<V: i64 = 8> () -> (out: i64)
    %1 = hw.instance "inst2" @constantGen<V: i64 = 8> () -> (out: i64)
    hw.output %0, %1 : i64, i64
  }
}
