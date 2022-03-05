// RUN: circt-opt %s --hw-elaborate-generate | FileCheck %s

module {
// CHECK-LABEL:   hw.module @accumulate_N_3(
// CHECK-SAME:                              %[[VAL_0:.*]]: !hw.array<3xi32>) -> (out: i32) {
// CHECK:           %[[VAL_1:.*]] = sv.wire  : !hw.inout<i32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = comb.extract %[[VAL_2]] from 0 : (i64) -> i2
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = hw.array_get %[[VAL_0]]{{\[}}%[[VAL_3]]] : !hw.array<3xi32>
// CHECK:           %[[VAL_6:.*]] = hw.constant 3 : i32
// CHECK:           %[[VAL_7:.*]] = hw.constant 1 : i64
// CHECK:           %[[VAL_8:.*]] = comb.extract %[[VAL_7]] from 0 : (i64) -> i2
// CHECK:           %[[VAL_9:.*]] = hw.array_get %[[VAL_0]]{{\[}}%[[VAL_8]]] : !hw.array<3xi32>
// CHECK:           %[[VAL_10:.*]] = comb.add %[[VAL_5]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = hw.constant 2 : i64
// CHECK:           %[[VAL_12:.*]] = comb.extract %[[VAL_11]] from 0 : (i64) -> i2
// CHECK:           %[[VAL_13:.*]] = hw.array_get %[[VAL_0]]{{\[}}%[[VAL_12]]] : !hw.array<3xi32>
// CHECK:           %[[VAL_14:.*]] = comb.add %[[VAL_10]], %[[VAL_13]] : i32
// CHECK:           sv.assign %[[VAL_1]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_15:.*]] = sv.read_inout %[[VAL_1]] : !hw.inout<i32>
// CHECK:           hw.output %[[VAL_15]] : i32
// CHECK:         }
  hw.module @accumulate<N: i32>(%vec : !hw.array<#hw.param.decl.ref<"N"> x i32>) -> (out: i32) {
    %tmp = sv.wire : !hw.inout<i32>
    hw.generate {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : index
      %first = hw.array_get %vec[%c0] : !hw.array<#hw.param.decl.ref<"N"> x i32>
      %N_i32 = hw.param.value i32 = #hw.param.decl.ref<"N">
      %N_index = arith.index_cast %N_i32 : i32 to index
      %sum = scf.for %iv = %c1 to %N_index step %c1 iter_args(%sum_iter = %first) -> (i32) {
        %iv_i64 = arith.index_cast %iv : index to i64
        %v = hw.array_get %vec[%iv_i64] : !hw.array<#hw.param.decl.ref<"N"> x i32>
        %partial_sum = comb.add %sum_iter, %v : i32
        scf.yield %partial_sum : i32
      }
      sv.assign %tmp, %sum : i32
    }
    %tmp_read = sv.read_inout %tmp : !hw.inout<i32>
    hw.output %tmp_read : i32
  }

// CHECK-LABEL:   hw.module @top(
  hw.module @top(%vec : !hw.array<3 x i32>) -> (out: i32) {
    // CHECK:           %[[VAL_1:.*]] = hw.instance "inst" @accumulate_N_3(vec: %[[VAL_0]]: !hw.array<3xi32>) -> (out: i32)
    %tmp = hw.instance "inst" @accumulate<N: i32 = 3>(vec: %vec : !hw.array<3xi32>) -> (out: i32)
    hw.output %tmp : i32
  }
}