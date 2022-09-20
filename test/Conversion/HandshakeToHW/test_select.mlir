// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_select_in_ui1_ui32_ui32_out_ui32(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !esi.channel<i1>, %[[VAL_1:.*]]: !esi.channel<i32>, %[[VAL_2:.*]]: !esi.channel<i32>) -> (out0: !esi.channel<i32>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_8:.*]] : i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_11:.*]] : i32
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = esi.wrap.vr %[[VAL_14:.*]], %[[VAL_15:.*]] : i32
// CHECK:           %[[VAL_16:.*]] = hw.constant false
// CHECK:           %[[VAL_17:.*]] = comb.concat %[[VAL_16]], %[[VAL_3]] : i1, i1
// CHECK:           %[[VAL_18:.*]] = hw.constant 1 : i2
// CHECK:           %[[VAL_19:.*]] = comb.shl %[[VAL_18]], %[[VAL_17]] : i2
// CHECK:           %[[VAL_20:.*]] = comb.mux %[[VAL_3]], %[[VAL_7]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_15]] = comb.and %[[VAL_20]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_15]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.extract %[[VAL_19]] from 0 : (i2) -> i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_21]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_22:.*]] = comb.extract %[[VAL_19]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_8]] = comb.and %[[VAL_22]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_14]] = comb.mux %[[VAL_3]], %[[VAL_6]], %[[VAL_9]] : i32
// CHECK:           hw.output %[[VAL_12]] : !esi.channel<i32>
// CHECK:         }

handshake.func @test_select(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: none, ...) -> (i32, none) {
  %0 = select %arg0, %arg1, %arg2 : i32
  return %0, %arg3 : i32, none
}
