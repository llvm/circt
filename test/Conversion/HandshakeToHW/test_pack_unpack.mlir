// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_pack_in_ui64_ui32_out_tuple_ui64_ui32(
// CHECK-SAME:                                                               %[[VAL_0:.*]]: !esi.channel<i64>,
// CHECK-SAME:                                                               %[[VAL_1:.*]]: !esi.channel<i32>) -> (out0: !esi.channel<!hw.struct<field0: i64, field1: i32>>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i64
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_9:.*]], %[[VAL_10:.*]] : !hw.struct<field0: i64, field1: i32>
// CHECK:           %[[VAL_10]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_8]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_9]] = hw.struct_create (%[[VAL_2]], %[[VAL_5]]) : !hw.struct<field0: i64, field1: i32>
// CHECK:           hw.output %[[VAL_7]] : !esi.channel<!hw.struct<field0: i64, field1: i32>>
// CHECK:         }

handshake.func @test_pack(%arg0: i64, %arg1: i32, %ctrl: none, ...) -> (tuple<i64, i32>, none) {
  %0 = pack %arg0, %arg1 : tuple<i64, i32>
  return %0, %ctrl : tuple<i64, i32>, none
}

// -----

// CHECK-LABEL:   hw.module @handshake_unpack_in_tuple_ui64_ui32_out_ui64_ui32(
// CHECK-SAME:                                                                 %[[VAL_0:.*]]: !esi.channel<!hw.struct<field0: i64, field1: i32>>,
// CHECK-SAME:                                                                 %[[VAL_1:.*]]: i1,
// CHECK-SAME:                                                                 %[[VAL_2:.*]]: i1) -> (out0: !esi.channel<i64>, out1: !esi.channel<i32>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : !hw.struct<field0: i64, field1: i32>
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.wrap.vr %[[VAL_8:.*]]#0, %[[VAL_9:.*]] : i64
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.wrap.vr %[[VAL_8]]#1, %[[VAL_12:.*]] : i32
// CHECK:           %[[VAL_13:.*]] = hw.constant false
// CHECK:           %[[VAL_14:.*]] = hw.constant true
// CHECK:           %[[VAL_15:.*]] = comb.xor %[[VAL_5]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_16:.*]] = comb.and %[[VAL_17:.*]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_18:.*]] = seq.compreg %[[VAL_16]], %[[VAL_1]], %[[VAL_2]], %[[VAL_13]]  : i1
// CHECK:           %[[VAL_19:.*]] = hw.constant true
// CHECK:           %[[VAL_20:.*]] = comb.xor %[[VAL_18]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_9]] = comb.and %[[VAL_20]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.and %[[VAL_7]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_17]] = comb.and %[[VAL_21]], %[[VAL_18]] : i1
// CHECK:           %[[VAL_22:.*]] = hw.constant true
// CHECK:           %[[VAL_23:.*]] = comb.xor %[[VAL_5]], %[[VAL_22]] : i1
// CHECK:           %[[VAL_24:.*]] = comb.and %[[VAL_25:.*]], %[[VAL_23]] : i1
// CHECK:           %[[VAL_26:.*]] = seq.compreg %[[VAL_24]], %[[VAL_1]], %[[VAL_2]], %[[VAL_13]]  : i1
// CHECK:           %[[VAL_27:.*]] = hw.constant true
// CHECK:           %[[VAL_28:.*]] = comb.xor %[[VAL_26]], %[[VAL_27]] : i1
// CHECK:           %[[VAL_12]] = comb.and %[[VAL_28]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_29:.*]] = comb.and %[[VAL_11]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_25]] = comb.and %[[VAL_29]], %[[VAL_26]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_17]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_8]]:2 = hw.struct_explode %[[VAL_3]] : !hw.struct<field0: i64, field1: i32>
// CHECK:           hw.output %[[VAL_6]], %[[VAL_10]] : !esi.channel<i64>, !esi.channel<i32>
// CHECK:         }

handshake.func @test_unpack(%arg0: tuple<i64, i32>, %ctrl: none, ...) -> (i64, i32, none) {
  %0, %1 = unpack %arg0 : tuple<i64, i32>
  return %0, %1, %ctrl : i64, i32, none
}
