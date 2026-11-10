// RUN: circt-opt -split-input-file -lower-handshake-to-hw %s | FileCheck %s

// CHECK:   hw.module @handshake_cond_br_in_ui1_ui64_out_ui64_ui64(in %[[VAL_0:.*]] : !esi.channel<i1, ValidReady>, in %[[VAL_1:.*]] : !esi.channel<i64, ValidReady>, out outTrue : !esi.channel<i64, ValidReady>, out outFalse : !esi.channel<i64, ValidReady>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : !esi.channel<i1, ValidReady>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : !esi.channel<i64, ValidReady>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_5]], %[[VAL_9:.*]] : !esi.channel<i64, ValidReady>
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.wrap.vr %[[VAL_5]], %[[VAL_12:.*]] : !esi.channel<i64, ValidReady>
// CHECK:           %[[VAL_13:.*]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_9]] = comb.and %[[VAL_2]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_14:.*]] = hw.constant true
// CHECK:           %[[VAL_15:.*]] = comb.xor %[[VAL_2]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_12]] = comb.and %[[VAL_15]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_16:.*]] = comb.mux %[[VAL_2]], %[[VAL_8]], %[[VAL_11]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_16]], %[[VAL_13]] : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_10]] : !esi.channel<i64, ValidReady>, !esi.channel<i64, ValidReady>
// CHECK:         }

handshake.func @test_conditional_branch(%arg0: i1, %arg1: index) -> (index, index) {
  %0:2 = cond_br %arg0, %arg1 : index
  return %0#0, %0#1 : index, index
}

// -----

// CHECK:   hw.module @handshake_cond_br_in_ui1_none_out_none_none_2ins_2outs_ctrl(in %[[VAL_0:.*]] : !esi.channel<i1, ValidReady>, in %[[VAL_1:.*]] : !esi.channel<i0, ValidReady>, out outTrue : !esi.channel<i0, ValidReady>, out outFalse : !esi.channel<i0, ValidReady>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : !esi.channel<i1, ValidReady>
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : !esi.channel<i0, ValidReady>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_5]], %[[VAL_9:.*]] : !esi.channel<i0, ValidReady>
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.wrap.vr %[[VAL_5]], %[[VAL_12:.*]] : !esi.channel<i0, ValidReady>
// CHECK:           %[[VAL_13:.*]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_9]] = comb.and %[[VAL_2]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_14:.*]] = hw.constant true
// CHECK:           %[[VAL_15:.*]] = comb.xor %[[VAL_2]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_12]] = comb.and %[[VAL_15]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_16:.*]] = comb.mux %[[VAL_2]], %[[VAL_8]], %[[VAL_11]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_16]], %[[VAL_13]] : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_10]] : !esi.channel<i0, ValidReady>, !esi.channel<i0, ValidReady>
// CHECK:         }

handshake.func @test_nonetyped_conditional_branch(%arg0: i1, %arg1: none) -> (none, none) {
  %0:2 = cond_br %arg0, %arg1 : none
  return %0#0, %0#1 : none, none
}
