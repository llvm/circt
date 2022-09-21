// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// Test a control merge that is control only.

// CHECK-LABEL:   hw.module @handshake_control_merge_out_ui64_2ins_2outs_ctrl(
// CHECK-SAME:              %[[VAL_0:.*]]: !esi.channel<none>, %[[VAL_1:.*]]: !esi.channel<none>, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (dataOut: !esi.channel<none>, index: !esi.channel<i64>) {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_6:.*]] : none
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_9:.*]] : none
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.wrap.vr %[[VAL_12:.*]], %[[VAL_13:.*]] : none
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = esi.wrap.vr %[[VAL_16:.*]], %[[VAL_17:.*]] : i64
// CHECK:           %[[VAL_18:.*]] = hw.constant 0 : i2
// CHECK:           %[[VAL_19:.*]] = hw.constant false
// CHECK:           %[[VAL_20:.*]] = seq.compreg %[[VAL_21:.*]], %[[VAL_2]], %[[VAL_3]], %[[VAL_18]]  : i2
// CHECK:           %[[VAL_22:.*]] = seq.compreg %[[VAL_23:.*]], %[[VAL_2]], %[[VAL_3]], %[[VAL_19]]  : i1
// CHECK:           %[[VAL_24:.*]] = seq.compreg %[[VAL_25:.*]], %[[VAL_2]], %[[VAL_3]], %[[VAL_19]]  : i1
// CHECK:           %[[VAL_26:.*]] = comb.extract %[[VAL_27:.*]] from 0 : (i2) -> i1
// CHECK:           %[[VAL_28:.*]] = comb.extract %[[VAL_27]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_29:.*]] = comb.or %[[VAL_26]], %[[VAL_28]] : i1
// CHECK:           %[[VAL_30:.*]] = comb.extract %[[VAL_20]] from 0 : (i2) -> i1
// CHECK:           %[[VAL_31:.*]] = comb.extract %[[VAL_20]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_32:.*]] = comb.or %[[VAL_30]], %[[VAL_31]] : i1
// CHECK:           %[[VAL_33:.*]] = hw.constant -2 : i2
// CHECK:           %[[VAL_34:.*]] = comb.mux %[[VAL_8]], %[[VAL_18]], %[[VAL_33]] : i2
// CHECK:           %[[VAL_35:.*]] = hw.constant 1 : i2
// CHECK:           %[[VAL_36:.*]] = comb.mux %[[VAL_5]], %[[VAL_34]], %[[VAL_35]] : i2
// CHECK:           %[[VAL_27]] = comb.mux %[[VAL_32]], %[[VAL_20]], %[[VAL_36]] : i2
// CHECK:           %[[VAL_37:.*]] = hw.constant true
// CHECK:           %[[VAL_38:.*]] = comb.xor %[[VAL_22]], %[[VAL_37]] : i1
// CHECK:           %[[VAL_13]] = comb.and %[[VAL_29]], %[[VAL_38]] : i1
// CHECK:           %[[VAL_12]] = esi.none : none
// CHECK:           %[[VAL_39:.*]] = comb.xor %[[VAL_24]], %[[VAL_37]] : i1
// CHECK:           %[[VAL_17]] = comb.and %[[VAL_29]], %[[VAL_39]] : i1
// CHECK:           %[[VAL_16]] = hw.constant 0 : i64
// CHECK:           %[[VAL_40:.*]] = hw.constant 1 : i64
// CHECK:           %[[VAL_21]] = comb.mux %[[VAL_41:.*]], %[[VAL_18]], %[[VAL_27]] : i2
// CHECK:           %[[VAL_42:.*]] = comb.and %[[VAL_13]], %[[VAL_11]] : i1
// CHECK:           %[[VAL_43:.*]] = comb.or %[[VAL_42]], %[[VAL_22]] : i1
// CHECK:           %[[VAL_44:.*]] = comb.and %[[VAL_17]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_45:.*]] = comb.or %[[VAL_44]], %[[VAL_24]] : i1
// CHECK:           %[[VAL_41]] = comb.and %[[VAL_43]], %[[VAL_45]] : i1
// CHECK:           %[[VAL_23]] = comb.mux %[[VAL_41]], %[[VAL_19]], %[[VAL_43]] : i1
// CHECK:           %[[VAL_25]] = comb.mux %[[VAL_41]], %[[VAL_19]], %[[VAL_45]] : i1
// CHECK:           %[[VAL_46:.*]] = comb.mux %[[VAL_41]], %[[VAL_27]], %[[VAL_18]] : i2
// CHECK:           %[[VAL_6]] = comb.icmp eq %[[VAL_46]], %[[VAL_35]] : i2
// CHECK:           %[[VAL_9]] = comb.icmp eq %[[VAL_46]], %[[VAL_33]] : i2
// CHECK:           hw.output %[[VAL_10]], %[[VAL_14]] : !esi.channel<none>, !esi.channel<i64>
// CHECK:         }

handshake.func @test_cmerge(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, index, none) {
  %0:2 = control_merge %arg0, %arg1 : none
  return %0#0, %0#1, %arg2 : none, index, none
}

// -----

// Test a control merge that also outputs the selected input's data.

// CHECK-LABEL:   hw.module @handshake_control_merge_in_ui64_ui64_ui64_out_ui64_ui64(
// CHECK-SAME:                 %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: !esi.channel<i64>, %[[VAL_2:.*]]: !esi.channel<i64>, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (dataOut: !esi.channel<i64>, index: !esi.channel<i64>) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_7:.*]] : i64
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_10:.*]] : i64
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_13:.*]] : i64
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = esi.wrap.vr %[[VAL_16:.*]], %[[VAL_17:.*]] : i64
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = esi.wrap.vr %[[VAL_16]], %[[VAL_20:.*]] : i64
// CHECK:           %[[VAL_21:.*]] = hw.constant 0 : i3
// CHECK:           %[[VAL_22:.*]] = hw.constant false
// CHECK:           %[[VAL_23:.*]] = seq.compreg %[[VAL_24:.*]], %[[VAL_3]], %[[VAL_4]], %[[VAL_21]]  : i3
// CHECK:           %[[VAL_25:.*]] = seq.compreg %[[VAL_26:.*]], %[[VAL_3]], %[[VAL_4]], %[[VAL_22]]  : i1
// CHECK:           %[[VAL_27:.*]] = seq.compreg %[[VAL_28:.*]], %[[VAL_3]], %[[VAL_4]], %[[VAL_22]]  : i1
// CHECK:           %[[VAL_29:.*]] = comb.extract %[[VAL_30:.*]] from 0 : (i3) -> i1
// CHECK:           %[[VAL_31:.*]] = comb.extract %[[VAL_30]] from 1 : (i3) -> i1
// CHECK:           %[[VAL_32:.*]] = comb.extract %[[VAL_30]] from 2 : (i3) -> i1
// CHECK:           %[[VAL_33:.*]] = comb.or %[[VAL_29]], %[[VAL_31]], %[[VAL_32]] : i1
// CHECK:           %[[VAL_34:.*]] = comb.extract %[[VAL_23]] from 0 : (i3) -> i1
// CHECK:           %[[VAL_35:.*]] = comb.extract %[[VAL_23]] from 1 : (i3) -> i1
// CHECK:           %[[VAL_36:.*]] = comb.extract %[[VAL_23]] from 2 : (i3) -> i1
// CHECK:           %[[VAL_37:.*]] = comb.or %[[VAL_34]], %[[VAL_35]], %[[VAL_36]] : i1
// CHECK:           %[[VAL_38:.*]] = hw.constant -4 : i3
// CHECK:           %[[VAL_39:.*]] = comb.mux %[[VAL_12]], %[[VAL_21]], %[[VAL_38]] : i3
// CHECK:           %[[VAL_40:.*]] = hw.constant 2 : i3
// CHECK:           %[[VAL_41:.*]] = comb.mux %[[VAL_9]], %[[VAL_39]], %[[VAL_40]] : i3
// CHECK:           %[[VAL_42:.*]] = hw.constant 1 : i3
// CHECK:           %[[VAL_43:.*]] = comb.mux %[[VAL_6]], %[[VAL_41]], %[[VAL_42]] : i3
// CHECK:           %[[VAL_30]] = comb.mux %[[VAL_37]], %[[VAL_23]], %[[VAL_43]] : i3
// CHECK:           %[[VAL_44:.*]] = hw.constant true
// CHECK:           %[[VAL_45:.*]] = comb.xor %[[VAL_25]], %[[VAL_44]] : i1
// CHECK:           %[[VAL_17]] = comb.and %[[VAL_33]], %[[VAL_45]] : i1
// CHECK:           %[[VAL_16]] = hw.constant 0 : i64
// CHECK:           %[[VAL_46:.*]] = comb.xor %[[VAL_27]], %[[VAL_44]] : i1
// CHECK:           %[[VAL_20]] = comb.and %[[VAL_33]], %[[VAL_46]] : i1
// CHECK:           %[[VAL_47:.*]] = hw.constant 1 : i64
// CHECK:           %[[VAL_48:.*]] = hw.constant 2 : i64
// CHECK:           %[[VAL_24]] = comb.mux %[[VAL_49:.*]], %[[VAL_21]], %[[VAL_30]] : i3
// CHECK:           %[[VAL_50:.*]] = comb.and %[[VAL_17]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_51:.*]] = comb.or %[[VAL_50]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_52:.*]] = comb.and %[[VAL_20]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_53:.*]] = comb.or %[[VAL_52]], %[[VAL_27]] : i1
// CHECK:           %[[VAL_49]] = comb.and %[[VAL_51]], %[[VAL_53]] : i1
// CHECK:           %[[VAL_26]] = comb.mux %[[VAL_49]], %[[VAL_22]], %[[VAL_51]] : i1
// CHECK:           %[[VAL_28]] = comb.mux %[[VAL_49]], %[[VAL_22]], %[[VAL_53]] : i1
// CHECK:           %[[VAL_54:.*]] = comb.mux %[[VAL_49]], %[[VAL_30]], %[[VAL_21]] : i3
// CHECK:           %[[VAL_7]] = comb.icmp eq %[[VAL_54]], %[[VAL_42]] : i3
// CHECK:           %[[VAL_10]] = comb.icmp eq %[[VAL_54]], %[[VAL_40]] : i3
// CHECK:           %[[VAL_13]] = comb.icmp eq %[[VAL_54]], %[[VAL_38]] : i3
// CHECK:           hw.output %[[VAL_14]], %[[VAL_18]] : !esi.channel<i64>, !esi.channel<i64>
// CHECK:         }

handshake.func @test_cmerge_data(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  %0:2 = control_merge %arg0, %arg1, %arg2 : index
  return %0#0, %0#1 : index, index
}
