// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK:   hw.module @handshake_sync_in_none_ui32_out_none_ui32(in %[[VAL_0:.*]] : !esi.channel<i0>, in %[[VAL_1:.*]] : !esi.channel<i32>, in %[[CLOCK:.*]] : !seq.clock, in %[[VAL_3:.*]] : i1, out out0 : !esi.channel<i0>, out out1 : !esi.channel<i32>) {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_6:.*]] : i0
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.wrap.vr %[[VAL_4]], %[[VAL_11:.*]] : i0
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = esi.wrap.vr %[[VAL_7]], %[[VAL_14:.*]] : i32
// CHECK:           %[[VAL_15:.*]] = comb.and %[[VAL_5]], %[[VAL_8]] : i1
// CHECK:           %[[VAL_6]] = comb.and %[[VAL_16:.*]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_17:.*]] = hw.constant false
// CHECK:           %[[VAL_18:.*]] = hw.constant true
// CHECK:           %[[VAL_19:.*]] = comb.xor %[[VAL_16]], %[[VAL_18]] : i1
// CHECK:           %[[VAL_20:.*]] = comb.and %[[VAL_21:.*]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_22:.*]] = seq.compreg sym @emitted_0 %[[VAL_20]], %[[CLOCK]] reset %[[VAL_3]], %[[VAL_17]]  : i1
// CHECK:           %[[VAL_23:.*]] = comb.xor %[[VAL_22]], %[[VAL_18]] : i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_23]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_24:.*]] = comb.and %[[VAL_10]], %[[VAL_11]] : i1
// CHECK:           %[[VAL_21]] = comb.or %[[VAL_24]], %[[VAL_22]] {sv.namehint = "done0"} : i1
// CHECK:           %[[VAL_25:.*]] = comb.xor %[[VAL_16]], %[[VAL_18]] : i1
// CHECK:           %[[VAL_26:.*]] = comb.and %[[VAL_27:.*]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_28:.*]] = seq.compreg sym @emitted_1 %[[VAL_26]], %[[CLOCK]] reset %[[VAL_3]], %[[VAL_17]]  : i1
// CHECK:           %[[VAL_29:.*]] = comb.xor %[[VAL_28]], %[[VAL_18]] : i1
// CHECK:           %[[VAL_14]] = comb.and %[[VAL_29]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_30:.*]] = comb.and %[[VAL_13]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_27]] = comb.or %[[VAL_30]], %[[VAL_28]] {sv.namehint = "done1"} : i1
// CHECK:           %[[VAL_16]] = comb.and %[[VAL_21]], %[[VAL_27]] {sv.namehint = "allDone"} : i1
// CHECK:           hw.output %[[VAL_9]], %[[VAL_12]] : !esi.channel<i0>, !esi.channel<i32>
// CHECK:         }

handshake.func @test_sync(%arg0: none, %arg1: i32) -> (none, i32) {
  %res:2 = sync %arg0, %arg1 : none, i32
  return %res#0, %res#1 : none, i32
}

// -----

// CHECK:   hw.module @handshake_sync_in_none_ui32_out_none_ui32(in %[[VAL_0:.*]] : !esi.channel<i0>, in %[[VAL_1:.*]] : !esi.channel<i32>, in %[[CLOCK:.*]] : !seq.clock, in %[[VAL_3:.*]] : i1, out out0 : !esi.channel<i0>, out out1 : !esi.channel<i32>) {
// CHECK:   hw.module @handshake_sync_in_none_none_ui32_out_none_none_ui32(in %[[VAL_0:.*]] : !esi.channel<i0>, in %[[VAL_1:.*]] : !esi.channel<i0>, in %[[VAL_2:.*]] : !esi.channel<i32>, in %[[CLOCK:.*]] : !seq.clock, in %[[VAL_3:.*]] : i1, out out0 : !esi.channel<i0>, out out1 : !esi.channel<i0>, out out2 : !esi.channel<i32>) {

handshake.func @test_sync_naming(%arg0: none, %arg1: none, %arg2: i32) -> (none, none, i32) {
  %res:2 = sync %arg0, %arg2 : none, i32
  %a, %b, %c = sync %arg1, %res#0, %res#1 : none, none, i32
  return %a, %b, %c : none, none, i32
}
