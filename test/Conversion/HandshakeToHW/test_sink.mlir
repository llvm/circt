// RUN: circt-opt -lower-handshake-to-hw %s | FileCheck %s

// CHECK:   hw.module @handshake_sink_in_ui64(in %[[VAL_0:.*]] : !esi.channel<i64>) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i64
// CHECK:           %[[VAL_3]] = hw.constant true
// CHECK:           hw.output
// CHECK:         }

handshake.func @test_sink(%arg0: index) {
  sink %arg0 : index
  return
}
