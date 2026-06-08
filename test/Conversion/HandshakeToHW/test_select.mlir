// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK:   hw.module @arith_select_in_ui1_ui32_ui32_out_ui32(in %[[VAL_0:.*]] : !esi.channel<i1>, in %[[VAL_1:.*]] : !esi.channel<i32>, in %[[VAL_2:.*]] : !esi.channel<i32>, out out0 : !esi.channel<i32>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_5]] : i32
// CHECK:            %[[VAL_8:.*]], %[[VAL_9:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.wrap.vr %[[VAL_12:.*]], %[[VAL_13:.*]] : i32
// CHECK:           %[[VAL_13]] = comb.and %[[VAL_4]], %[[VAL_7]], %[[VAL_9]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_11]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_12]] = comb.mux %[[VAL_3]], %[[VAL_6]], %[[VAL_8]] : i32
// CHECK:           hw.output %[[VAL_10]] : !esi.channel<i32>
// CHECK:         }

handshake.func @test_select(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: none, ...) -> (i32, none) {
  %0 = arith.select %arg0, %arg1, %arg2 : i32
  return %0, %arg3 : i32, none
}

// -----

// CHECK:   hw.module @arith_maxsi_in_ui32_ui32_out_ui32(in %[[VAL_0:.*]] : !esi.channel<i32>, in %[[VAL_1:.*]] : !esi.channel<i32>, out out0 : !esi.channel<i32>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i32
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_11:.*]], %[[VAL_10:.*]] : i32
// CHECK:           %[[VAL_10]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_8]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_9:.*]] = comb.icmp sge %[[VAL_2]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_11]] = comb.mux %[[VAL_9]], %[[VAL_2]], %[[VAL_5]] : i32
// CHECK:           hw.output %[[VAL_7]] : !esi.channel<i32>
// CHECK:         }

handshake.func @test_maxsi(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %0 = arith.maxsi %arg0, %arg1 : i32
  return %0, %arg2 : i32, none
}

// -----

// CHECK:   hw.module @arith_minui_in_ui64_ui64_out_ui64(in %[[VAL_0:.*]] : !esi.channel<i64>, in %[[VAL_1:.*]] : !esi.channel<i64>, out out0 : !esi.channel<i64>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i64
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_11:.*]], %[[VAL_10:.*]] : i64
// CHECK:           %[[VAL_10]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_8]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_9:.*]] = comb.icmp ule %[[VAL_2]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_11]] = comb.mux %[[VAL_9]], %[[VAL_2]], %[[VAL_5]] : i64
// CHECK:           hw.output %[[VAL_7]] : !esi.channel<i64>
// CHECK:         }

handshake.func @test_minui(%arg0: i64, %arg1: i64, %arg2: none, ...) -> (i64, none) {
  %0 = arith.minui %arg0, %arg1 : i64
  return %0, %arg2 : i64, none
}
