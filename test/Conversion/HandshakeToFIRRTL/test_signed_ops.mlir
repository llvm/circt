// RUN: circt-opt -split-input-file -lower-handshake-to-firrtl %s | FileCheck %s


// CHECK:  firrtl.circuit "test_cmpi"  {
// CHECK:    firrtl.module @arith_cmpi_in_ui32_ui32_out_ui1_sgt(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>) {
// CHECK:      %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_9:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:      %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:      %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:      %[[VAL_12:.*]] = firrtl.asSInt %[[VAL_5]] : (!firrtl.uint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_13:.*]] = firrtl.asSInt %[[VAL_8]] : (!firrtl.uint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_14:.*]] = firrtl.gt %[[VAL_12]], %[[VAL_13]] : (!firrtl.sint<32>, !firrtl.sint<32>) -> !firrtl.uint<1>
// CHECK:      %[[VAL_15:.*]] = firrtl.asUInt %[[VAL_14]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_11]], %[[VAL_15]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      %[[VAL_16:.*]] = firrtl.and %[[VAL_3]], %[[VAL_6]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_9]], %[[VAL_16]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      %[[VAL_17:.*]] = firrtl.and %[[VAL_10]], %[[VAL_16]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_4]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_7]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:    }

handshake.func @test_cmpi(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i1, none) {
  %0 = arith.cmpi sgt, %arg0, %arg1 : i32
  return %0, %arg2 : i1, none
}

// -----

// CHECK:  firrtl.circuit "test_divsi"  {
// CHECK:    firrtl.module @arith_divsi_in_ui32_ui32_out_ui32(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {
// CHECK:      %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_9:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_12:.*]] = firrtl.asSInt %[[VAL_5]] : (!firrtl.uint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_13:.*]] = firrtl.asSInt %[[VAL_8]] : (!firrtl.uint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_14:.*]] = firrtl.div %[[VAL_12]], %[[VAL_13]] : (!firrtl.sint<32>, !firrtl.sint<32>) -> !firrtl.sint<33>
// CHECK:      %[[VAL_15:.*]] = firrtl.bits %[[VAL_14]] 31 to 0 : (!firrtl.sint<33>) -> !firrtl.uint<32>
// CHECK:      firrtl.connect %[[VAL_11]], %[[VAL_15]] : !firrtl.uint<32>, !firrtl.uint<32>
// CHECK:      %[[VAL_16:.*]] = firrtl.and %[[VAL_3]], %[[VAL_6]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_9]], %[[VAL_16]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      %[[VAL_17:.*]] = firrtl.and %[[VAL_10]], %[[VAL_16]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_4]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_7]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:    }

handshake.func @test_divsi(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %0 = arith.divsi %arg0, %arg1 : i32
  return %0, %arg2 : i32, none
}

// -----

// CHECK:  firrtl.circuit "test_remsi"  {
// CHECK:    firrtl.module @arith_remsi_in_ui32_ui32_out_ui32(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {
// CHECK:      %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_9:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_12:.*]] = firrtl.asSInt %[[VAL_5]] : (!firrtl.uint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_13:.*]] = firrtl.asSInt %[[VAL_8]] : (!firrtl.uint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_14:.*]] = firrtl.rem %[[VAL_12]], %[[VAL_13]] : (!firrtl.sint<32>, !firrtl.sint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_15:.*]] = firrtl.asUInt %[[VAL_14]] : (!firrtl.sint<32>) -> !firrtl.uint<32>
// CHECK:      firrtl.connect %[[VAL_11]], %[[VAL_15]] : !firrtl.uint<32>, !firrtl.uint<32>
// CHECK:      %[[VAL_16:.*]] = firrtl.and %[[VAL_3]], %[[VAL_6]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_9]], %[[VAL_16]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      %[[VAL_17:.*]] = firrtl.and %[[VAL_10]], %[[VAL_16]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_4]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_7]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:    }

handshake.func @test_remsi(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %0 = arith.remsi %arg0, %arg1 : i32
  return %0, %arg2 : i32, none
}

// -----

// CHECK:  firrtl.circuit "test_extsi"  {
// CHECK:    firrtl.module @arith_extsi_in_ui16_out_ui32(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<16>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {
// CHECK:      %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<16>>
// CHECK:      %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<16>>
// CHECK:      %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<16>>
// CHECK:      %[[VAL_6:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_7:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_8:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_9:.*]] = firrtl.asSInt %[[VAL_5]] : (!firrtl.uint<16>) -> !firrtl.sint<16>
// CHECK:      %[[VAL_10:.*]] = firrtl.pad %[[VAL_9]], 16 : (!firrtl.sint<16>) -> !firrtl.sint<16>
// CHECK:      %[[VAL_11:.*]] = firrtl.asUInt %[[VAL_10]] : (!firrtl.sint<16>) -> !firrtl.uint<16>
// CHECK:      firrtl.connect %[[VAL_8]], %[[VAL_11]] : !firrtl.uint<32>, !firrtl.uint<16>
// CHECK:      firrtl.connect %[[VAL_6]], %[[VAL_3]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      %[[VAL_12:.*]] = firrtl.and %[[VAL_7]], %[[VAL_3]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_4]], %[[VAL_12]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:    }

handshake.func @test_extsi(%arg0: i16, %arg1: none, ...) -> (i32, none) {
  %0 = arith.extsi %arg0 : i16 to i32
  return %0, %arg1 : i32, none
}

// -----

// CHECK:  firrtl.circuit "test_shrsi"  {
// CHECK:    firrtl.module @arith_shrsi_in_ui32_ui32_out_ui32(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {
// CHECK:      %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_9:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:      %[[VAL_12:.*]] = firrtl.asSInt %[[VAL_5]] : (!firrtl.uint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_14:.*]] = firrtl.dshr %[[VAL_12]], %[[VAL_8]] : (!firrtl.sint<32>, !firrtl.uint<32>) -> !firrtl.sint<32>
// CHECK:      %[[VAL_15:.*]] = firrtl.asUInt %[[VAL_14]] : (!firrtl.sint<32>) -> !firrtl.uint<32>
// CHECK:      firrtl.connect %[[VAL_11]], %[[VAL_15]] : !firrtl.uint<32>, !firrtl.uint<32>
// CHECK:      %[[VAL_16:.*]] = firrtl.and %[[VAL_3]], %[[VAL_6]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_9]], %[[VAL_16]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      %[[VAL_17:.*]] = firrtl.and %[[VAL_10]], %[[VAL_16]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_4]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:      firrtl.connect %[[VAL_7]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:    }

handshake.func @test_shrsi(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %0 = arith.shrsi %arg0, %arg1 : i32
  return %0, %arg2 : i32, none
}
