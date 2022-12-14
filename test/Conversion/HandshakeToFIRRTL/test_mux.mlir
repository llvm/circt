// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK:           firrtl.module @handshake_mux_in_ui64_ui64_ui64_out_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_6:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_9:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_12:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_13:.*]] = firrtl.subfield %[[VAL_3]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_14:.*]] = firrtl.subfield %[[VAL_3]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_15:.*]] = firrtl.subfield %[[VAL_3]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_16:.*]] = firrtl.bits %[[VAL_6]] 0 to 0 : (!firrtl.uint<64>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_17:.*]] = firrtl.mux(%[[VAL_16]], %[[VAL_12]], %[[VAL_9]]) : (!firrtl.uint<1>, !firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:             firrtl.connect %[[VAL_15]], %[[VAL_17]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:             %[[VAL_18:.*]] = firrtl.bits %[[VAL_6]] 0 to 0 : (!firrtl.uint<64>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_19:.*]] = firrtl.mux(%[[VAL_18]], %[[VAL_10]], %[[VAL_7]]) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_20:.*]] = firrtl.and %[[VAL_19]], %[[VAL_4]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_13]], %[[VAL_20]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_21:.*]] = firrtl.and %[[VAL_20]], %[[VAL_14]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_5]], %[[VAL_21]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_22:.*]] = firrtl.tail %[[VAL_6]], 63 : (!firrtl.uint<64>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_23:.*]] = firrtl.constant 1 : !firrtl.uint<1>
// CHECK:             %[[VAL_24:.*]] = firrtl.dshl %[[VAL_23]], %[[VAL_22]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
// CHECK:             %[[VAL_25:.*]] = firrtl.bits %[[VAL_24]] 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_26:.*]] = firrtl.and %[[VAL_25]], %[[VAL_21]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_8]], %[[VAL_26]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_27:.*]] = firrtl.bits %[[VAL_24]] 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_28:.*]] = firrtl.and %[[VAL_27]], %[[VAL_21]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_11]], %[[VAL_28]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }

// CHECK: firrtl.module @test_mux(in %[[VAL_29:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_30:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_31:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_32:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_33:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_34:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_35:.*]]: !firrtl.clock, in %[[VAL_36:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_37:.*]], %[[VAL_38:.*]], %[[VAL_39:.*]], %[[VAL_40:.*]] = firrtl.instance handshake_mux0  @handshake_mux_in_ui64_ui64_ui64_out_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG3:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @test_mux(%arg0: index, %arg1: index, %arg2: index, %arg3: none, ...) -> (index, none) {
  %0 = mux %arg0 [%arg1, %arg2] : index, index
  return %0, %arg3 : index, none
}

// -----

// Test a mux tree with an odd number of inputs.

// CHECK:           firrtl.module @handshake_mux_in_ui64_ui64_ui64_ui64_out_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK: %[[DATA1:.+]] = firrtl.subfield %[[VAL_1]][data]
// CHECK: %[[DATA2:.+]] = firrtl.subfield %[[VAL_2]][data]
// CHECK: %[[DATA3:.+]] = firrtl.subfield %[[VAL_3]][data]
// CHECK: %[[RESULT:.+]] = firrtl.subfield %[[VAL_4]][data]
// CHECK: %[[MUX1:.+]] = firrtl.mux({{.+}}, %[[DATA2]], %[[DATA1]])
// CHECK: %[[MUX2:.+]] = firrtl.mux({{.+}}, %[[DATA3]], %[[MUX1]])
// CHECK: firrtl.connect %[[RESULT]], %[[MUX2]]
handshake.func @test_mux_3way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: none, ...) -> (index, none) {
  %0 = mux %arg0 [%arg1, %arg2, %arg3] : index, index
  return %0, %arg4 : index, none
}

// -----

// Test a mux tree with multiple full layers.

// CHECK:           firrtl.module @handshake_mux_in_ui64_ui64_ui64_ui64_ui64_ui64_ui64_ui64_ui64_out_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_5:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_6:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_7:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_8:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_9:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK: %[[DATA1:.+]] = firrtl.subfield %[[VAL_1]][data]
// CHECK: %[[DATA2:.+]] = firrtl.subfield %[[VAL_2]][data]
// CHECK: %[[DATA3:.+]] = firrtl.subfield %[[VAL_3]][data]
// CHECK: %[[DATA4:.+]] = firrtl.subfield %[[VAL_4]][data]
// CHECK: %[[DATA5:.+]] = firrtl.subfield %[[VAL_5]][data]
// CHECK: %[[DATA6:.+]] = firrtl.subfield %[[VAL_6]][data]
// CHECK: %[[DATA7:.+]] = firrtl.subfield %[[VAL_7]][data]
// CHECK: %[[DATA8:.+]] = firrtl.subfield %[[VAL_8]][data]
// CHECK: %[[RESULT:.+]] = firrtl.subfield %[[VAL_9]][data]
// CHECK: %[[MUX1:.+]] = firrtl.mux({{.+}}, %[[DATA2]], %[[DATA1]])
// CHECK: %[[MUX2:.+]] = firrtl.mux({{.+}}, %[[DATA4]], %[[DATA3]])
// CHECK: %[[MUX5:.+]] = firrtl.mux({{.+}}, %[[MUX2]], %[[MUX1]])
// CHECK: %[[MUX3:.+]] = firrtl.mux({{.+}}, %[[DATA6]], %[[DATA5]])
// CHECK: %[[MUX4:.+]] = firrtl.mux({{.+}}, %[[DATA8]], %[[DATA7]])
// CHECK: %[[MUX6:.+]] = firrtl.mux({{.+}}, %[[MUX4]], %[[MUX3]])
// CHECK: %[[MUX7:.+]] = firrtl.mux({{.+}}, %[[MUX6]], %[[MUX5]])
// CHECK: firrtl.connect %[[RESULT]], %[[MUX7]]
handshake.func @test_mux_8way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: none, ...) -> (index, none) {
  %0 = mux %arg0 [%arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8] : index, index
  return %0, %arg9 : index, none
}

// -----

// Test a mux tree with multiple layers and a partial first layer (odd).

// CHECK:           firrtl.module @handshake_mux_in_ui64_ui64_ui64_ui64_ui64_ui64_out_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_5:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_6:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK: %[[DATA1:.+]] = firrtl.subfield %[[VAL_1]][data]
// CHECK: %[[DATA2:.+]] = firrtl.subfield %[[VAL_2]][data]
// CHECK: %[[DATA3:.+]] = firrtl.subfield %[[VAL_3]][data]
// CHECK: %[[DATA4:.+]] = firrtl.subfield %[[VAL_4]][data]
// CHECK: %[[DATA5:.+]] = firrtl.subfield %[[VAL_5]][data]
// CHECK: %[[RESULT:.+]] = firrtl.subfield %[[VAL_6]][data]
// CHECK: %[[MUX1:.+]] = firrtl.mux({{.+}}, %[[DATA2]], %[[DATA1]])
// CHECK: %[[MUX2:.+]] = firrtl.mux({{.+}}, %[[DATA4]], %[[DATA3]])
// CHECK: %[[MUX3:.+]] = firrtl.mux({{.+}}, %[[MUX2]], %[[MUX1]])
// CHECK: %[[MUX4:.+]] = firrtl.mux({{.+}}, %[[DATA5]], %[[MUX3]])
// CHECK: firrtl.connect %[[RESULT]], %[[MUX4]]
handshake.func @test_mux_5way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: none, ...) -> (index, none) {
  %0 = mux %arg0 [%arg1, %arg2, %arg3, %arg4, %arg5] : index, index
  return %0, %arg6 : index, none
}


// -----

// Test a mux tree with multiple layers and a partial first layer (even).

// CHECK:           firrtl.module @handshake_mux_in_ui64_ui64_ui64_ui64_ui64_ui64_ui64_out_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_5:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_6:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_7:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK: %[[DATA1:.+]] = firrtl.subfield %[[VAL_1]][data]
// CHECK: %[[DATA2:.+]] = firrtl.subfield %[[VAL_2]][data]
// CHECK: %[[DATA3:.+]] = firrtl.subfield %[[VAL_3]][data]
// CHECK: %[[DATA4:.+]] = firrtl.subfield %[[VAL_4]][data]
// CHECK: %[[DATA5:.+]] = firrtl.subfield %[[VAL_5]][data]
// CHECK: %[[DATA6:.+]] = firrtl.subfield %[[VAL_6]][data]
// CHECK: %[[RESULT:.+]] = firrtl.subfield %[[VAL_7]][data]
// CHECK: %[[MUX1:.+]] = firrtl.mux({{.+}}, %[[DATA2]], %[[DATA1]])
// CHECK: %[[MUX2:.+]] = firrtl.mux({{.+}}, %[[DATA4]], %[[DATA3]])
// CHECK: %[[MUX3:.+]] = firrtl.mux({{.+}}, %[[MUX2]], %[[MUX1]])
// CHECK: %[[MUX4:.+]] = firrtl.mux({{.+}}, %[[DATA6]], %[[DATA5]])
// CHECK: %[[MUX5:.+]] = firrtl.mux({{.+}}, %[[MUX4]], %[[MUX3]])
// CHECK: firrtl.connect %[[RESULT]], %[[MUX5]]
handshake.func @test_mux_6way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: none, ...) -> (index, none) {
  %0 = mux %arg0 [%arg1, %arg2, %arg3, %arg4, %arg5, %arg6] : index, index
  return %0, %arg7 : index, none
}
