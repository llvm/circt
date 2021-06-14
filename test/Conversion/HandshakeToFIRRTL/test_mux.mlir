// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_mux_3ins_1outs_ui64(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, in %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, in %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, out %arg3: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %2 = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %3 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %5 = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %6 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %7 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %9 = firrtl.subfield %arg3("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %10 = firrtl.subfield %arg3("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %11 = firrtl.subfield %arg3("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %12 = firrtl.bits %2 0 to 0 : (!firrtl.uint<64>) -> !firrtl.uint<1>
// CHECK:   %13 = firrtl.mux(%12, %8, %5) : (!firrtl.uint<1>, !firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %11, %13 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   %14 = firrtl.bits %2 0 to 0 : (!firrtl.uint<64>) -> !firrtl.uint<1>
// CHECK:   %15 = firrtl.mux(%14, %6, %3) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %16 = firrtl.and %15, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %9, %16 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %17 = firrtl.and %16, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %17 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %18 = firrtl.tail %2, 63 : (!firrtl.uint<64>) -> !firrtl.uint<1>
// CHECK:   %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
// CHECK:   %19 = firrtl.dshl %c1_ui1, %18 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
// CHECK:   %20 = firrtl.bits %19 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
// CHECK:   %21 = firrtl.and %20, %17 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %4, %21 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %22 = firrtl.bits %19 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
// CHECK:   %23 = firrtl.and %22, %17 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %7, %23 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }

// CHECK: firrtl.module @test_mux(in %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, in %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, in %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, in %arg3: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, out %arg4: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, out %arg5: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_mux(%arg0: index, %arg1: index, %arg2: index, %arg3: none, ...) -> (index, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_arg2, %inst_arg3 = firrtl.instance @handshake_mux_3ins_1outs_ui64  {name = ""} : !firrtl.flip<bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>, !firrtl.flip<bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>, !firrtl.flip<bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  %0 = "handshake.mux"(%arg0, %arg1, %arg2): (index, index, index) -> index
  handshake.return %0, %arg3 : index, none
}

// -----

// Test a mux tree with an odd number of inputs.

// CHECK-LABEL: firrtl.module @handshake_mux_4ins_1outs_ui64
// CHECK: %[[DATA1:.+]] = firrtl.subfield %arg1("data")
// CHECK: %[[DATA2:.+]] = firrtl.subfield %arg2("data")
// CHECK: %[[DATA3:.+]] = firrtl.subfield %arg3("data")
// CHECK: %[[RESULT:.+]] = firrtl.subfield %arg4("data")
// CHECK: %[[MUX1:.+]] = firrtl.mux({{.+}}, %[[DATA2]], %[[DATA1]])
// CHECK: %[[MUX2:.+]] = firrtl.mux({{.+}}, %[[DATA3]], %[[MUX1]])
// CHECK: firrtl.connect %[[RESULT]], %[[MUX2]]
handshake.func @test_mux_3way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: none, ...) -> (index, none) {
  %0 = "handshake.mux"(%arg0, %arg1, %arg2, %arg3): (index, index, index, index) -> index
  handshake.return %0, %arg4 : index, none
}

// -----

// Test a mux tree with multiple full layers.

// CHECK-LABEL: firrtl.module @handshake_mux_9ins_1outs_ui64
// CHECK: %[[DATA1:.+]] = firrtl.subfield %arg1("data")
// CHECK: %[[DATA2:.+]] = firrtl.subfield %arg2("data")
// CHECK: %[[DATA3:.+]] = firrtl.subfield %arg3("data")
// CHECK: %[[DATA4:.+]] = firrtl.subfield %arg4("data")
// CHECK: %[[DATA5:.+]] = firrtl.subfield %arg5("data")
// CHECK: %[[DATA6:.+]] = firrtl.subfield %arg6("data")
// CHECK: %[[DATA7:.+]] = firrtl.subfield %arg7("data")
// CHECK: %[[DATA8:.+]] = firrtl.subfield %arg8("data")
// CHECK: %[[RESULT:.+]] = firrtl.subfield %arg9("data")
// CHECK: %[[MUX1:.+]] = firrtl.mux({{.+}}, %[[DATA2]], %[[DATA1]])
// CHECK: %[[MUX2:.+]] = firrtl.mux({{.+}}, %[[DATA4]], %[[DATA3]])
// CHECK: %[[MUX3:.+]] = firrtl.mux({{.+}}, %[[DATA6]], %[[DATA5]])
// CHECK: %[[MUX4:.+]] = firrtl.mux({{.+}}, %[[DATA8]], %[[DATA7]])
// CHECK: %[[MUX5:.+]] = firrtl.mux({{.+}}, %[[MUX2]], %[[MUX1]])
// CHECK: %[[MUX6:.+]] = firrtl.mux({{.+}}, %[[MUX4]], %[[MUX3]])
// CHECK: %[[MUX7:.+]] = firrtl.mux({{.+}}, %[[MUX6]], %[[MUX5]])
// CHECK: firrtl.connect %[[RESULT]], %[[MUX7]]
handshake.func @test_mux_8way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: none, ...) -> (index, none) {
  %0 = "handshake.mux"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8): (index, index, index, index, index, index, index, index, index) -> index
  handshake.return %0, %arg9 : index, none
}

// -----

// Test a mux tree with multiple layers and a partial first layer (odd).

// CHECK-LABEL: firrtl.module @handshake_mux_6ins_1outs_ui64
// CHECK: %[[DATA1:.+]] = firrtl.subfield %arg1("data")
// CHECK: %[[DATA2:.+]] = firrtl.subfield %arg2("data")
// CHECK: %[[DATA3:.+]] = firrtl.subfield %arg3("data")
// CHECK: %[[DATA4:.+]] = firrtl.subfield %arg4("data")
// CHECK: %[[DATA5:.+]] = firrtl.subfield %arg5("data")
// CHECK: %[[RESULT:.+]] = firrtl.subfield %arg6("data")
// CHECK: %[[MUX1:.+]] = firrtl.mux({{.+}}, %[[DATA2]], %[[DATA1]])
// CHECK: %[[MUX2:.+]] = firrtl.mux({{.+}}, %[[DATA4]], %[[DATA3]])
// CHECK: %[[MUX3:.+]] = firrtl.mux({{.+}}, %[[MUX2]], %[[MUX1]])
// CHECK: %[[MUX4:.+]] = firrtl.mux({{.+}}, %[[DATA5]], %[[MUX3]])
// CHECK: firrtl.connect %[[RESULT]], %[[MUX4]]
handshake.func @test_mux_5way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: none, ...) -> (index, none) {
  %0 = "handshake.mux"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5): (index, index, index, index, index, index) -> index
  handshake.return %0, %arg6 : index, none
}

// -----

// Test a mux tree with multiple layers and a partial first layer (even).

// CHECK-LABEL: firrtl.module @handshake_mux_7ins_1outs_ui64
// CHECK: %[[DATA1:.+]] = firrtl.subfield %arg1("data")
// CHECK: %[[DATA2:.+]] = firrtl.subfield %arg2("data")
// CHECK: %[[DATA3:.+]] = firrtl.subfield %arg3("data")
// CHECK: %[[DATA4:.+]] = firrtl.subfield %arg4("data")
// CHECK: %[[DATA5:.+]] = firrtl.subfield %arg5("data")
// CHECK: %[[DATA6:.+]] = firrtl.subfield %arg6("data")
// CHECK: %[[RESULT:.+]] = firrtl.subfield %arg7("data")
// CHECK: %[[MUX1:.+]] = firrtl.mux({{.+}}, %[[DATA2]], %[[DATA1]])
// CHECK: %[[MUX2:.+]] = firrtl.mux({{.+}}, %[[DATA4]], %[[DATA3]])
// CHECK: %[[MUX3:.+]] = firrtl.mux({{.+}}, %[[DATA6]], %[[DATA5]])
// CHECK: %[[MUX4:.+]] = firrtl.mux({{.+}}, %[[MUX2]], %[[MUX1]])
// CHECK: %[[MUX5:.+]] = firrtl.mux({{.+}}, %[[MUX3]], %[[MUX4]])
// CHECK: firrtl.connect %[[RESULT]], %[[MUX5]]
handshake.func @test_mux_6way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: none, ...) -> (index, none) {
  %0 = "handshake.mux"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6): (index, index, index, index, index, index, index) -> index
  handshake.return %0, %arg7 : index, none
}
