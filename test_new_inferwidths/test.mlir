firrtl.circuit "InferConstant" {
  // CHECK-LABEL: @InferConstant
  // CHECK-SAME: out %out0: !firrtl.uint<42>
  // CHECK-SAME: out %out1: !firrtl.sint<42>
  firrtl.module @InferConstant(out %out0: !firrtl.uint, out %out1: !firrtl.sint) {
    %0 = firrtl.constant 1 : !firrtl.uint<42>
    %1 = firrtl.constant 2 : !firrtl.sint<42>
    // CHECK: {{.+}} = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: {{.+}} = firrtl.constant 0 : !firrtl.sint<1>
    // CHECK: {{.+}} = firrtl.constant 200 : !firrtl.uint<8>
    // CHECK: {{.+}} = firrtl.constant 200 : !firrtl.sint<9>
    // CHECK: {{.+}} = firrtl.constant -200 : !firrtl.sint<9>
    %2 = firrtl.constant 0 : !firrtl.uint
    %3 = firrtl.constant 0 : !firrtl.sint
    %4 = firrtl.constant 200 : !firrtl.uint
    %5 = firrtl.constant 200 : !firrtl.sint
    %6 = firrtl.constant -200 : !firrtl.sint
    firrtl.connect %out0, %0 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out1, %1 : !firrtl.sint, !firrtl.sint<42>
  }
}