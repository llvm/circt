firrtl.circuit "InterModuleMultipleBar" {
  // Inter-module width inference for multiple instances per module.
  // CHECK-LABEL: @InterModuleMultipleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  // CHECK-LABEL: @InterModuleMultipleBar
  // CHECK-SAME: in %in1: !firrtl.uint<17>
  // CHECK-SAME: in %in2: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  firrtl.module @InterModuleMultipleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.add %in, %in : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  firrtl.module @InterModuleMultipleBar(in %in1: !firrtl.uint<17>, in %in2: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst1_in, %inst1_out = firrtl.instance inst1 @InterModuleMultipleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %inst2_in, %inst2_out = firrtl.instance inst2 @InterModuleMultipleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %0 = firrtl.xor %inst1_out, %inst2_out : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %inst1_in, %in1 : !firrtl.uint, !firrtl.uint<17>
    firrtl.connect %inst2_in, %in2 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
}