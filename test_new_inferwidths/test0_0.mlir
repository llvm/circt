firrtl.circuit "InferSpecialConstant" {
  firrtl.module @InferSpecialConstant() {
    // CHECK: %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
    %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
  }
}