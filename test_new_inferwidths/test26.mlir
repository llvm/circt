firrtl.circuit "InterModuleSimpleBar" {
  // Inter-module width inference for one-to-one module-instance correspondence.
  // CHECK-LABEL: @InterModuleSimpleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  // CHECK-LABEL: @InterModuleSimpleBar
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<44>
  firrtl.module @InterModuleSimpleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.add %in, %in : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  firrtl.module @InterModuleSimpleBar(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst_in, %inst_out = firrtl.instance inst @InterModuleSimpleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %0 = firrtl.add %inst_out, %inst_out : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
}