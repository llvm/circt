firrtl.circuit "InterModuleGoodCycleBar" {
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<39>
  firrtl.module @InterModuleGoodCycleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.shr %in, 3 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  // CHECK-LABEL: @InterModuleGoodCycleBar
  // CHECK-SAME: out %out: !firrtl.uint<39>
  firrtl.module @InterModuleGoodCycleBar(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst_in, %inst_out = firrtl.instance inst  @InterModuleGoodCycleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    firrtl.connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %inst_in, %inst_out : !firrtl.uint, !firrtl.uint
    firrtl.connect %out, %inst_out : !firrtl.uint, !firrtl.uint
  }
}