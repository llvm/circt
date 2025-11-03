// -----// IR Dump Before InferWidths (firrtl-infer-widths) //----- //
firrtl.circuit "A" {
  firrtl.module @A(in %in: !firrtl.uint<4>, in %clock: !firrtl.clock, out %out: !firrtl.uint) attributes {convention = #firrtl<convention scalarized>} {
    %x = firrtl.reg %clock : !firrtl.clock, !firrtl.uint
    %0 = firrtl.tail %x, 1 : (!firrtl.uint) -> !firrtl.uint
    %1 = firrtl.add %0, %in : (!firrtl.uint, !firrtl.uint<4>) -> !firrtl.uint
    firrtl.connect %x, %1 : !firrtl.uint
    firrtl.connect %out, %x : !firrtl.uint
  }
}
