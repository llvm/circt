// -----// IR Dump Before InferWidths (firrtl-infer-widths) //----- //
firrtl.circuit "Foo" {
  firrtl.module @Foo(in %in: !firrtl.uint<4>, in %clock: !firrtl.clock, out %out: !firrtl.uint) attributes {convention = #firrtl<convention scalarized>} {
    %x1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint
    %x2 = firrtl.wire : !firrtl.uint
    %x3 = firrtl.wire : !firrtl.uint
    %0 = firrtl.mul %x2, %in : (!firrtl.uint, !firrtl.uint<4>) -> !firrtl.uint
    %1 = firrtl.mul %0, %x2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %x1, %1 : !firrtl.uint
    %2 = firrtl.shr %x1, 2 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.pad %2, 1 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %x3, %3 : !firrtl.uint
    %4 = firrtl.tail %x3, 2 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %x2, %4 : !firrtl.uint
    firrtl.connect %out, %x1 : !firrtl.uint
  }
}