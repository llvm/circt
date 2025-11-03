firrtl.circuit "ComparisonOp" {
  firrtl.module @ComparisonOp(in %a: !firrtl.uint<2>, in %b: !firrtl.uint<3>) {
    // CHECK: %6 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %7 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %8 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %9 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %10 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %11 = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.leq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %1 = firrtl.lt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %2 = firrtl.geq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %3 = firrtl.gt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %4 = firrtl.eq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %5 = firrtl.neq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %6 = firrtl.wire : !firrtl.uint
    %7 = firrtl.wire : !firrtl.uint
    %8 = firrtl.wire : !firrtl.uint
    %9 = firrtl.wire : !firrtl.uint
    %10 = firrtl.wire : !firrtl.uint
    %11 = firrtl.wire : !firrtl.uint
    firrtl.connect %6, %0 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %7, %1 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %8, %2 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %9, %3 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %10, %4 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %11, %5 : !firrtl.uint, !firrtl.uint<1>
  }
}