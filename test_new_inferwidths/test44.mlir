firrtl.circuit "MuxIntrinsics" { 
  firrtl.module @MuxIntrinsics(in %sel_0w: !firrtl.uint<0>, in %sel_1w: !firrtl.uint<1>, in %high: !firrtl.uint<1>, in %low: !firrtl.uint<1>, out %out1: !firrtl.uint, out %out2: !firrtl.uint) {
    %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
    %c3_ui3 = firrtl.constant 3 : !firrtl.uint<3>
    %c2_ui2 = firrtl.constant 2 : !firrtl.uint<2>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1 = firrtl.constant 0: !firrtl.uint
    %sel = firrtl.wire : !firrtl.uint
    firrtl.connect %sel, %sel_0w : !firrtl.uint, !firrtl.uint<0>
    // CHECK: firrtl.int.mux2cell
    // CHECK-SAME: (!firrtl.uint<0>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %0 = firrtl.int.mux2cell(%sel, %c0_ui1, %c1) : (!firrtl.uint, !firrtl.uint<1>, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out1, %0: !firrtl.uint, !firrtl.uint
    %sel2 = firrtl.wire : !firrtl.uint
    firrtl.connect %sel2, %sel_1w : !firrtl.uint, !firrtl.uint<1>
    // CHECK: firrtl.int.mux4cell
    // CHECK-SAME: (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint<1>) -> !firrtl.uint<3>
    %1 = firrtl.int.mux4cell(%sel2, %c1_ui1, %c2_ui2, %c3_ui3, %c1) : (!firrtl.uint, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out2, %1: !firrtl.uint, !firrtl.uint
  }
}