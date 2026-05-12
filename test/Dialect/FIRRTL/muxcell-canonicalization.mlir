// Test for mux2cell/mux4cell canonicalization (Issue #5448)
// RUN: circt-opt -canonicalize %s | FileCheck %s

// These tests verify that:
// 1. WITHOUT the feature: mux2cell/mux4cell operations remain unsimplified
// 2. WITH the feature: mux2cell/mux4cell are simplified to not(sel) or sel

firrtl.circuit "MuxCellCanonicalization" {
  // CHECK-LABEL: firrtl.module @MuxCellCanonicalization
  firrtl.module @MuxCellCanonicalization(
    in %sel: !firrtl.uint<1>,
    out %out_mux2_not: !firrtl.uint<1>,
    out %out_mux4_not: !firrtl.uint<1>,
    out %out_mux4_identity: !firrtl.uint<1>
  ) {
    %c0 = firrtl.constant 0 : !firrtl.uint<1>
    %c1 = firrtl.constant 1 : !firrtl.uint<1>

    // Test: mux2cell(sel, 0, 1) -> not(sel)
    // This simplifies to a NOT operation when selector is 1-bit.
    // Without the canonicalization pattern, this would remain as mux2cell.
    // CHECK: [[NOT1:%.+]] = firrtl.not %sel
    // CHECK: firrtl.matchingconnect %out_mux2_not, [[NOT1]]
    %mux2 = firrtl.int.mux2cell(%sel, %c0, %c1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.matchingconnect %out_mux2_not, %mux2 : !firrtl.uint<1>

    // Test: mux4cell(sel, 0, 1, 0, 1) -> not(sel)
    // This simplifies to a NOT operation. The selector is padded to uint<2>.
    // Pattern matches only when selector and operand types match (both uint<1>).
    // Without the canonicalization pattern, this would remain as mux4cell with pad.
    // CHECK: [[NOT2:%.+]] = firrtl.not %sel
    // CHECK: firrtl.matchingconnect %out_mux4_not, [[NOT2]]
    %mux4 = firrtl.int.mux4cell(%sel, %c0, %c1, %c0, %c1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.matchingconnect %out_mux4_not, %mux4 : !firrtl.uint<1>

    // Test: mux4cell(sel, 1, 0, 1, 0) -> sel (identity pattern)
    // This simplifies to just the selector. Operands are (1, 0, 1, 0) which maps to:
    //   sel=0 -> v0=0, sel=1 -> v1=1, sel=2 -> v2=0, sel=3 -> v3=1
    // So for 1-bit selector: sel=0 returns 0, sel=1 returns 1, which is exactly sel.
    // Without the fold, this would remain as mux4cell.
    // CHECK: firrtl.matchingconnect %out_mux4_identity, %sel
    %mux4id = firrtl.int.mux4cell(%sel, %c1, %c0, %c1, %c0) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.matchingconnect %out_mux4_identity, %mux4id : !firrtl.uint<1>
  }
}
