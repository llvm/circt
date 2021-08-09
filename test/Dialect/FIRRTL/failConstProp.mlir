// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imconstprop)' -canonicalize='top-down=true region-simplify=true' --split-input-file  %s | FileCheck %s
// These are constant propagation candidates yet to be implemented. 
// Added to xfail to keep track of cases implemented in the Scala firrtl compiler.
; XFAIL: *

//"Registers with constant reset and connection to the same constant" should "be replaced with that constant"
firrtl.circuit "regConstReset"   {
  firrtl.module @regConstReset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
    %c11_ui8 = firrtl.constant 11 : !firrtl.uint<8>
    %r = firrtl.regreset %clock, %reset, %c11_ui8  : !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    %0 = firrtl.mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: %[[C1:.+]] = firrtl.constant 11 
    // CHECK: firrtl.connect %z, %[[C1]]
  }
}

//"Const prop of registers" should "do limited speculative expansion of optimized muxes to absorb bigger cones"
firrtl.circuit "constPropRegMux"   {
  firrtl.module @constPropRegMux(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %r1 = firrtl.reg %clock  : !firrtl.uint<1>
  %r2 = firrtl.reg %clock  : !firrtl.uint<1>
  %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  %0 = firrtl.mux(%en, %c1_ui1, %r1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %r1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  %1 = firrtl.mux(%en, %r2, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %r2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  %2 = firrtl.xor %r1, %r2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %[[C2:.+]] = firrtl.constant 1
    // CHECK: firrtl.connect %out, %[[C2]]
  }
}
