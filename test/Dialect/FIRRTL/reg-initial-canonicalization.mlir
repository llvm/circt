// RUN: circt-opt -canonicalize %s | FileCheck %s

firrtl.circuit "RegInitialCanon" {
  firrtl.module @RegInitialCanon() {}

  // A constant register whose initial value differs from the folded constant
  // must NOT be folded away.
  // CHECK-LABEL: firrtl.module @FoldBlocked
  firrtl.module @FoldBlocked(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, out %q: !firrtl.uint<8>) {
    // CHECK: %r = firrtl.reg %clock {initial = 5 : ui8}
    %r = firrtl.reg %clock {initial = 5 : ui8} : !firrtl.clock, !firrtl.uint<8>
    %c3 = firrtl.constant 3 : !firrtl.uint<8>
    %m = firrtl.mux(%cond, %r, %c3) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.matchingconnect %r, %m : !firrtl.uint<8>
    firrtl.matchingconnect %q, %r : !firrtl.uint<8>
  }

  // A constant register whose initial value equals the folded constant may be
  // folded away.
  // CHECK-LABEL: firrtl.module @FoldAllowed
  firrtl.module @FoldAllowed(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>, out %q: !firrtl.uint<8>) {
    // CHECK-NOT: firrtl.reg
    // CHECK: %c3_ui8 = firrtl.constant 3
    %r = firrtl.reg %clock {initial = 3 : ui8} : !firrtl.clock, !firrtl.uint<8>
    %c3 = firrtl.constant 3 : !firrtl.uint<8>
    %m = firrtl.mux(%cond, %r, %c3) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.matchingconnect %r, %m : !firrtl.uint<8>
    firrtl.matchingconnect %q, %r : !firrtl.uint<8>
  }

  // Promotion of a `reg` with a hidden reset to a `regreset` must carry the
  // `initial` attribute through.
  // CHECK-LABEL: firrtl.module @PromotePreservesInitial
  firrtl.module @PromotePreservesInitial(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>) {
    // CHECK: firrtl.regreset {{.*}} {initial = 9 : ui8}
    %r = firrtl.reg %clock {initial = 9 : ui8} : !firrtl.clock, !firrtl.uint<8>
    %c1 = firrtl.constant 1 : !firrtl.uint<8>
    %m = firrtl.mux(%reset, %c1, %d) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    firrtl.matchingconnect %r, %m : !firrtl.uint<8>
    firrtl.matchingconnect %q, %r : !firrtl.uint<8>
  }

  // Zero-reset regreset rewritten to a plain reg forwards `initial`.
  // CHECK-LABEL: firrtl.module @ZeroResetForwardsInitial
  firrtl.module @ZeroResetForwardsInitial(in %clock: !firrtl.clock, out %q: !firrtl.uint<8>) {
    // CHECK: firrtl.reg %clock {initial = 4 : ui8}
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    %r = firrtl.regreset %clock, %zero, %c0 {initial = 4 : ui8} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %q, %r : !firrtl.uint<8>
  }
}
