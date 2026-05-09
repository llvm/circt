// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include reset-disconnector | FileCheck %s --check-prefixes=CHECK,ALL
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg 'reg1.*firrtl\.regreset' --keep-best=0 --include reset-disconnector | FileCheck %s --check-prefixes=CHECK,KEEP_REG1

// Test converting all regreset to reg, and independently keeping one
// CHECK-LABEL: firrtl.circuit "Test"
firrtl.circuit "Test" {
  // CHECK-LABEL: firrtl.module @Test
  firrtl.module @Test(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    // ALL: %reg1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    // ALL: %reg2 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<16>
    // ALL-NOT: regreset
    // KEEP_REG1: %reg1 = firrtl.regreset %clock, %reset
    // KEEP_REG1: %reg2 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<16>
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    %c0_ui16 = firrtl.constant 0 : !firrtl.uint<16>
    %reg1 = firrtl.regreset %clock, %reset, %c0_ui8 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
    %reg2 = firrtl.regreset %clock, %reset, %c0_ui16 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<16>, !firrtl.uint<16>
  }
}
