// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets,any(firrtl-check-uninferred-resets)))' --verify-diagnostics %s
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets,firrtl-inliner,any(firrtl-check-uninferred-resets)))' %s | FileCheck %s

firrtl.circuit "LegalUninferredReset" {
  // The following two diagnostics will only appear if firrtl-inliner is not enabled
  // expected-note @+2 {{the module with this uninferred reset port was defined here}}
  // expected-error @+1 {{a port "reset" with abstract reset type was unable to be inferred by InferResets}}
  firrtl.module private @Adder(in %clock: !firrtl.clock, in %reset: !firrtl.reset, in %in: !firrtl.uint<10>, out %out: !firrtl.uint<10>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.const.uint<1>
    %0 = firrtl.add %in, %c1_ui1 : (!firrtl.uint<10>, !firrtl.const.uint<1>) -> !firrtl.uint<11>
    %_out_T = firrtl.node interesting_name %0 : !firrtl.uint<11>
    %1 = firrtl.tail %_out_T, 1 : (!firrtl.uint<11>) -> !firrtl.uint<10>
    %_out_T_1 = firrtl.node interesting_name %1 : !firrtl.uint<10>
    firrtl.matchingconnect %out, %_out_T_1 : !firrtl.uint<10>
  }
  firrtl.module @LegalUninferredReset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
  }
}

// CHECK-NOT: firrtl.circuit "Adder"
// CHECK: firrtl.circuit "LegalUninferredReset"