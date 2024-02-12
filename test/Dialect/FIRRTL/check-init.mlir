// RUN: circt-opt -allow-unregistered-dialect --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-check-init)))' --split-input-file --verify-diagnostics %s | FileCheck %s

// missing wire init
// CHECK: firrtl.circuit "simplewire"
firrtl.circuit "simplewire"   {
  firrtl.module @simplewire() {
  // expected-error @below {{wire not initialized}}}
    %x = firrtl.wire  : !firrtl.uint<1>
  }
}

// -----

// missing output
// CHECK-NOT: firrtl.circuit "simpleout"
firrtl.circuit "simpleout"   {
  // expected-error @below {{port not initialized}}}
  firrtl.module @simpleout(out %a: !firrtl.uint<1>) {
  }
}


// -----

// missing wire init
// CHECK: firrtl.circuit "simplewire"
firrtl.circuit "simplewire"   {
  firrtl.module @simplewire() {
  // expected-error @below {{wire not initialized}}}
    %x = firrtl.wire  : !firrtl.uint<1>
    %c = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.strictconnect %x, %c : !firrtl.uint<1>
  }
}
