// RUN: circt-opt -allow-unregistered-dialect --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-check-init)))' --split-input-file --verify-diagnostics %s | FileCheck %s

// missing wire init
// CHECK: firrtl.circuit "simplewire"
firrtl.circuit "simplewire"   {
  firrtl.module @simplewire() {
  // expected-error @below {{Wire is not initialized.}}
    %x = firrtl.wire  : !firrtl.uint<1>
  }
}

// -----

// missing output
// CHECK-NOT: firrtl.circuit "simpleout"
firrtl.circuit "simpleout"   {
  // expected-error @below {{Port is not initialized.}}
  firrtl.module @simpleout(out %a: !firrtl.uint<1>) {
  }
}


// -----

// missing wire init
// CHECK: firrtl.circuit "simplewire"
firrtl.circuit "simplewire"   {
  firrtl.module @simplewire() {
  // expected-error @below {{Wire is not initialized.}}
    %x = firrtl.wire  : !firrtl.uint<1>
    %c = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.strictconnect %x, %c : !firrtl.uint<1>
  }
}

// -----

// missing wire init
// CHECK: firrtl.circuit "bundlewire"
firrtl.circuit "bundlewirec"   {
  firrtl.module @bundlewirec(in %a: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not fully initialized.}}
    %x = firrtl.wire  : !firrtl.bundle<a: uint<1>, b:uint<2>>
    %c = firrtl.constant 1 : !firrtl.uint<1>
    %t = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.strictconnect %t, %c : !firrtl.uint<1>
  }
  firrtl.module @bundlewirep(in %a: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not fully initialized.}}
    %x = firrtl.wire  : !firrtl.bundle<a: uint<1>, b:uint<2>>
    %t = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.strictconnect %t, %a : !firrtl.uint<1>
  }
}
