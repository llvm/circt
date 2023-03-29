// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imdeadcodeelim))' -verify-diagnostics %s | FileCheck %s
// XFAIL: *

// IMDCE does wrong thing trying to create temporary for a reference in some situations.
// This case can be fixed, but if run before expand.when's this is not solvable without
// a proper temporary for reference types.

// CHECK-LABEL: firrtl.circuit "NoWireForLiveRefInputPort"
firrtl.circuit "NoWireForLiveRefInputPort" {
   // CHECK-NOT: @Child
  firrtl.module private @Child(in %in: !firrtl.ref<uint<1>>) { }
  // CHECK: @NoWireForLiveRefInputPort
  firrtl.module @NoWireForLiveRefInputPort(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK-NEXT: %[[REF:.+]] = firrtl.ref.send %in
    // CHECK-NEXT: %[[RES:.+]] = firrtl.ref.resolve %[[REF]]
    // CHECK-NEXT: firrtl.strictconnect %out, %[[RES]]
    // CHECK-NEXT: }
    %child_ref = firrtl.instance child @Child(in in: !firrtl.ref<uint<1>>)
    %res = firrtl.ref.resolve %child_ref : !firrtl.ref<uint<1>>
    %ref = firrtl.ref.send %in : !firrtl.uint<1>
    firrtl.ref.define %child_ref, %ref : !firrtl.ref<uint<1>>
    firrtl.strictconnect %out, %res : !firrtl.uint<1>
  }
}

