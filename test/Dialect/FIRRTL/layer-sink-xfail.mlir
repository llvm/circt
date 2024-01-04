// RUN: circt-opt -pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-layer-sink)))" %s | FileCheck %s
// XFAIL: *

// Tests of things which do not currently sink, but should.
//
// CHECK-LABEL: firrtl.circuit "LayerSinkExpectedFailures"
firrtl.circuit "LayerSinkExpectedFailures" {
  firrtl.layer @Subaccess bind {}
  firrtl.layer @Subfield bind {}
  firrtl.layer @Subindex bind {}

  // CHECk: firrtl.module @LayerSinkExpectedFailures
  firrtl.module @LayerSinkExpectedFailures(in %a: !firrtl.uint<1>) {
    %wire_subaccess = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.subaccess %wire_subaccess[%a] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
    firrtl.strictconnect %2, %2 : !firrtl.uint<1>
    firrtl.layerblock @Subaccess {
      %layer_wire_subaccess = firrtl.node %wire_subaccess : !firrtl.vector<uint<1>, 2>
    }
    // CHECK-NEXT: firrtl.layerblock @Subaccess {
    // CHECK-NEXT:   %wire_subaccess = firrtl.wire
    // CHECK-NEXT:   %2 = firrtl.subaccess %wire_subaccess[%a] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
    // CHECK-NEXT:   firrtl.strictconnect %2, %2 : !firrtl.uint<1>
    // CHECK-NEXT:   firrtl.strictconnect %3, %2
    // CHECK-NEXT:   %layer_wire_subaccess = firrtl.node %wire_subaccess
    // CHECK-NEXT: }

    %wire_subfield = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %wire_subfield[a] : !firrtl.bundle<a: uint<1>>
    firrtl.strictconnect %0, %0 : !firrtl.uint<1>
    firrtl.layerblock @Subfield {
      %layer_wire_subfield = firrtl.node %wire_subfield : !firrtl.bundle<a: uint<1>>
    }
    // CHECK-NEXT: firrtl.layerblock @Subfield {
    // CHECK-NEXT:   %wire_subfield = firrtl.wire
    // CHECK-NEXT:   %0 = firrtl.subfield %wire_subfield[a]
    // CHECK-NEXT:   firrtl.strictconnect %0, %0
    // CHECK-NEXT:   %layer_wire_subfield = firrtl.node %wire_subfield
    // CHECK-NEXT: }

    %wire_subindex = firrtl.wire : !firrtl.vector<uint<1>, 1>
    %1 = firrtl.subindex %wire_subindex[0] : !firrtl.vector<uint<1>, 1>
    firrtl.strictconnect %1, %1 : !firrtl.uint<1>
    firrtl.layerblock @Subindex {
      %layer_wire_subindex = firrtl.node %wire_subindex : !firrtl.vector<uint<1>, 1>
    }
    // CHECK-NEXT: firrtl.layerblock @Subindex {
    // CHECK-NEXT:   %wire_subindex = firrtl.wire
    // CHECK-NEXT:   %1 = firrtl.subindex %wire_subindex[0]
    // CHECK-NEXT:   firrtl.strictconnect %1, %1
    // CHECK-NEXT:   %layer_wire_subindex = firrtl.node %wire_subindex
    // CHECK-NEXT: }
  }
}
