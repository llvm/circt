// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-annotate-input-only-modules))' %s | FileCheck %s

// Test that modules with non-hardware output ports (Probe, Integer) ARE annotated
firrtl.circuit "ModuleWithNonHardwareOutput" {
  // CHECK-LABEL: firrtl.module @ModuleWithNonHardwareOutput
  firrtl.module @ModuleWithNonHardwareOutput(in %a: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>, out %i: !firrtl.integer) {
    %child_x, %child_probe, %child_int = firrtl.instance child @ChildWithNonHardwareOutput(in x: !firrtl.uint<8>, out probe: !firrtl.probe<uint<8>>, out int: !firrtl.integer)
    firrtl.matchingconnect %child_x, %a : !firrtl.uint<8>
    firrtl.ref.define %p, %child_probe : !firrtl.probe<uint<8>>
    firrtl.propassign %i, %child_int : !firrtl.integer
  }

  // CHECK-LABEL: firrtl.module private @ChildWithNonHardwareOutput
  // CHECK-SAME: attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}
  firrtl.module private @ChildWithNonHardwareOutput(in %x: !firrtl.uint<8>, out %probe: !firrtl.probe<uint<8>>, out %int: !firrtl.integer) {
    %0 = firrtl.ref.send %x : !firrtl.uint<8>
    firrtl.ref.define %probe, %0 : !firrtl.probe<uint<8>>
    %c42_i = firrtl.integer 42
    firrtl.propassign %int, %c42_i : !firrtl.integer
  }
}

// Test that modules with existing annotations are preserved
firrtl.circuit "ExistingAnnotations" {
  // CHECK-LABEL: firrtl.module @ExistingAnnotations
  firrtl.module @ExistingAnnotations(in %a: !firrtl.uint<8>) {
    %child_x = firrtl.instance child @ChildWithAnnotation(in x: !firrtl.uint<8>)
    firrtl.matchingconnect %child_x, %a : !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module private @ChildWithAnnotation
  // CHECK-SAME: annotations = [{class = "some.other.Annotation"}, {class = "firrtl.passes.InlineAnnotation"}]
  firrtl.module private @ChildWithAnnotation(in %x: !firrtl.uint<8>) attributes {annotations = [{class = "some.other.Annotation"}]} {
  }
}

// Test public modules are NOT annotated (even if input-only)
firrtl.circuit "PublicInputOnly" {
  // CHECK-LABEL: firrtl.module @PublicInputOnly
  // CHECK-NOT: InlineAnnotation
  firrtl.module @PublicInputOnly(in %a: !firrtl.uint<8>) {
  }
}
