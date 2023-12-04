// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-signatures))' %s | FileCheck --check-prefixes=CHECK %s

firrtl.circuit "Prop" {
  // CHECK-LABEL @Prop(out %y: !firrtl.string)
  firrtl.module @Prop(out %y: !firrtl.string) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.string "test"
    // CHECK: firrtl.propassign
    firrtl.propassign %y, %0 : !firrtl.string
  }

  firrtl.module @emptyVec(in %vi : !firrtl.vector<uint<4>, 0>, out %vo : !firrtl.vector<uint<4>, 0>) {
    firrtl.strictconnect %vo, %vi : !firrtl.vector<uint<4>, 0>
  }

}
