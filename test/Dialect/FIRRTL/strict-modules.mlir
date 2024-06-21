// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-strict-modules))' %s | FileCheck %s

firrtl.circuit "TopLevel" {

  // CHECK-LABEL: @TopLevel
  firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                             out %sink: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) {
    %w = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
    firrtl.matchingconnect %w, %source : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
    firrtl.matchingconnect %sink, %w : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>

// CHECK: %0 = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>> 
// CHECK: firrtl.strictconnect %sink, %0 : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>> 
// CHECK: %w = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
// CHECK: firrtl.matchingconnect %w, %source : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
// CHECK: firrtl.matchingconnect %0, %w : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
  }

  // CHECK-LABEL: @Subfield
  firrtl.module @Subfield(in %source: !firrtl.uint<1>,
                             out %sink: !firrtl.uint<1>) {
    %w = firrtl.wire : !firrtl.bundle<valid: uint<1>>
    %w_valid = firrtl.subfield %w[valid] : !firrtl.bundle<valid: uint<1>>
    firrtl.matchingconnect %w_valid, %source : !firrtl.uint<1>
    firrtl.matchingconnect %sink, %w_valid : !firrtl.uint<1>

// CHECK: %0 = firrtl.wire : !firrtl.uint<1>
// CHECK: firrtl.strictconnect %sink, %0 : !firrtl.uint<1>
// CHECK: %w = firrtl.wire : !firrtl.bundle<valid: uint<1>>
// CHECK: %1 = firrtl.subfield %w[valid] : !firrtl.bundle<valid: uint<1>>
// CHECK: firrtl.matchingconnect %1, %source : !firrtl.uint<1>
// CHECK: firrtl.matchingconnect %0, %1 : !firrtl.uint<1>
}

  // CHECK-LABEL: @Subindex
  firrtl.module @Subindex(in %source: !firrtl.uint<1>,
                             out %sink: !firrtl.uint<1>) {
    %w = firrtl.wire : !firrtl.vector<uint<1>,1>
    %w_valid = firrtl.subindex %w[0] : !firrtl.vector<uint<1>,1>
    firrtl.matchingconnect %w_valid, %source : !firrtl.uint<1>
    firrtl.matchingconnect %sink, %w_valid : !firrtl.uint<1>

// CHECK: %0 = firrtl.wire : !firrtl.uint<1>
// CHECK: firrtl.strictconnect %sink, %0 : !firrtl.uint<1>
// CHECK: %w = firrtl.wire : !firrtl.vector<uint<1>, 1>
// CHECK: %1 = firrtl.subindex %w[0] : !firrtl.vector<uint<1>, 1>
// CHECK: firrtl.matchingconnect %1, %source : !firrtl.uint<1>
// CHECK: firrtl.matchingconnect %0, %1 : !firrtl.uint<1>
}


    // CHECK-LABEL: @Inst
  firrtl.module @Inst(in %source: !firrtl.uint<1>,
                             out %sink: !firrtl.uint<1>) {
    %s_source, %s_sink = firrtl.instance s @Subfield(in source : !firrtl.uint<1>, out sink : !firrtl.uint<1>)
    firrtl.matchingconnect %s_source, %source : !firrtl.uint<1>
    firrtl.matchingconnect %sink, %s_sink : !firrtl.uint<1>

// CHECK: %0 = firrtl.wire : !firrtl.uint<1>
// CHECK: firrtl.strictconnect %sink, %0 : !firrtl.uint<1>
// CHECK: %s_source, %s_sink = firrtl.instance s @Subfield(in source: !firrtl.uint<1>, out sink: !firrtl.uint<1>)
// CHECK: firrtl.matchingconnect %s_source, %source : !firrtl.uint<1>
// CHECK: firrtl.matchingconnect %0, %s_sink : !firrtl.uint<1>
  }


}
