// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{preserve-aggregate=all})' %s | FileCheck %s
// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{preserve-aggregate=all preserve-public-types=false})' %s | FileCheck --check-prefix=NOT_PRESERVE_PUBLIC_TYPES %s
// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{preserve-aggregate=vec})' %s | FileCheck --check-prefix=VEC %s
// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{preserve-aggregate=1d-vec})' %s | FileCheck --check-prefix=1D_VEC %s

firrtl.circuit "TopLevel" {
  // CHECK-LABEL: firrtl.extmodule @External(in source_valid: !firrtl.uint<1>)
  // CHECK-LABEL: firrtl.module @TopLevel(in %source_valid: !firrtl.uint<1>, out %sink_valid: !firrtl.uint<1>)
  // NOT_PRESERVE_PUBLIC_TYPES-LABEL: firrtl.extmodule @External(in source: !firrtl.bundle<valid: uint<1>>)
  // NOT_PRESERVE_PUBLIC_TYPES-LABEL: firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>>, out %sink: !firrtl.bundle<valid: uint<1>>)
  firrtl.extmodule @External(in source: !firrtl.bundle<valid: uint<1>>)
  firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>>,
                          out %sink: !firrtl.bundle<valid: uint<1>>) {
  }
  // CHECK: @Foo(in %a: !firrtl.bundle<a: vector<vector<uint<1>, 2>, 2>>)
  // VEC: @Foo(in %a_a: !firrtl.vector<vector<uint<1>, 2>, 2>)
  // 1D_VEC: @Foo(in %a_a_0: !firrtl.vector<uint<1>, 2>, in %a_a_1: !firrtl.vector<uint<1>, 2>)
  firrtl.module private @Foo(in %a: !firrtl.bundle<a: vector<vector<uint<1>, 2>, 2>>) {
  }

  // CHECK-LABEL: FieldSensitiveSymbolsOnOps
  firrtl.module @FieldSensitiveSymbolsOnOps(in %in: !firrtl.bundle<b: uint<1>, c: bundle<d: uint<1>>, e: uint<1>>) {
    // CHECK-NEXT:  %a = firrtl.wire sym [<@a_subfield_1,1,public>, <@a_subfield_2,2,public>]
    // CHECK-SAME: {circt.fieldID = 0 : i32, class = "circt.test"}
    %a = firrtl.wire  {annotations = [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
                                      {circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
                                      {circt.fieldID = 0 : i32, class = "circt.test"}]}
                                      : !firrtl.bundle<b: uint<1>, c: bundle<d: uint<1>>, e: uint<1>>
    firrtl.strictconnect %a, %in : !firrtl.bundle<b: uint<1>, c: bundle<d: uint<1>>, e: uint<1>>
  }
}
