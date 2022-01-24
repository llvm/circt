// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{preserve-aggregate=true})' %s | FileCheck %s
// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-lower-types{preserve-aggregate=true preserve-public-types=false})' %s | FileCheck --check-prefix=NOT_PRESERVE_PUBLIC_TYPES %s

firrtl.circuit "TopLevel" {
  // CHECK-LABEL: firrtl.extmodule @External(in source_valid: !firrtl.uint<1>)
  // CHECK-LABEL: firrtl.module @TopLevel(in %source_valid: !firrtl.uint<1>, out %sink_valid: !firrtl.uint<1>)
  // NOT_PRESERVE_PUBLIC_TYPES-LABEL: firrtl.extmodule @External(in source: !firrtl.bundle<valid: uint<1>>)
  // NOT_PRESERVE_PUBLIC_TYPES-LABEL: firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>>, out %sink: !firrtl.bundle<valid: uint<1>>)
  firrtl.extmodule @External(in source: !firrtl.bundle<valid: uint<1>>)
  firrtl.module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>>,
                          out %sink: !firrtl.bundle<valid: uint<1>>) {
  }
}