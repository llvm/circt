// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-blackbox-reader)' %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.extmodule @Foo()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  firrtl.extmodule @Foo() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello.v", text = "// world"}]}
}
