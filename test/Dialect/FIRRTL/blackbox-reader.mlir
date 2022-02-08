// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-blackbox-reader)' %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.extmodule @ExtFoo()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  firrtl.extmodule @ExtFoo() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello.v", text = "// world"}]}
  // CHECK-LABEL: firrtl.module @DUTBlackboxes
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.module @DUTBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}, {class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello_dut.v", text = "// world"}]} {
  }
  firrtl.module @Foo() {
    firrtl.instance foo @ExtFoo()
    firrtl.instance dut @DUTBlackboxes()
  }
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"../testbench/hello.v">, symbols = []}
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"./hello_dut.v">, symbols = []}
}
