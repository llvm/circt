// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-blackbox-reader)' %s | FileCheck %s

firrtl.circuit "Foo" attributes {annotations = [
{class = "sifive.enterprise.firrtl.TestBenchDirAnnotation", dirname = "../testbench"},
{class = "sifive.enterprise.firrtl.ExtractCoverageAnnotation", directory = "cover"}
]}
{
  // CHECK-LABEL: firrtl.extmodule @ExtFoo()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  firrtl.extmodule @ExtFoo() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello.v", text = "// world"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo2()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.extmodule @ExtFoo2() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello2.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo3()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.extmodule @ExtFoo3() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello3.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.module @DUTBlackboxes
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.module @DUTBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}, {class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello_dut.v", text = "// world"}]} {
      firrtl.instance foo2  @ExtFoo2()
  }
  firrtl.module @Foo() {
    firrtl.instance foo @ExtFoo()
    firrtl.instance foo3 @ExtFoo3()
    firrtl.instance dut @DUTBlackboxes()
  }
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"../testbench/hello.v">, symbols = []}
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"cover/hello2.v">, symbols = []}
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"../testbench/hello3.v">, symbols = []}
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"./hello_dut.v">, symbols = []}
}
