// RUN: split-file %s %t
// RUN: cd %t
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-blackbox-reader))' Foo.mlir | FileCheck Foo.mlir
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-blackbox-reader))' NoDUT.mlir | FileCheck NoDUT.mlir

//--- Baz.sv
/* Baz */
//--- Foo.mlir
firrtl.circuit "Foo" attributes {annotations = [
{class = "sifive.enterprise.firrtl.TestBenchDirAnnotation", dirname = "../testbench"},
{class = "sifive.enterprise.firrtl.ExtractCoverageAnnotation", directory = "cover"}
]}
{
  // CHECK-LABEL: firrtl.extmodule @ExtFoo()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  // CHECK-SAME: class = "circt.VerbatimBlackBoxAnno", files = [{content = "// world", name = "hello.v", output_file = "../testbench/hello.v"}]
  firrtl.extmodule @ExtFoo() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello.v", text = "// world"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo2()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  // CHECK-SAME: class = "circt.VerbatimBlackBoxAnno", files = [{content = "// world", name = "hello2.v", output_file = "cover/hello2.v"}]
  firrtl.extmodule @ExtFoo2() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello2.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo3()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  // CHECK-SAME: class = "circt.VerbatimBlackBoxAnno", files = [{content = "// world", name = "hello3.v", output_file = "../testbench/hello3.v"}]
  firrtl.extmodule @ExtFoo3() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello3.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.module @DUTBlackboxes
  firrtl.module @DUTBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
      firrtl.instance foo2  @ExtFoo2()
      firrtl.instance bar @Bar()
      firrtl.instance baz @Baz()
  }
  // CHECK-LABEL: firrtl.extmodule @Bar()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  // CHECK-SAME: class = "circt.VerbatimBlackBoxAnno", files = [{content = "/* Bar */\0A", name = "Bar.v", output_file = "bar/Bar.v"}]
  firrtl.extmodule @Bar() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "Bar.v", text = "/* Bar */\0A"}], output_file = #hw.output_file<"bar/">}
  // CHECK-LABEL: firrtl.extmodule @Baz()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxPathAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  // CHECK-SAME: class = "circt.VerbatimBlackBoxAnno", files = [{content = "{{.*}}/* Baz */{{.*}}", name = "Baz.sv", output_file = "baz/Baz.sv"}]
  firrtl.extmodule @Baz() attributes {annotations = [{class = "firrtl.transforms.BlackBoxPathAnno", path = "Baz.sv"}], output_file = #hw.output_file<"baz/">}
  // CHECK-LABEL: firrtl.extmodule @Qux()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  // CHECK-SAME: class = "circt.VerbatimBlackBoxAnno", files = [{content = "/* Qux */\0A", name = "Qux.sv", output_file = "qux/NotQux.jpeg"}]
  firrtl.extmodule @Qux() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "Qux.sv", text = "/* Qux */\0A"}], output_file = #hw.output_file<"qux/NotQux.jpeg">}
  firrtl.module @Foo() {
    firrtl.instance foo @ExtFoo()
    firrtl.instance foo3 @ExtFoo3()
    firrtl.instance dut @DUTBlackboxes()
  }


}

//--- NoDUT.mlir
// Check that a TestBenchDirAnnotation has no effect without the presence of a
// MarkDUTAnnotation.
//
// CHECK: firrtl.circuit "NoDUT"
firrtl.circuit "NoDUT" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    dirname = "testbench"
  }
]} {
  // CHECK-LABEL: firrtl.extmodule @NoDUTBlackBox()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  // CHECK-SAME: class = "circt.VerbatimBlackBoxAnno", files = [{content = "module NoDUTBlackBox();\0Aendmodule\0A", name = "NoDUTBlackBox.sv", output_file = "./NoDUTBlackBox.sv"}]
  firrtl.extmodule @NoDUTBlackBox() attributes {annotations = [
  {
    class = "firrtl.transforms.BlackBoxInlineAnno",
    name = "NoDUTBlackBox.sv",
    text = "module NoDUTBlackBox();\nendmodule\n",
    target = "~NoDUT|NoDUTBlackBox"
  }
]}
  firrtl.module @NoDUT() {
    firrtl.instance noDUTBlackBox @NoDUTBlackBox()
  }
}
