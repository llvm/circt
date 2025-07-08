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
  firrtl.extmodule @ExtFoo() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello.v", text = "// world"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo2()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.extmodule @ExtFoo2() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello2.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo3()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.extmodule @ExtFoo3() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello3.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.module @DUTBlackboxes
  firrtl.module @DUTBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
      firrtl.instance foo2  @ExtFoo2()
      firrtl.instance bar @Bar()
      firrtl.instance baz @Baz()
  }
  firrtl.extmodule @Bar() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "Bar.v", text = "/* Bar */\0A"}], output_file = #hw.output_file<"bar/">}
  firrtl.extmodule @Baz() attributes {annotations = [{class = "firrtl.transforms.BlackBoxPathAnno", path = "Baz.sv"}], output_file = #hw.output_file<"baz/">}
  firrtl.extmodule @Qux() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "Qux.sv", text = "/* Qux */\0A"}], output_file = #hw.output_file<"qux/NotQux.jpeg">}
  firrtl.module @Foo() {
    firrtl.instance foo @ExtFoo()
    firrtl.instance foo3 @ExtFoo3()
    firrtl.instance dut @DUTBlackboxes()
  }

  // CHECK:      emit.file "..{{/|\\\\}}testbench{{/|\\\\}}hello.v" sym @blackbox_hello.v {
  // CHECK-NEXT:   emit.verbatim "// world"
  // CHECK-NEXT: }
  // CHECK:      emit.file "cover{{/|\\\\}}hello2.v" sym @blackbox_hello2.v {
  // CHECK-NEXT:   emit.verbatim "// world"
  // CHECK-NEXT: }
  // CHECK:      emit.file "..{{/|\\\\}}testbench{{/|\\\\}}hello3.v" sym @blackbox_hello3.v {
  // CHECK-NEXT:   emit.verbatim "// world"
  // CHECK-NEXT: }
  // CHECK:      emit.file "bar{{/|\\\\}}Bar.v" sym @blackbox_Bar.v {
  // CHECK-NEXT:   emit.verbatim "/* Bar */\0A"
  // CHECK-NEXT: }
  // CHECK:      emit.file "baz{{/|\\\\}}Baz.sv" sym @blackbox_Baz.sv {
  // CHECK-NEXT:   emit.verbatim "/* Baz */{{(\\0D)?}}\0A"
  // CHECK-NEXT: }
  // CHECK:      emit.file "qux{{/|\\\\}}NotQux.jpeg" sym @blackbox_Qux.sv {
  // CHECK-NEXT:   emit.verbatim "/* Qux */\0A"
  // CHECK-NEXT: }
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
  // CHECK:      emit.file ".{{/|\\\\}}NoDUTBlackBox.sv" sym @blackbox_NoDUTBlackBox.sv {
  // CHECK-NEXT:   emit.verbatim "module NoDUTBlackBox();\0Aendmodule\0A"
  // CHECK-NEXT: }
}
