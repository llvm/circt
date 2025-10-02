// RUN: split-file %s %t
// RUN: cd %t
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-blackbox-reader))' Foo.mlir | FileCheck Foo.mlir
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-blackbox-reader))' NoDUT.mlir | FileCheck NoDUT.mlir
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-blackbox-reader),lower-firrtl-to-hw)' InlineFiles.mlir | FileCheck InlineFiles.mlir

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
  // CHECK-SAME: class = "circt.InlineFilesAnnotation"
  // CHECK-SAME: files = [@blackbox_hello.v]
  firrtl.extmodule @ExtFoo() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello.v", text = "// world"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo2()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "circt.InlineFilesAnnotation"
  // CHECK-SAME: files = [@blackbox_hello2.v]
  firrtl.extmodule @ExtFoo2() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello2.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo3()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "circt.InlineFilesAnnotation"
  // CHECK-SAME: files = [@blackbox_hello3.v]
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

//--- InlineFiles.mlir
// Test that inline files are properly linked to hw.module.extern operations
// after FIRRTL-to-HW lowering.
//
firrtl.circuit "InlineFilesTest" {
  // CHECK-LABEL: hw.module.extern @ExtWithInline()
  // CHECK-SAME: files = [@blackbox_inline1.v, @blackbox_inline2.sv]
  firrtl.extmodule @ExtWithInline() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "inline1.v", text = "module ExtWithInline(); endmodule"},
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "inline2.sv", text = "// Another file"}
  ]}

  // CHECK-LABEL: hw.module.extern @ExtWithoutInline()
  // CHECK-NOT: files
  firrtl.extmodule @ExtWithoutInline()

  // CHECK-LABEL: hw.module.extern @ExtWithSingleInline()
  // CHECK-SAME: files = [@blackbox_single.v]
  firrtl.extmodule @ExtWithSingleInline() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "single.v", text = "// Single inline file"}
  ]}

  firrtl.module @InlineFilesTest() {
    firrtl.instance ext1 @ExtWithInline()
    firrtl.instance ext2 @ExtWithoutInline()
    firrtl.instance ext3 @ExtWithSingleInline()
  }

  // CHECK: emit.file ".{{/|\\\\}}inline1.v" sym @blackbox_inline1.v {
  // CHECK-NEXT: emit.verbatim "module ExtWithInline(); endmodule"
  // CHECK-NEXT: }
  // CHECK: emit.file ".{{/|\\\\}}inline2.sv" sym @blackbox_inline2.sv {
  // CHECK-NEXT: emit.verbatim "// Another file"
  // CHECK-NEXT: }
  // CHECK: emit.file ".{{/|\\\\}}single.v" sym @blackbox_single.v {
  // CHECK-NEXT: emit.verbatim "// Single inline file"
  // CHECK-NEXT: }
}
