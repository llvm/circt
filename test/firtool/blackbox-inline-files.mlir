// Test that inline files are properly linked to hw.module.extern operations
// RUN: firtool %s --ir-hw | FileCheck %s

firrtl.circuit "InlineFilesTest" {
  // CHECK-LABEL: hw.module.extern @ExtWithMultipleInlines()
  // CHECK-SAME: files = [@blackbox_inline1.v, @blackbox_inline2.sv]
  firrtl.extmodule @ExtWithMultipleInlines() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "inline1.v", text = "module ExtWithMultipleInlines(); endmodule"},
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "inline2.sv", text = "// Another inline file"}
  ]}
  
  // CHECK-LABEL: hw.module.extern @ExtWithSingleInline()
  // CHECK-SAME: files = [@blackbox_single.v]
  firrtl.extmodule @ExtWithSingleInline() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "single.v", text = "// Single inline file"}
  ]}
  
  // CHECK-LABEL: hw.module.extern @ExtWithPathOnly()
  // CHECK-NOT: files
  firrtl.extmodule @ExtWithPathOnly()
  
  // CHECK-LABEL: hw.module.extern @ExtWithMixedAnnotations()
  // CHECK-SAME: files = [@blackbox_mixed.v]
  firrtl.extmodule @ExtWithMixedAnnotations() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "mixed.v", text = "// Mixed annotations"},
    {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}
  ]}
  
  firrtl.module @InlineFilesTest() {
    firrtl.instance ext1 @ExtWithMultipleInlines()
    firrtl.instance ext2 @ExtWithSingleInline()
    firrtl.instance ext3 @ExtWithPathOnly()
    firrtl.instance ext4 @ExtWithMixedAnnotations()
  }

  // Verify that emit.file operations are created with the correct symbols
  // CHECK: emit.file ".{{/|\\\\}}inline1.v" sym @blackbox_inline1.v
  // CHECK: emit.file ".{{/|\\\\}}inline2.sv" sym @blackbox_inline2.sv
  // CHECK: emit.file ".{{/|\\\\}}single.v" sym @blackbox_single.v
  // CHECK: emit.file ".{{/|\\\\}}mixed.v" sym @blackbox_mixed.v
}
