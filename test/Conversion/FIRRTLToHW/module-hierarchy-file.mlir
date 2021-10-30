// RUN: circt-opt -lower-firrtl-to-hw  %s | FileCheck %s

firrtl.circuit "MyTestHarness" attributes {annotations = [
  {class = "sifive.enterprise.firrtl.ModuleHierarchyAnnotation", filename = "./dir1/filename1.json" },
  {class = "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation", filename = "./dir2/filename2.json" }]}
{
  // CHECK-LABEL: hw.module @MyDUT
  // CHECK: attributes {firrtl.moduleHierarchyFile = #hw.output_file<"./dir1/filename1.json", excludeFromFileList>}
  firrtl.module @MyDUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}

  // CHECK-LABEL: hw.module @MyTestHarness
  // CHECK: attributes {firrtl.moduleHierarchyFile = #hw.output_file<"./dir2/filename2.json", excludeFromFileList>}
  firrtl.module @MyTestHarness() {
    firrtl.instance myDUT @MyDUT()
  }
}
