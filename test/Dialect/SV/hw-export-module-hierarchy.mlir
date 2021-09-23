// RUN: circt-opt -pass-pipeline=hw-export-module-hierarchy %s | FileCheck %s

// CHECK: sv.verbatim "{\22instance_name\22:\22TestHarness\22,\22module_name\22:\22TestHarness\22,\22instances\22:[{\22instance_name\22:\22main_design\22,\22module_name\22:\22MainDesign\22,\22instances\22:[{\22instance_name\22:\22inner\22,\22module_name\22:\22InnerModule\22,\22instances\22:[]}]}]}" {output_file = {directory = "", exclude_from_filelist = true, exclude_replicated_ops = true, name = "testharness_hier.json"}, symbols = []}

hw.module @InnerModule(%in: i1) -> (out: i1) {
  hw.output %in : i1
}

hw.module @MainDesign(%in: i1) -> (out: i1) {
  %0 = hw.instance "inner" @InnerModule(in: %in: i1) -> (out: i1)
  hw.output %0 : i1
}

hw.module @TestHarness() attributes {firrtl.moduleHierarchyFile = {directory = "", exclude_from_filelist = true, exclude_replicated_ops = true, name = "testharness_hier.json"}} {
  %0 = hw.constant 1 : i1
  hw.instance "main_design" @MainDesign(in: %0: i1) -> (out: i1)
}
