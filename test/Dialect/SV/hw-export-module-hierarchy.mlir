// RUN: rm -rf %t
// RUN: circt-opt -pass-pipeline='hw-export-module-hierarchy{dir-name=%t}' %s
// RUN: FileCheck %s < %t/testharness_hier.json

// CHECK: {"instance_name":"TestHarness","module_name":"TestHarness","instances":[{"instance_name":"main_design","module_name":"MainDesign","instances":[{"instance_name":"inner","module_name":"InnerModule","instances":[]}]}]}

hw.module @InnerModule(%in: i1) -> (out: i1) {
  hw.output %in : i1
}

hw.module @MainDesign(%in: i1) -> (out: i1) {
  %0 = hw.instance "inner" @InnerModule(in: %in: i1) -> (out: i1)
  hw.output %0 : i1
}

hw.module @TestHarness() attributes {firrtl.moduleHierarchyFile = #hw.output_file<"testharness_hier.json", excludeFromFileList>} {
  %0 = hw.constant 1 : i1
  hw.instance "main_design" @MainDesign(in: %0: i1) -> (out: i1)
}
