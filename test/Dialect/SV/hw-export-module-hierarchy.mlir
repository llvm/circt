// RUN: rm -rf %t
// RUN: circt-opt -pass-pipeline='builtin.module(hw-export-module-hierarchy{dir-name=%t})' %s
// RUN: FileCheck %s < %t/testharness_hier.json

// CHECK:      {
// CHECK-NEXT:   "instance_name": "TestHarness",
// CHECK-NEXT:   "module_name": "TestHarness",
// CHECK-NEXT:   "instances": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "instance_name": "main_design",
// CHECK-NEXT:       "module_name": "MainDesign",
// CHECK-NEXT:       "instances": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "instance_name": "inner",
// CHECK-NEXT:           "module_name": "InnerModule",
// CHECK-NEXT:           "instances": []
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

hw.module @InnerModule(%in: i1) -> (out: i1) {
  hw.output %in : i1
}

hw.module @MainDesign(%in: i1) -> (out: i1) {
  %0 = hw.instance "inner" @InnerModule(in: %in: i1) -> (out: i1)
  hw.output %0 : i1
}

hw.module @TestHarness() attributes {firrtl.moduleHierarchyFile = [#hw.output_file<"testharness_hier.json", excludeFromFileList>]} {
  %0 = hw.constant 1 : i1
  hw.instance "main_design" @MainDesign(in: %0: i1) -> (out: i1)
}
