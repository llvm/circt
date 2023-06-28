// RUN: circt-opt --namehints-insensitive-cse %s | FileCheck %s
// CHECK-LABEL: MyModule
hw.module @MyModule(%a : i1, %b: i1) -> (c : i1, d: i1) {
  // CHECK: %0 = comb.and %a, %b {sv.namehint = "foo"} : i1
  %0 = comb.and %a, %b {sv.namehint = "foo"} : i1
  %1 = comb.and %a, %b {sv.namehint = "bar"} : i1
  // CHECK-NEXT: %0, %0
  hw.output %0, %1 : i1, i1
}
