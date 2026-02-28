// RUN: split-file %s %t
// RUN: not circt-lec %t/a.mlir %t/b.mlir --c1 top_a --c2 top_not_exist 2>&1 | FileCheck %s

// CHECK: module "top_not_exist" was not found in the second module

//--- a.mlir
hw.module @top_a(in %a : i8, out b : i8) {
  hw.output %a : i8
}

//--- b.mlir
hw.module @top_b(in %a : i8, out b : i8) {
  hw.output %a : i8
}
