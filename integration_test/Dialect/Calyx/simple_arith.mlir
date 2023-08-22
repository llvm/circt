// This test lowers an SCF construct through Calyx, FSM and (TODO)
// to RTL.
// RUN: hlstool %s --calyx-hw --ir --output-level=sv | FileCheck %s

// TODO: ... simulate the hardware!

// CHECK: hw.module @control
// CHECK: hw.module @main
// CHECK: hw.instance "controller" @control
func.func @main(%arg0 : i32, %arg1 : i32) -> i32 {
  %0 = arith.cmpi slt, %arg0, %arg1 : i32
  cf.cond_br %0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg0 : i32)
^bb2:
  cf.br ^bb3(%arg1 : i32)
^bb3(%1 : i32):
  return %1 : i32
}
