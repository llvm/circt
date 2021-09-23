// RUN: circt-opt %s --lower-scf-to-calyx -split-input-file | FileCheck %s

// CHECK:      module  {
// CHECK-NEXT:   calyx.program "main"  {
// CHECK-NEXT:     calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
// CHECK-NEXT:       calyx.wires  {
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.control  {
// CHECK-NEXT:         calyx.seq  {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
module {
  func @main() {
    return
  }
}

// -----

// CHECK:      module  {
// CHECK-NEXT:   calyx.program "main"  {
// CHECK-NEXT:     calyx.component @main(%in0: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-NEXT:       %true = hw.constant true
// CHECK-NEXT:       %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register "ret_arg0_reg" : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:       calyx.wires  {
// CHECK-NEXT:         calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:         calyx.group @ret_assign_0  {
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.in = %in0 : i32
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.control  {
// CHECK-NEXT:         calyx.seq  {
// CHECK-NEXT:           calyx.enable @ret_assign_0
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
module {
  func @main(%arg0 : i32) -> i32 {
    return %arg0 : i32
  }
}
