// RUN: circt-opt %s --lower-scf-to-calyx -canonicalize -split-input-file | FileCheck %s

// CHECK:      module attributes {calyx.entrypoint = "main"} {
// CHECK-NEXT:   calyx.component @main(%in0: i32, %in1: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_slt_0.left, %std_slt_0.right, %std_slt_0.out = calyx.std_slt @std_slt_0 : i32, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %bb3_arg0_reg.in, %bb3_arg0_reg.write_en, %bb3_arg0_reg.clk, %bb3_arg0_reg.reset, %bb3_arg0_reg.out, %bb3_arg0_reg.done = calyx.register @bb3_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.comb_group @bb0_0  {
// CHECK-NEXT:         calyx.assign %std_slt_0.left = %in0 : i32
// CHECK-NEXT:         calyx.assign %std_slt_0.right = %in1 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb1_to_bb3  {
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.in = %in0 : i32
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %bb3_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb2_to_bb3  {
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.in = %in1 : i32
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %bb3_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %bb3_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.if %std_slt_0.out with @bb0_0  {
// CHECK-NEXT:           calyx.seq  {
// CHECK-NEXT:             calyx.enable @bb1_to_bb3
// CHECK-NEXT:           }
// CHECK-NEXT:         } else  {
// CHECK-NEXT:           calyx.seq  {
// CHECK-NEXT:             calyx.enable @bb2_to_bb3
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
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
}

// -----

// Test a while op where the loop body contains basic blocks that may be simplified.

// CHECK:      module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%in0: i32, %in1: i32, %in2: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %c0_i32 = hw.constant 0 : i32
// CHECK-DAG:      %std_add_2.left, %std_add_2.right, %std_add_2.out = calyx.std_add @std_add_2 : i32, i32, i32
// CHECK-DAG:      %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %std_slt_1.left, %std_slt_1.right, %std_slt_1.out = calyx.std_slt @std_slt_1 : i32, i32, i1
// CHECK-DAG:      %std_slt_0.left, %std_slt_0.right, %std_slt_0.out = calyx.std_slt @std_slt_0 : i32, i32, i1
// CHECK-DAG:      %while_0_arg2_reg.in, %while_0_arg2_reg.write_en, %while_0_arg2_reg.clk, %while_0_arg2_reg.reset, %while_0_arg2_reg.out, %while_0_arg2_reg.done = calyx.register @while_0_arg2_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %while_0_arg1_reg.in, %while_0_arg1_reg.write_en, %while_0_arg1_reg.clk, %while_0_arg1_reg.reset, %while_0_arg1_reg.out, %while_0_arg1_reg.done = calyx.register @while_0_arg1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %while_0_arg0_reg.in, %while_0_arg0_reg.write_en, %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %while_0_arg0_reg.out, %while_0_arg0_reg.done = calyx.register @while_0_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %bb3_arg1_reg.in, %bb3_arg1_reg.write_en, %bb3_arg1_reg.clk, %bb3_arg1_reg.reset, %bb3_arg1_reg.out, %bb3_arg1_reg.done = calyx.register @bb3_arg1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %bb3_arg0_reg.in, %bb3_arg0_reg.write_en, %bb3_arg0_reg.clk, %bb3_arg0_reg.reset, %bb3_arg0_reg.out, %bb3_arg0_reg.done = calyx.register @bb3_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @assign_while_0_init_0  {
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.in = %in0 : i32
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @assign_while_0_init_1  {
// CHECK-NEXT:         calyx.assign %while_0_arg1_reg.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %while_0_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %while_0_arg1_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @assign_while_0_init_2  {
// CHECK-NEXT:         calyx.assign %while_0_arg2_reg.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %while_0_arg2_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %while_0_arg2_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.comb_group @bb0_0  {
// CHECK-NEXT:         calyx.assign %std_slt_0.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_slt_0.right = %in1 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.comb_group @bb0_1  {
// CHECK-NEXT:         calyx.assign %std_slt_1.left = %while_0_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_slt_1.right = %while_0_arg2_reg.out : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb1_to_bb3  {
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.in = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %bb3_arg1_reg.in = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %bb3_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_0.left = %while_0_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %while_0_arg2_reg.out : i32
// CHECK-NEXT:         %0 = comb.and %bb3_arg1_reg.done, %bb3_arg0_reg.done : i1
// CHECK-NEXT:         calyx.group_done %0 ? %true : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb2_to_bb3  {
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.in = %std_add_1.out : i32
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %bb3_arg1_reg.in = %std_add_1.out : i32
// CHECK-NEXT:         calyx.assign %bb3_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_1.left = %while_0_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_add_1.right = %while_0_arg2_reg.out : i32
// CHECK-NEXT:         %0 = comb.and %bb3_arg1_reg.done, %bb3_arg0_reg.done : i1
// CHECK-NEXT:         calyx.group_done %0 ? %true : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @assign_while_0_latch  {
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.in = %std_add_2.out : i32
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %while_0_arg1_reg.in = %bb3_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %while_0_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %while_0_arg2_reg.in = %bb3_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %while_0_arg2_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_2.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_add_2.right = %in2 : i32
// CHECK-NEXT:         %0 = comb.and %while_0_arg2_reg.done, %while_0_arg1_reg.done, %while_0_arg0_reg.done : i1
// CHECK-NEXT:         calyx.group_done %0 ? %true : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %while_0_arg2_reg.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.par {
// CHECK-NEXT:           calyx.enable @assign_while_0_init_0
// CHECK-NEXT:           calyx.enable @assign_while_0_init_1
// CHECK-NEXT:           calyx.enable @assign_while_0_init_2
// CHECK-NEXT:         }
// CHECK-NEXT:         calyx.while %std_slt_0.out with @bb0_0  {
// CHECK-NEXT:           calyx.seq  {
// CHECK-NEXT:             calyx.if %std_slt_1.out with @bb0_1  {
// CHECK-NEXT:               calyx.seq  {
// CHECK-NEXT:                 calyx.enable @bb1_to_bb3
// CHECK-NEXT:               }
// CHECK-NEXT:             } else  {
// CHECK-NEXT:               calyx.seq  {
// CHECK-NEXT:                 calyx.enable @bb2_to_bb3
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:             calyx.enable @assign_while_0_latch
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %cst = arith.constant 0 : i32
    %0:3 = scf.while (%arg3 = %arg0, %arg4 = %cst, %arg5 = %cst) : (i32, i32, i32) -> (i32, i32, i32) {
      %1 = arith.cmpi slt, %arg3, %arg1 : i32
      scf.condition(%1) %arg3, %arg4, %arg5 : i32, i32, i32
    } do {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
      %1:2 = scf.execute_region -> (i32, i32) {
        %4 = arith.cmpi slt, %arg4, %arg5 : i32
        cf.cond_br %4, ^bb0, ^bb1
      ^bb0:
        %3 = arith.addi %arg4, %arg5 : i32
        scf.yield %3, %3 : i32, i32
      ^bb1:
        %5 = arith.addi %arg4, %arg5 : i32
        scf.yield %5, %5 : i32, i32
      }
      %2 = arith.addi %arg3, %arg2 : i32
      scf.yield %2, %1#0, %1#1 : i32, i32, i32
    }
    return %0#2 : i32
  }
}

// -----

// Test an executeRegion with multiple yields.

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%in0: i32, %in1: i32, %in2: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %c0_i32 = hw.constant 0 : i32
// CHECK-DAG:      %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
// CHECK-DAG:      %std_sub_0.left, %std_sub_0.right, %std_sub_0.out = calyx.std_sub @std_sub_0 : i32, i32, i32
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %std_slt_1.left, %std_slt_1.right, %std_slt_1.out = calyx.std_slt @std_slt_1 : i32, i32, i1
// CHECK-DAG:      %std_slt_0.left, %std_slt_0.right, %std_slt_0.out = calyx.std_slt @std_slt_0 : i32, i32, i1
// CHECK-DAG:      %while_0_arg2_reg.in, %while_0_arg2_reg.write_en, %while_0_arg2_reg.clk, %while_0_arg2_reg.reset, %while_0_arg2_reg.out, %while_0_arg2_reg.done = calyx.register @while_0_arg2_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %while_0_arg1_reg.in, %while_0_arg1_reg.write_en, %while_0_arg1_reg.clk, %while_0_arg1_reg.reset, %while_0_arg1_reg.out, %while_0_arg1_reg.done = calyx.register @while_0_arg1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %while_0_arg0_reg.in, %while_0_arg0_reg.write_en, %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %while_0_arg0_reg.out, %while_0_arg0_reg.done = calyx.register @while_0_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %bb3_arg1_reg.in, %bb3_arg1_reg.write_en, %bb3_arg1_reg.clk, %bb3_arg1_reg.reset, %bb3_arg1_reg.out, %bb3_arg1_reg.done = calyx.register @bb3_arg1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %bb3_arg0_reg.in, %bb3_arg0_reg.write_en, %bb3_arg0_reg.clk, %bb3_arg0_reg.reset, %bb3_arg0_reg.out, %bb3_arg0_reg.done = calyx.register @bb3_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.group @assign_while_0_init_0  {
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.in = %in0 : i32
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @assign_while_0_init_1  {
// CHECK-NEXT:         calyx.assign %while_0_arg1_reg.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %while_0_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %while_0_arg1_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @assign_while_0_init_2  {
// CHECK-NEXT:         calyx.assign %while_0_arg2_reg.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %while_0_arg2_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %while_0_arg2_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.comb_group @bb0_0  {
// CHECK-NEXT:         calyx.assign %std_slt_0.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_slt_0.right = %in1 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.comb_group @bb0_1  {
// CHECK-NEXT:         calyx.assign %std_slt_1.left = %while_0_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_slt_1.right = %while_0_arg2_reg.out : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb1_to_bb3  {
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.in = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %bb3_arg1_reg.in = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %bb3_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_0.left = %while_0_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %while_0_arg2_reg.out : i32
// CHECK-NEXT:         %0 = comb.and %bb3_arg1_reg.done, %bb3_arg0_reg.done : i1
// CHECK-NEXT:         calyx.group_done %0 ? %true : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb2_to_bb3  {
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.in = %std_sub_0.out : i32
// CHECK-NEXT:         calyx.assign %bb3_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %bb3_arg1_reg.in = %std_sub_0.out : i32
// CHECK-NEXT:         calyx.assign %bb3_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_sub_0.left = %while_0_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_sub_0.right = %while_0_arg2_reg.out : i32
// CHECK-NEXT:         %0 = comb.and %bb3_arg1_reg.done, %bb3_arg0_reg.done : i1
// CHECK-NEXT:         calyx.group_done %0 ? %true : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @assign_while_0_latch  {
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.in = %std_add_1.out : i32
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %while_0_arg1_reg.in = %bb3_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %while_0_arg1_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %while_0_arg2_reg.in = %bb3_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %while_0_arg2_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_1.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_add_1.right = %in2 : i32
// CHECK-NEXT:         %0 = comb.and %while_0_arg2_reg.done, %while_0_arg1_reg.done, %while_0_arg0_reg.done : i1
// CHECK-NEXT:         calyx.group_done %0 ? %true : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %while_0_arg2_reg.out : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.par  {
// CHECK-NEXT:           calyx.enable @assign_while_0_init_0
// CHECK-NEXT:           calyx.enable @assign_while_0_init_1
// CHECK-NEXT:           calyx.enable @assign_while_0_init_2
// CHECK-NEXT:         }
// CHECK-NEXT:         calyx.while %std_slt_0.out with @bb0_0  {
// CHECK-NEXT:           calyx.seq  {
// CHECK-NEXT:             calyx.if %std_slt_1.out with @bb0_1  {
// CHECK-NEXT:               calyx.seq  {
// CHECK-NEXT:                 calyx.enable @bb1_to_bb3
// CHECK-NEXT:               }
// CHECK-NEXT:             } else  {
// CHECK-NEXT:               calyx.seq  {
// CHECK-NEXT:                 calyx.enable @bb2_to_bb3
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:             calyx.enable @assign_while_0_latch
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         calyx.enable @ret_assign_0
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %cst = arith.constant 0 : i32
    %0:3 = scf.while (%arg3 = %arg0, %arg4 = %cst, %arg5 = %cst) : (i32, i32, i32) -> (i32, i32, i32) {
      %1 = arith.cmpi slt, %arg3, %arg1 : i32
      scf.condition(%1) %arg3, %arg4, %arg5 : i32, i32, i32
    } do {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
      %1:2 = scf.execute_region -> (i32, i32) {
        %4 = arith.cmpi slt, %arg4, %arg5 : i32
        cf.cond_br %4, ^bb0, ^bb1
      ^bb0:
        %3 = arith.addi %arg4, %arg5 : i32
        scf.yield %3, %3 : i32, i32
      ^bb1:
        %5 = arith.subi %arg4, %arg5 : i32
        scf.yield %5, %5 : i32, i32
      }
      %2 = arith.addi %arg3, %arg2 : i32
      scf.yield %2, %1#0, %1#1 : i32, i32, i32
    }
    return %0#2 : i32
  }
}

// -----

// Test control flow where the conditional is computed as a sequence of
// combinational operations. We expect all combinational logic in this sequence
// to be inlined into the combinational group. This also tests multiple returns.

// CHECK:     module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:  calyx.component @main(%in0: i32, %in1: i32, %in2: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %std_ge_0.left, %std_ge_0.right, %std_ge_0.out = calyx.std_ge @std_ge_0 : i32, i32, i1
// CHECK-DAG:      %std_add_1.left, %std_add_1.right, %std_add_1.out = calyx.std_add @std_add_1 : i32, i32, i32
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires  {
// CHECK-NEXT:       calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:       calyx.comb_group @bb0_2  {
// CHECK-NEXT:         calyx.assign %std_ge_0.left = %std_add_1.out : i32
// CHECK-NEXT:         calyx.assign %std_ge_0.right = %in2 : i32
// CHECK-NEXT:         calyx.assign %std_add_1.left = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.left = %in0 : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %in1 : i32
// CHECK-NEXT:         calyx.assign %std_add_1.right = %in1 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_0  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %in1 : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @ret_assign_1  {
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.in = %in2 : i32
// CHECK-NEXT:         calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control  {
// CHECK-NEXT:       calyx.seq  {
// CHECK-NEXT:         calyx.if %std_ge_0.out with @bb0_2  {
// CHECK-NEXT:           calyx.seq  {
// CHECK-NEXT:             calyx.enable @ret_assign_0
// CHECK-NEXT:           }
// CHECK-NEXT:         } else  {
// CHECK-NEXT:           calyx.seq  {
// CHECK-NEXT:             calyx.enable @ret_assign_1
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main(%a0 : i32, %a1 : i32, %a2 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.addi %0, %a1 : i32
    %b = arith.cmpi uge, %1, %a2 : i32
    cf.cond_br %b, ^bb1, ^bb2
  ^bb1:
    return %a1 : i32
  ^bb2:
    return %a2 : i32
  }
}

// -----

// Test control flow order of While Loops w/ other operations in same basic block.
// The ordering of the loops & operatioins should be maintained

// CHECK: module attributes {calyx.entrypoint = "main"} {
// CHECK-LABEL:   calyx.component @main(%clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%done: i1 {done}) {
// CHECK-DAG:      %true = hw.constant true
// CHECK-DAG:      %c0_i32 = hw.constant 0 : i32
// CHECK-DAG:      %c1_i32 = hw.constant 1 : i32
// CHECK-DAG:      %c38_i32 = hw.constant 38 : i32
// CHECK-DAG:      %std_slice_2.in, %std_slice_2.out = calyx.std_slice @std_slice_2 : i32, i6
// CHECK-DAG:      %std_slice_1.in, %std_slice_1.out = calyx.std_slice @std_slice_1 : i32, i6
// CHECK-DAG:      %std_slice_0.in, %std_slice_0.out = calyx.std_slice @std_slice_0 : i32, i6
// CHECK-DAG:      %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:      %std_slt_0.left, %std_slt_0.right, %std_slt_0.out = calyx.std_slt @std_slt_0 : i32, i32, i1
// CHECK-DAG:      %load_0_reg.in, %load_0_reg.write_en, %load_0_reg.clk, %load_0_reg.reset, %load_0_reg.out, %load_0_reg.done = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:      %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.clk, %mem_0.read_data, %mem_0.done = calyx.memory @mem_0 <[38] x 32> [6] {external = true} : i6, i32, i1, i1, i32, i1
// CHECK-DAG:      %while_0_arg0_reg.in, %while_0_arg0_reg.write_en, %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %while_0_arg0_reg.out, %while_0_arg0_reg.done = calyx.register @while_0_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     calyx.wires {
// CHECK-NEXT:       calyx.group @assign_while_0_init_0 {
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb0_0 {
// CHECK-NEXT:         calyx.assign %std_slice_2.in = %c0_i32 : i32
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_2.out : i6
// CHECK-NEXT:         calyx.assign %load_0_reg.in = %mem_0.read_data : i32
// CHECK-NEXT:         calyx.assign %load_0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %load_0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.comb_group @bb0_1 {
// CHECK-NEXT:         calyx.assign %std_slt_0.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_slt_0.right = %c38_i32 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb0_2 {
// CHECK-NEXT:         calyx.assign %std_slice_1.in = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_1.out : i6
// CHECK-NEXT:         calyx.assign %mem_0.write_data = %load_0_reg.out : i32
// CHECK-NEXT:         calyx.assign %mem_0.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %mem_0.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @assign_while_0_latch {
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.in = %std_add_0.out : i32
// CHECK-NEXT:         calyx.assign %while_0_arg0_reg.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %std_add_0.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:         calyx.assign %std_add_0.right = %c1_i32 : i32
// CHECK-NEXT:         calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @bb0_4 {
// CHECK-NEXT:         calyx.assign %std_slice_0.in = %c1_i32 : i32
// CHECK-NEXT:         calyx.assign %mem_0.addr0 = %std_slice_0.out : i6
// CHECK-NEXT:         calyx.assign %mem_0.write_data = %load_0_reg.out : i32
// CHECK-NEXT:         calyx.assign %mem_0.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %mem_0.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control {
// CHECK-NEXT:       calyx.seq {
// CHECK-NEXT:         calyx.enable @bb0_0
// CHECK-NEXT:         calyx.enable @assign_while_0_init_0
// CHECK-NEXT:         calyx.while %std_slt_0.out with @bb0_1 {
// CHECK-NEXT:           calyx.seq {
// CHECK-NEXT:             calyx.enable @bb0_2
// CHECK-NEXT:             calyx.enable @assign_while_0_latch
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         calyx.enable @bb0_4
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {toplevel}
// CHECK-NEXT: }
module {
  func.func @main() {
    // declaring variables  
    %c38 = arith.constant 38 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<38xi32>

    // load memory 
    %0 = memref.load %alloca[%c0] : memref<38xi32>
    // while loop 
    %1 = scf.while (%arg0 = %c0) : (index) -> index {
      %2 = arith.cmpi slt, %arg0, %c38 : index
      scf.condition(%2) %arg0 : index
    } do {
    ^bb0(%arg0: index):
      memref.store %0, %alloca[%arg0]: memref<38xi32>
      %2 = arith.addi %arg0, %c1 : index
      scf.yield %2 : index
    }
    // store memory 
    memref.store %0, %alloca[%c1]: memref<38xi32>
    return
  }
}
