// RUN: circt-opt %s -lower-scf-to-calyx | FileCheck %s

// CHECK:     calyx.program "minimal"
// CHECK:       calyx.component @minimal
// CHECK-DAG:     %[[TRUE:.+]] = hw.constant true
// CHECK-DAG:     %[[C0:.+]] = hw.constant 0 : i64
// CHECK-DAG:     %[[C10:.+]] = hw.constant 10 : i64
// CHECK-DAG:     %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i64, i64, i64
// CHECK-DAG:     %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i64, i64, i1
// CHECK-DAG:     %while_0_arg0_reg.in, %while_0_arg0_reg.write_en, %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %while_0_arg0_reg.out, %while_0_arg0_reg.done = calyx.register @while_0_arg0_reg : i64, i1, i1, i1, i64, i1
// CHECK:         calyx.wires
// CHECK-DAG:       calyx.group @assign_while_0_init
// CHECK-DAG:         calyx.assign %while_0_arg0_reg.in = %[[C0]] : i64
// CHECK-DAG:         calyx.assign %while_0_arg0_reg.write_en = %[[TRUE]] : i1
// CHECK-DAG:         calyx.group_done %while_0_arg0_reg.done : i1
// CHECK-DAG:       calyx.comb_group @bb0_0
// CHECK-DAG:         calyx.assign %std_lt_0.left = %while_0_arg0_reg.out : i64
// CHECK-DAG:         calyx.assign %std_lt_0.right = %[[C10]] : i64
// CHECK-DAG:       calyx.group @assign_while_0_latch
// CHECK-DAG:         calyx.assign %while_0_arg0_reg.in = %std_add_0.out : i64
// CHECK-DAG:         calyx.assign %while_0_arg0_reg.write_en = %[[TRUE]] : i1
// CHECK-DAG:         calyx.group_done %while_0_arg0_reg.done : i1
// CHECK:         calyx.control
// CHECK-NEXT       calyx.seq
// CHECK-NEXT         calyx.enable @assign_while_0_init
// CHECK-NEXT         calyx.while %std_lt_0.out with @bb0_0
// CHECK-NEXT           calyx.seq
// CHECK-NEXT             calyx.enable @assign_while_0_latch
// CHECK-NEXT           }
// CHECK-NEXT         }
// CHECK-NEXT       }
// CHECK-NEXT     }
func @minimal() {
  %c0_i64 = arith.constant 0 : i64
  %c10_i64 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64
  staticlogic.pipeline.while II =  1 iter_args(%arg0 = %c0_i64) : (i64) -> () {
    %0 = arith.cmpi ult, %arg0, %c10_i64 : i64
    staticlogic.pipeline.register %0 : i1
  } do {
    %0 = staticlogic.pipeline.stage  {
      %1 = arith.addi %arg0, %c1_i64 : i64
      staticlogic.pipeline.register %1 : i64
    } : i64
    staticlogic.pipeline.terminator iter_args(%0), results() : (i64) -> ()
  }
  return
}
