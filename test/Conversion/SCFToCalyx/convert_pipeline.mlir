// RUN: circt-opt %s -lower-scf-to-calyx | FileCheck %s

// CHECK:     calyx.program "minimal"
// CHECK:       calyx.component @minimal
// CHECK-DAG:     %[[TRUE:.+]] = hw.constant true
// CHECK-DAG:     %[[C0:.+]] = hw.constant 0 : i64
// CHECK-DAG:     %[[C10:.+]] = hw.constant 10 : i64
// CHECK-DAG:     %[[ADD_LEFT:.+]], %[[ADD_RIGHT:.+]], %[[ADD_OUT:.+]] = calyx.std_add
// CHECK-DAG:     %[[LT_LEFT:.+]], %[[LT_RIGHT:.+]], %[[LT_OUT:.+]] = calyx.std_lt
// CHECK-DAG:     %[[ITER_ARG_IN:.+]], %[[ITER_ARG_EN:.+]], %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %[[ITER_ARG_OUT:.+]], %[[ITER_ARG_DONE:.+]] = calyx.register
// CHECK:         calyx.wires
// CHECK-DAG:       calyx.group @assign_while_0_init
// CHECK-DAG:         calyx.assign %[[ITER_ARG_IN]] = %[[C0]] : i64
// CHECK-DAG:         calyx.assign %[[ITER_ARG_EN]] = %[[TRUE]] : i1
// CHECK-DAG:         calyx.group_done %[[ITER_ARG_DONE]] : i1
// CHECK-DAG:       calyx.comb_group @bb0_0
// CHECK-DAG:         calyx.assign %[[LT_LEFT]] = %[[ITER_ARG_OUT]] : i64
// CHECK-DAG:         calyx.assign %[[LT_RIGHT]] = %[[C10]] : i64
// CHECK-DAG:       calyx.group @assign_while_0_latch
// CHECK-DAG:         calyx.assign %[[ITER_ARG_IN]] = %[[ADD_OUT]] : i64
// CHECK-DAG:         calyx.assign %[[ITER_ARG_EN]] = %[[TRUE]] : i1
// CHECK-DAG:         calyx.assign %[[ADD_LEFT]] = %[[ITER_ARG_OUT]] : i64
// CHECK-DAG:         calyx.assign %[[ADD_RIGHT]] = %c1_i64 : i64
// CHECK-DAG:         calyx.group_done %[[ITER_ARG_DONE]] : i1
// CHECK:         calyx.control
// CHECK-NEXT:      calyx.seq
// CHECK-NEXT:        calyx.enable @assign_while_0_init
// CHECK-NEXT:        calyx.while %[[LT_OUT]] with @bb0_0
// CHECK-NEXT:          calyx.par
// CHECK-NEXT:            calyx.enable @assign_while_0_latch
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
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
