// RUN: circt-opt %s -lower-scf-to-calyx -split-input-file | FileCheck %s

// CHECK:     calyx.program "minimal"
// CHECK:       calyx.component @minimal
// CHECK-DAG:     %[[TRUE:.+]] = hw.constant true
// CHECK-DAG:     %[[C0:.+]] = hw.constant 0 : i64
// CHECK-DAG:     %[[C10:.+]] = hw.constant 10 : i64
// CHECK-DAG:     %[[ADD_LEFT:.+]], %[[ADD_RIGHT:.+]], %[[ADD_OUT:.+]] = calyx.std_add
// CHECK-DAG:     %[[LT_LEFT:.+]], %[[LT_RIGHT:.+]], %[[LT_OUT:.+]] = calyx.std_lt
// CHECK-DAG:     %[[ITER_ARG_IN:.+]], %[[ITER_ARG_EN:.+]], %while_0_arg0_reg.clk, %while_0_arg0_reg.reset, %[[ITER_ARG_OUT:.+]], %[[ITER_ARG_DONE:.+]] = calyx.register
// CHECK:         calyx.wires
// CHECK-DAG:       calyx.group @[[INIT_GROUP:.+]] {
// CHECK-DAG:         calyx.assign %[[ITER_ARG_IN]] = %[[C0]] : i64
// CHECK-DAG:         calyx.assign %[[ITER_ARG_EN]] = %[[TRUE]] : i1
// CHECK-DAG:         calyx.group_done %[[ITER_ARG_DONE]] : i1
// CHECK-DAG:       calyx.comb_group @[[COND_GROUP:.+]] {
// CHECK-DAG:         calyx.assign %[[LT_LEFT]] = %[[ITER_ARG_OUT]] : i64
// CHECK-DAG:         calyx.assign %[[LT_RIGHT]] = %[[C10]] : i64
// CHECK-DAG:       calyx.group @[[COMPUTE_GROUP:.+]] {
// CHECK-DAG:         calyx.assign %[[ITER_ARG_IN]] = %[[ADD_OUT]] : i64
// CHECK-DAG:         calyx.assign %[[ITER_ARG_EN]] = %[[TRUE]] : i1
// CHECK-DAG:         calyx.assign %[[ADD_LEFT]] = %[[ITER_ARG_OUT]] : i64
// CHECK-DAG:         calyx.assign %[[ADD_RIGHT]] = %c1_i64 : i64
// CHECK-DAG:         calyx.group_done %[[ITER_ARG_DONE]] : i1
// CHECK:         calyx.control
// CHECK-NEXT:      calyx.seq
// CHECK-NEXT:        calyx.par
// CHECK-NEXT:          calyx.enable @[[INIT_GROUP]]
// CHECK-NEXT:        }
// CHECK-NEXT:        calyx.while %[[LT_OUT]] with @[[COND_GROUP]]
// CHECK-NEXT:          calyx.par
// CHECK-NEXT:            calyx.enable @[[COMPUTE_GROUP]]
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
    %0 = staticlogic.pipeline.stage start = 0 {
      %1 = arith.addi %arg0, %c1_i64 : i64
      staticlogic.pipeline.register %1 : i64
    } : i64
    staticlogic.pipeline.terminator iter_args(%0), results() : (i64) -> ()
  }
  return
}

// -----

// CHECK:     calyx.program "dot"
// CHECK:       calyx.component @dot(%[[MEM0_READ:.+]]: i32, {{.+}}, %[[MEM1_READ:.+]]: i32, {{.+}}) -> ({{.+}} %[[MEM0_ADDR:.+]]: i6, {{.+}} %[[MEM1_ADDR:.+]]: i6, {{.+}} %[[OUT:.+]]: i32, {{.+}}) {
// CHECK-DAG:     %[[TRUE:.+]] = hw.constant true
// CHECK-DAG:     %[[C0:.+]] = hw.constant 0 : i32
// CHECK-DAG:     %[[C64:.+]] = hw.constant 64 : i32
// CHECK-DAG:     %[[C1:.+]] = hw.constant 1 : i32
// CHECK-DAG:     %[[SLICE0_IN:.+]], %[[SLICE0_OUT:.+]] = calyx.std_slice @std_slice_0
// CHECK-DAG:     %[[SLICE1_IN:.+]], %[[SLICE1_OUT:.+]] = calyx.std_slice @std_slice_1
// CHECK-DAG:     %[[ADD0_LEFT:.+]], %[[ADD0_RIGHT:.+]], %[[ADD0_OUT:.+]] = calyx.std_add @std_add_0
// CHECK-DAG:     %[[ADD1_LEFT:.+]], %[[ADD1_RIGHT:.+]], %[[ADD1_OUT:.+]] = calyx.std_add @std_add_1
// CHECK-DAG:     %[[MUL_LEFT:.+]], %[[MUL_RIGHT:.+]], %[[MUL_GO:.+]], {{.+}}, {{.+}}, %[[MUL_OUT:.+]], %[[MUL_DONE:.+]] = calyx.std_mult_pipe
// CHECK-DAG:     %[[LT_LEFT:.+]], %[[LT_RIGHT:.+]], %[[LT_OUT:.+]] = calyx.std_lt
// CHECK-DAG:     %[[ITER_ARG0_IN:.+]], %[[ITER_ARG0_EN:.+]], {{.+}}, {{.+}}, %[[ITER_ARG0_OUT:.+]], %[[ITER_ARG0_DONE:.+]] = calyx.register @while_0_arg0_reg
// CHECK-DAG:     %[[ITER_ARG1_IN:.+]], %[[ITER_ARG1_EN:.+]], {{.+}}, {{.+}}, %[[ITER_ARG1_OUT:.+]], %[[ITER_ARG1_DONE:.+]] = calyx.register @while_0_arg1_reg
// CHECK-DAG:     %[[S0_REG0_IN:.+]], %[[S0_REG0_EN:.+]], {{.+}}, {{.+}}, %[[S0_REG0_OUT:.+]], %[[S0_REG0_DONE:.+]] = calyx.register @stage_0_register_0_reg
// CHECK-DAG:     %[[S0_REG1_IN:.+]], %[[S0_REG1_EN:.+]], {{.+}}, {{.+}}, %[[S0_REG1_OUT:.+]], %[[S0_REG1_DONE:.+]] = calyx.register @stage_0_register_1_reg
// CHECK-DAG:     %[[S1_REG0_IN:.+]], %[[S1_REG0_EN:.+]], {{.+}}, {{.+}}, %[[S1_REG0_OUT:.+]], %[[S1_REG0_DONE:.+]] = calyx.register @stage_1_register_0_reg
// CHECK-DAG:     %[[RET_IN:.+]], %[[RET_EN:.+]], {{.+}}, {{.+}}, %[[RET_OUT:.+]], %[[RET_DONE:.+]] = calyx.register @ret_arg0_reg
// CHECK:         calyx.wires
// CHECK-DAG:       calyx.assign %[[OUT]] = %[[RET_OUT]]
// CHECK-DAG:       calyx.group @[[INIT_GROUP0:.+]]  {
// CHECK-DAG:         calyx.assign %[[ITER_ARG0_IN]] = %[[C0]]
// CHECK-DAG:         calyx.assign %[[ITER_ARG0_EN]] = %[[TRUE]]
// CHECK-DAG:         calyx.group_done %[[ITER_ARG0_DONE]]
// CHECK-DAG:       calyx.group @[[INIT_GROUP1:.+]]  {
// CHECK-DAG:         calyx.assign %[[ITER_ARG1_IN]] = %[[C0]]
// CHECK-DAG:         calyx.assign %[[ITER_ARG1_EN]] = %[[TRUE]]
// CHECK-DAG:         calyx.group_done %[[ITER_ARG1_DONE]]
// CHECK-DAG:       calyx.comb_group @[[COND_GROUP:.+]]  {
// CHECK-DAG:         calyx.assign %[[LT_LEFT]] = %[[ITER_ARG0_OUT]]
// CHECK-DAG:         calyx.assign %[[LT_RIGHT]] = %[[C64]]
// CHECK-DAG:       calyx.group @[[S0_GROUP0:.+]]  {
// CHECK-DAG:         calyx.assign %[[SLICE1_IN]] = %[[ITER_ARG0_OUT]]
// CHECK-DAG:         calyx.assign %[[MEM0_ADDR]] = %[[SLICE1_OUT]]
// CHECK-DAG:         calyx.assign %[[S0_REG0_IN]] = %[[MEM0_READ]]
// CHECK-DAG:         calyx.assign %[[S0_REG0_EN]] = %[[TRUE]]
// CHECK-DAG:         calyx.group_done %[[S0_REG0_DONE]]
// CHECK-DAG:       calyx.group @[[S0_GROUP1:.+]]  {
// CHECK-DAG:         calyx.assign %[[SLICE0_IN]] = %[[ITER_ARG0_OUT]]
// CHECK-DAG:         calyx.assign %[[MEM1_ADDR]] = %[[SLICE0_OUT]]
// CHECK-DAG:         calyx.assign %[[S0_REG1_IN]] = %[[MEM1_READ]]
// CHECK-DAG:         calyx.assign %[[S0_REG1_EN]] = %[[TRUE]]
// CHECK-DAG:         calyx.group_done %[[S0_REG1_DONE]]
// CHECK-DAG:       calyx.group @[[S0_GROUP2:.+]]  {
// CHECK-DAG:         calyx.assign %[[ADD0_LEFT]] = %[[ITER_ARG0_OUT]]
// CHECK-DAG:         calyx.assign %[[ADD0_RIGHT]] = %[[C1]]
// CHECK-DAG:         calyx.assign %[[ITER_ARG0_IN]] = %[[ADD0_OUT]]
// CHECK-DAG:         calyx.assign %[[ITER_ARG0_EN]] = %[[TRUE]]
// CHECK-DAG:         calyx.group_done %[[ITER_ARG0_DONE]]
// CHECK-DAG:       calyx.group @[[S1_GROUP0:.+]]  {
// CHECK-DAG:         calyx.assign %[[MUL_LEFT]] = %[[S0_REG0_OUT]]
// CHECK-DAG:         calyx.assign %[[MUL_RIGHT]] = %[[S0_REG1_OUT]]
// CHECK-DAG:         calyx.assign %[[S1_REG0_IN]] = %[[MUL_OUT]]
// CHECK-DAG:         calyx.assign %[[S1_REG0_EN]] = %[[MUL_DONE]]
// CHECK-DAG:         calyx.assign %[[MUL_GO]] = %[[TRUE]]
// CHECK-DAG:         calyx.group_done %[[S1_REG0_DONE]]
// CHECK-DAG:       calyx.group @[[S2_GROUP0:.+]]  {
// CHECK-DAG:         calyx.assign %[[ADD1_LEFT]] = %[[ITER_ARG1_OUT]]
// CHECK-DAG:         calyx.assign %[[ADD1_RIGHT]] = %[[S1_REG0_OUT]]
// CHECK-DAG:         calyx.assign %[[ITER_ARG1_IN]] = %[[ADD1_OUT]]
// CHECK-DAG:         calyx.assign %[[ITER_ARG1_EN]] = %[[TRUE]]
// CHECK-DAG:         calyx.group_done %[[ITER_ARG1_DONE]]
// CHECK-DAG:       calyx.group @[[RET_GROUP:.+]]  {
// CHECK-DAG:         calyx.assign %[[RET_IN]] = %[[ITER_ARG1_OUT]]
// CHECK-DAG:         calyx.assign %[[RET_EN]] = %[[TRUE]]
// CHECK-DAG:         calyx.group_done %[[RET_DONE]]
// CHECK:         calyx.control
// CHECK-NEXT:      calyx.seq  {
// CHECK-NEXT:        calyx.seq  {
// CHECK-NEXT:          calyx.par  {
// CHECK-NEXT:            calyx.enable @[[INIT_GROUP0]]
// CHECK-NEXT:            calyx.enable @[[INIT_GROUP1]]
// CHECK-NEXT:          }
// CHECK-NEXT:          calyx.par  {
// CHECK-NEXT:            calyx.enable @[[S0_GROUP0]]
// CHECK-NEXT:            calyx.enable @[[S0_GROUP1]]
// CHECK-NEXT:            calyx.enable @[[S0_GROUP2]]
// CHECK-NEXT:          }
// CHECK-NEXT:          calyx.par  {
// CHECK-NEXT:            calyx.enable @[[S0_GROUP0]]
// CHECK-NEXT:            calyx.enable @[[S0_GROUP1]]
// CHECK-NEXT:            calyx.enable @[[S0_GROUP2]]
// CHECK-NEXT:            calyx.enable @[[S1_GROUP0]]
// CHECK-NEXT:          }
// CHECK-NEXT:          calyx.while %std_lt_0.out with @bb0_0  {
// CHECK-NEXT:            calyx.par  {
// CHECK-NEXT:              calyx.enable @[[S0_GROUP0]]
// CHECK-NEXT:              calyx.enable @[[S0_GROUP1]]
// CHECK-NEXT:              calyx.enable @[[S0_GROUP2]]
// CHECK-NEXT:              calyx.enable @[[S1_GROUP0]]
// CHECK-NEXT:              calyx.enable @[[S2_GROUP0]]
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:          calyx.par  {
// CHECK-NEXT:            calyx.enable @[[S1_GROUP0]]
// CHECK-NEXT:            calyx.enable @[[S2_GROUP0]]
// CHECK-NEXT:          }
// CHECK-NEXT:          calyx.par  {
// CHECK-NEXT:            calyx.enable @[[S2_GROUP0]]
// CHECK-NEXT:          }
// CHECK-NEXT:          calyx.enable @[[RET_GROUP]]
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
func @dot(%arg0: memref<64xi32>, %arg1: memref<64xi32>) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %0 = staticlogic.pipeline.while II =  1 iter_args(%arg2 = %c0, %arg3 = %c0_i32) : (index, i32) -> i32 {
    %1 = arith.cmpi ult, %arg2, %c64 : index
    staticlogic.pipeline.register %1 : i1
  } do {
    %1:3 = staticlogic.pipeline.stage start = 0  {
      %4 = memref.load %arg0[%arg2] : memref<64xi32>
      %5 = memref.load %arg1[%arg2] : memref<64xi32>
      %6 = arith.addi %arg2, %c1 : index
      staticlogic.pipeline.register %4, %5, %6 : i32, i32, index
    } : i32, i32, index
    %2 = staticlogic.pipeline.stage start = 1  {
      %4 = arith.muli %1#0, %1#1 : i32
      staticlogic.pipeline.register %4 : i32
    } : i32
    %3 = staticlogic.pipeline.stage start = 4  {
      %4 = arith.addi %arg3, %2 : i32
      staticlogic.pipeline.register %4 : i32
    } : i32
    staticlogic.pipeline.terminator iter_args(%1#2, %3), results(%3) : (index, i32) -> i32
  }
  return %0 : i32
}
