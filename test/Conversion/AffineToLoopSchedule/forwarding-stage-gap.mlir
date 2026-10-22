// RUN: circt-opt -convert-affine-to-loopschedule %s | FileCheck %s

// A combinational result may need forwarding across time slots with no
// scheduled operation while another operand is produced by a multi-cycle op.

// CHECK-LABEL: func @forward_across_empty_stages
// CHECK: %[[STAGE0:.*]]:3 = loopschedule.pipeline.stage start = 0
// CHECK-DAG: %[[SHORT:.*]] = arith.addi
// CHECK-DAG: %[[LONG:.*]] = arith.muli
// CHECK: loopschedule.register %[[SHORT]], %[[LONG]], %{{.*}}
// CHECK: %[[STAGE1:.*]] = loopschedule.pipeline.stage start = 1
// CHECK-NEXT: loopschedule.register %[[STAGE0]]#0
// CHECK: %[[STAGE2:.*]] = loopschedule.pipeline.stage start = 2
// CHECK-NEXT: loopschedule.register %[[STAGE1]]
// CHECK: loopschedule.pipeline.stage start = 3
// CHECK: arith.addi %[[STAGE2]], %[[STAGE0]]#1 : i32
func.func @forward_across_empty_stages(%out: memref<16xi32>,
                                       %a: i32, %b: i32) {
  affine.for %i = 0 to 16 {
    %short = arith.addi %a, %b : i32
    %long = arith.muli %a, %b : i32
    %sum = arith.addi %short, %long : i32
    affine.store %sum, %out[%i] : memref<16xi32>
  }
  return
}
