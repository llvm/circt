// RUN: circt-opt %s --arc-remove-unused-arc-arguments | FileCheck %s

// CHECK-LABEL: arc.define @OneOfThreeUsed(%arg0: i1)
arc.define @OneOfThreeUsed(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  // CHECK-NEXT: arc.output %arg0
  arc.output %arg1 : i1
}

// CHECK: @test1
hw.module @test1 (%arg0: i1, %arg1: i1, %arg2: i1, %clock: i1) -> (out: i1) {
  // CHECK-NEXT: arc.state @OneOfThreeUsed(%arg1) clock %clock lat 1 : (i1) -> i1
  %0 = arc.state @OneOfThreeUsed(%arg0, %arg1, %arg2) clock %clock lat 1 : (i1, i1, i1) -> i1
  hw.output %0 : i1
}

// CHECK-LABEL: arc.define @NoArgsToRemove()
arc.define @NoArgsToRemove() -> i1 {
  %0 = hw.constant 0 : i1
  arc.output %0 : i1
}

// CHECK: @test2
hw.module @test2 () -> (out: i1) {
  // CHECK-NEXT: arc.state @NoArgsToRemove() lat 0 : () -> i1
  %0 = arc.state @NoArgsToRemove() lat 0 : () -> i1
  hw.output %0 : i1
}
