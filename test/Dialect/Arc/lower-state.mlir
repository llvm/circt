// RUN: circt-opt %s --arc-lower-state | FileCheck %s

// CHECK-LABEL: arc.model @Empty
// CHECK-NEXT:  ^bb0(%arg0: !arc.storage):
// CHECK-NEXT:  }
hw.module @Empty() {
}

// CHECK-LABEL: arc.model @InputsAndOutputs
hw.module @InputsAndOutputs(in %a: i42, in %b: i17, out c: i42, out d: i17) {
  %0 = comb.add %a, %a : i42
  %1 = comb.add %b, %b : i17
  hw.output %0, %1 : i42, i17
  // CHECK-NEXT: (%arg0: !arc.storage):
  // CHECK-NEXT: [[INA:%.+]] = arc.root_input "a", %arg0 : (!arc.storage) -> !arc.state<i42>
  // CHECK-NEXT: [[INB:%.+]] = arc.root_input "b", %arg0 : (!arc.storage) -> !arc.state<i17>
  // CHECK-NEXT: [[OUTA:%.+]] = arc.root_output "c", %arg0 : (!arc.storage) -> !arc.state<i42>
  // CHECK-NEXT: [[OUTB:%.+]] = arc.root_output "d", %arg0 : (!arc.storage) -> !arc.state<i17>

  // CHECK-NEXT: arc.passthrough {
  // CHECK-NEXT:   [[A:%.+]] = arc.state_read [[INA]] : <i42>
  // CHECK-NEXT:   [[TMP:%.+]] = comb.add [[A]], [[A]] : i42
  // CHECK-NEXT:   arc.state_write [[OUTA]] = [[TMP]] : <i42>
  // CHECK-NEXT:   [[B:%.+]] = arc.state_read [[INB]] : <i17>
  // CHECK-NEXT:   [[TMP:%.+]] = comb.add [[B]], [[B]] : i17
  // CHECK-NEXT:   arc.state_write [[OUTB]] = [[TMP]] : <i17>
  // CHECK-NEXT: }
}

// CHECK-LABEL: arc.model @State
hw.module @State(in %clk: !seq.clock, in %en: i1, in %en2: i1) {
  %gclk = seq.clock_gate %clk, %en, %en2
  %3 = arc.state @DummyArc(%6) clock %clk latency 1 : (i42) -> i42
  %4 = arc.state @DummyArc(%5) clock %gclk latency 1 : (i42) -> i42
  %5 = comb.add %3, %3 : i42
  %6 = comb.add %4, %4 : i42
  // CHECK-NEXT: (%arg0: !arc.storage):
  // CHECK-NEXT: [[INCLK:%.+]] = arc.root_input "clk", %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[INEN:%.+]] = arc.root_input "en", %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[INEN2:%.+]] = arc.root_input "en2", %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[CLK_OLD:%.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[S0:%.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>
  // CHECK-NEXT: [[S1:%.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>

  // CHECK-NEXT: [[TMP2:%.+]] = arc.state_read [[INCLK]] : <i1>
  // CHECK-NEXT: arc.state_write [[CLK_OLD]] = [[TMP2]] : <i1>
  // CHECK-NEXT: [[TMP1:%.+]] = arc.state_read [[CLK_OLD]] : <i1>
  // CHECK-NEXT: [[TMP3:%.+]] = comb.icmp ne [[TMP1]], [[TMP2]] : i1
  // CHECK-NEXT: [[TMP4:%.+]] = comb.and [[TMP3]], [[TMP2]] : i1

  // CHECK-NEXT: arc.clock_tree [[TMP4]] {
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S1]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[TMP0]], [[TMP0]]
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.call @DummyArc([[TMP1]]) : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S0]] = [[TMP2]] : <i42>
  // CHECK-NEXT:   [[EN:%.+]] = arc.state_read [[INEN]] : <i1>
  // CHECK-NEXT:   [[EN2:%.+]] = arc.state_read [[INEN2]] : <i1>
  // CHECK-NEXT:   [[TMP3:%.+]] = comb.or [[EN]], [[EN2]] : i1
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S0]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[TMP0]], [[TMP0]]
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.call @DummyArc([[TMP1]]) : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S1]] = [[TMP2]] if [[TMP3]] : <i42>
  // CHECK-NEXT: }
}

// CHECK-LABEL: arc.model @State2
hw.module @State2(in %clk: !seq.clock) {
  %3 = arc.state @DummyArc(%3) clock %clk latency 1 : (i42) -> i42
  %4 = arc.state @DummyArc(%4) clock %clk latency 1 : (i42) -> i42
  // CHECK-NEXT: (%arg0: !arc.storage):
  // CHECK-NEXT: [[INCLK:%.+]] = arc.root_input "clk", %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[CLK_OLD:%.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[S0]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>
  // CHECK-NEXT: [[S1]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>

  // CHECK-NEXT: [[TMP2:%.+]] = arc.state_read [[INCLK]] : <i1>
  // CHECK-NEXT: arc.state_write [[CLK_OLD]] = [[TMP2]] : <i1>
  // CHECK-NEXT: [[TMP1:%.+]] = arc.state_read [[CLK_OLD]] : <i1>
  // CHECK-NEXT: [[TMP3:%.+]] = comb.icmp ne [[TMP1]], [[TMP2]] : i1
  // CHECK-NEXT: [[TMP4:%.+]] = comb.and [[TMP3]], [[TMP2]] : i1

  // CHECK-NEXT: arc.clock_tree [[TMP4]] {
  // CHECK-NEXT:   [[TMP0:%.+]] = arc.state_read [[S0:%.+]] : <i42>
  // CHECK-NEXT:   [[TMP1:%.+]] = arc.call @DummyArc([[TMP0]]) : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S0]] = [[TMP1]] : <i42>
  // CHECK-NEXT:   [[TMP2:%.+]] = arc.state_read [[S1:%.+]] : <i42>
  // CHECK-NEXT:   [[TMP3:%.+]] = arc.call @DummyArc([[TMP2]]) : (i42) -> i42
  // CHECK-NEXT:   arc.state_write [[S1]] = [[TMP3]] : <i42>
  // CHECK-NEXT: }
}

arc.define @DummyArc(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

// CHECK-LABEL: arc.model @NonMaskedMemoryWrite
hw.module @NonMaskedMemoryWrite(in %clk0: !seq.clock) {
  %c0_i2 = hw.constant 0 : i2
  %c9001_i42 = hw.constant 9001 : i42
  %mem = arc.memory <4 x i42, i2>
  arc.memory_write_port %mem, @identity(%c0_i2, %c9001_i42) clock %clk0 latency 1 : <4 x i42, i2>, i2, i42

  // CHECK-NEXT: (%arg0: !arc.storage):
  // CHECK-NEXT: [[INCLK:%.+]] = arc.root_input "clk0", %arg0 : (!arc.storage) -> !arc.state<i1>
  // CHECK-NEXT: [[MEM:%.+]] = arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<4 x i42, i2>
  // CHECK-NEXT: [[CLK_OLD:%.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i1>

  // CHECK-NEXT: [[TMP2:%.+]] = arc.state_read [[INCLK]] : <i1>
  // CHECK-NEXT: arc.state_write [[CLK_OLD]] = [[TMP2]] : <i1>
  // CHECK-NEXT: [[TMP1:%.+]] = arc.state_read [[CLK_OLD]] : <i1>
  // CHECK-NEXT: [[TMP3:%.+]] = comb.icmp ne [[TMP1]], [[TMP2]] : i1
  // CHECK-NEXT: [[TMP4:%.+]] = comb.and [[TMP3]], [[TMP2]] : i1

  // CHECK-NEXT: arc.clock_tree [[TMP4]] {
  // CHECK:        [[RES:%.+]]:2 = arc.call @identity(%c0_i2, %c9001_i42) : (i2, i42) -> (i2, i42)
  // CHECK:        arc.memory_write [[MEM]][[[RES]]#0], [[RES]]#1 : <4 x i42, i2>
  // CHECK-NEXT: }
}
arc.define @identity(%arg0: i2, %arg1: i42) -> (i2, i42) {
  arc.output %arg0, %arg1 : i2, i42
}

// CHECK-LABEL: arc.model @lowerMemoryReadPorts
hw.module @lowerMemoryReadPorts(out out0: i42, out out1: i42) {
  %c0_i2 = hw.constant 0 : i2
  %mem = arc.memory <4 x i42, i2>
  // CHECK: arc.memory_read {{%.+}}[%c0_i2] : <4 x i42, i2>
  %0 = arc.memory_read_port %mem[%c0_i2] : <4 x i42, i2>
  // CHECK: func.call @arcWithMemoryReadsIsLowered
  %1 = arc.call @arcWithMemoryReadsIsLowered(%mem) : (!arc.memory<4 x i42, i2>) -> i42
  hw.output %0, %1 : i42, i42
}

// CHECK-LABEL: func.func @arcWithMemoryReadsIsLowered(%arg0: !arc.memory<4 x i42, i2>) -> i42 attributes {llvm.linkage = #llvm.linkage<internal>}
arc.define @arcWithMemoryReadsIsLowered(%mem: !arc.memory<4 x i42, i2>) -> i42 {
  %c0_i2 = hw.constant 0 : i2
  // CHECK: arc.memory_read {{%.+}}[%c0_i2] : <4 x i42, i2>
  %0 = arc.memory_read_port %mem[%c0_i2] : <4 x i42, i2>
  // CHECK-NEXT: return
  arc.output %0 : i42
}

// CHECK-LABEL:  arc.model @maskedMemoryWrite
hw.module @maskedMemoryWrite(in %clk: !seq.clock) {
  %true = hw.constant true
  %c0_i2 = hw.constant 0 : i2
  %c9001_i42 = hw.constant 9001 : i42
  %c1010_i42 = hw.constant 1010 : i42
  %mem = arc.memory <4 x i42, i2>
  arc.memory_write_port %mem, @identity2(%c0_i2, %c9001_i42, %true, %c1010_i42) clock %clk enable mask latency 1 : <4 x i42, i2>, i2, i42, i1, i42
}
arc.define @identity2(%arg0: i2, %arg1: i42, %arg2: i1, %arg3: i42) -> (i2, i42, i1, i42) {
  arc.output %arg0, %arg1, %arg2, %arg3 : i2, i42, i1, i42
}
// CHECK:      %c9001_i42 = hw.constant 9001 : i42
// CHECK:      %c1010_i42 = hw.constant 1010 : i42
// CHECK:      [[RES:%.+]]:4 = arc.call @identity2(%c0_i2, %c9001_i42, %true, %c1010_i42) : (i2, i42, i1, i42) -> (i2, i42, i1, i42)
// CHECK:      [[RD:%.+]] = arc.memory_read [[MEM:%.+]][[[RES]]#0] : <4 x i42, i2>
// CHECK:      %c-1_i42 = hw.constant -1 : i42
// CHECK:      [[NEG_MASK:%.+]] = comb.xor bin [[RES]]#3, %c-1_i42 : i42
// CHECK:      [[OLD_MASKED:%.+]] = comb.and bin [[NEG_MASK]], [[RD]] : i42
// CHECK:      [[NEW_MASKED:%.+]] = comb.and bin [[RES]]#3, [[RES]]#1 : i42
// CHECK:      [[DATA:%.+]] = comb.or bin [[OLD_MASKED]], [[NEW_MASKED]] : i42
// CHECK:      arc.memory_write [[MEM]][[[RES]]#0], [[DATA]] if [[RES]]#2 : <4 x i42, i2>

// CHECK-LABEL: arc.model @Taps
hw.module @Taps() {
  // CHECK-NOT: arc.tap
  // CHECK-DAG: [[VALUE:%.+]] = hw.constant 0 : i42
  // CHECK-DAG: [[STATE:%.+]] = arc.alloc_state %arg0 tap {name = "myTap"}
  // CHECK-DAG: arc.state_write [[STATE]] = [[VALUE]]
  %c0_i42 = hw.constant 0 : i42
  arc.tap %c0_i42 {name = "myTap"} : i42
}

// CHECK-LABEL: arc.model @MaterializeOpsWithRegions
hw.module @MaterializeOpsWithRegions(in %clk0: !seq.clock, in %clk1: !seq.clock, out z: i42) {
  %true = hw.constant true
  %c19_i42 = hw.constant 19 : i42
  %0 = scf.if %true -> (i42) {
    scf.yield %c19_i42 : i42
  } else {
    %c42_i42 = hw.constant 42 : i42
    scf.yield %c42_i42 : i42
  }

  // CHECK:      arc.passthrough {
  // CHECK-NEXT:   %true = hw.constant true
  // CHECK-NEXT:   %c19_i42 = hw.constant 19
  // CHECK-NEXT:   [[TMP:%.+]] = scf.if %true -> (i42) {
  // CHECK-NEXT:     scf.yield %c19_i42
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %c42_i42 = hw.constant 42
  // CHECK-NEXT:     scf.yield %c42_i42
  // CHECK-NEXT:   }
  // CHECK-NEXT:   arc.state_write
  // CHECK-NEXT: }

  // CHECK:      [[CLK0:%.+]] = arc.state_read %in_clk0
  // CHECK:      [[TMP:%.+]] = comb.and {{%.+}}, [[CLK0]]
  // CHECK-NEXT: arc.clock_tree [[TMP]] {
  // CHECK-NEXT:   %true = hw.constant true
  // CHECK-NEXT:   %c19_i42 = hw.constant 19
  // CHECK-NEXT:   [[TMP:%.+]] = scf.if %true -> (i42) {
  // CHECK-NEXT:     scf.yield %c19_i42
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %c42_i42 = hw.constant 42
  // CHECK-NEXT:     scf.yield %c42_i42
  // CHECK-NEXT:   }
  // CHECK-NEXT:   arc.call @DummyArc([[TMP]])
  // CHECK-NEXT:   arc.state_write
  // CHECK-NEXT: }

  // CHECK:      [[CLK1:%.+]] = arc.state_read %in_clk1
  // CHECK:      [[TMP:%.+]] = comb.and {{%.+}}, [[CLK1]]
  // CHECK-NEXT: arc.clock_tree [[TMP]] {
  // CHECK-NEXT:   %true = hw.constant true
  // CHECK-NEXT:   %c19_i42 = hw.constant 19
  // CHECK-NEXT:   [[TMP:%.+]] = scf.if %true -> (i42) {
  // CHECK-NEXT:     scf.yield %c19_i42
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %c42_i42 = hw.constant 42
  // CHECK-NEXT:     scf.yield %c42_i42
  // CHECK-NEXT:   }
  // CHECK-NEXT:   arc.call @DummyArc([[TMP]])
  // CHECK-NEXT:   arc.state_write
  // CHECK-NEXT: }

  %1 = arc.state @DummyArc(%0) clock %clk0 latency 1 : (i42) -> i42
  %2 = arc.state @DummyArc(%0) clock %clk1 latency 1 : (i42) -> i42
  hw.output %0 : i42
}

arc.define @i1Identity(%arg0: i1) -> i1 {
  arc.output %arg0 : i1
}

arc.define @DummyArc2(%arg0: i42) -> (i42, i42) {
  arc.output %arg0, %arg0 : i42, i42
}

hw.module @stateReset(in %clk: !seq.clock, in %arg0: i42, in %rst: i1, out out0: i42, out out1: i42) {
  %0 = arc.call @i1Identity(%rst) : (i1) -> (i1)
  %1 = arc.call @i1Identity(%rst) : (i1) -> (i1)
  %2, %3 = arc.state @DummyArc2(%arg0) clock %clk enable %0 reset %1 latency 1 : (i42) -> (i42, i42)
  hw.output %2, %3 : i42, i42
}
// CHECK-LABEL: arc.model @stateReset
// CHECK: [[ALLOC1:%.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>
// CHECK: [[ALLOC2:%.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>
// CHECK: arc.clock_tree %{{.*}} {
// CHECK:   [[IN_RST:%.+]] = arc.state_read %in_rst : <i1>
// CHECK:   [[EN:%.+]] = arc.call @i1Identity([[IN_RST]]) : (i1) -> i1
// CHECK:   [[RST:%.+]] = arc.call @i1Identity([[IN_RST]]) : (i1) -> i1
// CHECK:   scf.if [[RST]] {
// CHECK:     arc.state_write [[ALLOC1]] = %c0_i42{{.*}} : <i42>
// CHECK:     arc.state_write [[ALLOC2]] = %c0_i42{{.*}} : <i42>
// CHECK:   } else {
// CHECK:     [[ARG:%.+]] = arc.state_read %in_arg0 : <i42>
// CHECK:     [[STATE:%.+]]:2 = arc.call @DummyArc2([[ARG]]) : (i42) -> (i42, i42)
// CHECK:     arc.state_write [[ALLOC1]] = [[STATE]]#0 if [[EN]] : <i42>
// CHECK:     arc.state_write [[ALLOC2]] = [[STATE]]#1 if [[EN]] : <i42>
// CHECK:   }
// CHECK: }

hw.module @SeparateResets(in %clock: !seq.clock, in %i0: i42, in %rst1: i1, in %rst2: i1, out out1: i42, out out2: i42) {
  %0 = arc.state @DummyArc(%i0) clock %clock reset %rst1 latency 1 {names = ["foo"]} : (i42) -> i42
  %1 = arc.state @DummyArc(%i0) clock %clock reset %rst2 latency 1 {names = ["bar"]} : (i42) -> i42
  hw.output %0, %1 : i42, i42
}

// CHECK-LABEL: arc.model @SeparateResets
// CHECK: [[FOO_ALLOC:%.+]] = arc.alloc_state %arg0 {name = "foo"} : (!arc.storage) -> !arc.state<i42>
// CHECK: [[BAR_ALLOC:%.+]] = arc.alloc_state %arg0 {name = "bar"} : (!arc.storage) -> !arc.state<i42>
// CHECK: arc.clock_tree %{{.*}} {
// CHECK:   [[IN_RST1:%.+]] = arc.state_read %in_rst1 : <i1>
// CHECK:   scf.if [[IN_RST1]] {
// CHECK:     %c0_i42{{.*}} = hw.constant 0 : i42
// CHECK:     arc.state_write [[FOO_ALLOC]] = %c0_i42{{.*}} : <i42>
// CHECK:   } else {
// CHECK:     [[IN_I0:%.+]] = arc.state_read %in_i0 : <i42>
// CHECK:     [[STATE:%.+]] = arc.call @DummyArc([[IN_I0]]) : (i42) -> i42
// CHECK:     arc.state_write [[FOO_ALLOC]] = [[STATE]] : <i42>
// CHECK:   }
// CHECK:   [[IN_RST2:%.+]] = arc.state_read %in_rst2 : <i1>
// CHECK:   scf.if [[IN_RST2]] {
// CHECK:     %c0_i42{{.*}} = hw.constant 0 : i42
// CHECK:     arc.state_write [[BAR_ALLOC]] = %c0_i42{{.*}} : <i42>
// CHECK:   } else {
// CHECK:     [[IN_I0_2:%.+]] = arc.state_read %in_i0 : <i42>
// CHECK:     [[STATE_2:%.+]] = arc.call @DummyArc([[IN_I0_2]]) : (i42) -> i42
// CHECK:     arc.state_write [[BAR_ALLOC]] = [[STATE_2]] : <i42>
// CHECK:   }

// Regression check on worklist producing false positive comb loop errors.
// CHECK-LABEL: @CombLoopRegression
hw.module @CombLoopRegression(in %clk: !seq.clock) {
  %0 = arc.state @CombLoopRegressionArc1(%3, %3) clock %clk latency 1 : (i1, i1) -> i1
  %1, %2 = arc.call @CombLoopRegressionArc2(%0) : (i1) -> (i1, i1)
  %3 = arc.call @CombLoopRegressionArc1(%1, %2) : (i1, i1) -> i1
}
arc.define @CombLoopRegressionArc1(%arg0: i1, %arg1: i1) -> i1 {
  arc.output %arg0 : i1
}
arc.define @CombLoopRegressionArc2(%arg0: i1) -> (i1, i1) {
  arc.output %arg0, %arg0 : i1, i1
}

// Regression check for invalid memory port lowering errors.
// CHECK-LABEL: arc.model @MemoryPortRegression
hw.module private @MemoryPortRegression(in %clock: !seq.clock, in %reset: i1, in %in: i3, out x: i3) {
  %0 = arc.memory <2 x i3, i1> {name = "ram_ext"}
  %1 = arc.memory_read_port %0[%3] : <2 x i3, i1>
  arc.memory_write_port %0, @identity3(%3, %in) clock %clock latency 1 : <2 x i3, i1>, i1, i3
  %3 = arc.state @Queue_arc_0(%reset) clock %clock latency 1 : (i1) -> i1
  %4 = arc.call @Queue_arc_1(%1) : (i3) -> i3
  hw.output %4 : i3
}
arc.define @identity3(%arg0: i1, %arg1: i3) -> (i1, i3) {
  arc.output %arg0, %arg1 : i1, i3
}
arc.define @Queue_arc_0(%arg0: i1) -> i1 {
  arc.output %arg0 : i1
}
arc.define @Queue_arc_1(%arg0: i3) -> i3 {
  arc.output %arg0 : i3
}

// CHECK-LABEL: arc.model @BlackBox
hw.module @BlackBox(in %clk: !seq.clock) {
  %0 = arc.state @DummyArc(%2) clock %clk latency 1 : (i42) -> i42
  %1 = comb.and %0, %0 : i42
  %ext.c, %ext.d = hw.instance "ext" @BlackBoxExt(a: %0: i42, b: %1: i42) -> (c: i42, d: i42)
  %2 = comb.or %ext.c, %ext.d : i42
  // CHECK-DAG: [[EXT_A:%.+]] = arc.alloc_state %arg0 {name = "ext/a"}
  // CHECK-DAG: [[EXT_B:%.+]] = arc.alloc_state %arg0 {name = "ext/b"}
  // CHECK-DAG: [[EXT_C:%.+]] = arc.alloc_state %arg0 {name = "ext/c"}
  // CHECK-DAG: [[EXT_D:%.+]] = arc.alloc_state %arg0 {name = "ext/d"}
  // CHECK-DAG: [[STATE:%.+]] = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i42>

  // Clock Tree
  // CHECK-DAG: [[TMP1:%.+]] = arc.state_read [[EXT_C]]
  // CHECK-DAG: [[TMP2:%.+]] = arc.state_read [[EXT_D]]
  // CHECK-DAG: [[TMP3:%.+]] = comb.or [[TMP1]], [[TMP2]]
  // CHECK-DAG: [[TMP4:%.+]] = arc.call @DummyArc([[TMP3]])
  // CHECK-DAG: arc.state_write [[STATE]] = [[TMP4]]

  // Passthrough
  // CHECK-DAG: [[TMP1:%.+]] = arc.state_read [[STATE]]
  // CHECK-DAG: [[TMP2:%.+]] = comb.and [[TMP1]], [[TMP1]]
  // CHECK-DAG: arc.state_write [[EXT_A]] = [[TMP1]]
  // CHECK-DAG: arc.state_write [[EXT_B]] = [[TMP2]]
}
// CHECK-NOT: hw.module.extern private @BlackBoxExt
hw.module.extern private @BlackBoxExt(in %a: i42, in %b: i42, out c: i42, out d: i42)


func.func private @func(%arg0: i32, %arg1: i32) -> i32
// CHECK-LABEL: arc.model @adder
hw.module @adder(in %clock : i1, in %a : i32, in %b : i32, out c : i32) {
  %0 = seq.to_clock %clock
  %1 = sim.func.dpi.call @func(%a, %b) clock %0 : (i32, i32) -> i32
  // CHECK:      arc.clock_tree
  // CHECK-NEXT:   %[[A:.+]] = arc.state_read %in_a : <i32>
  // CHECK-NEXT:   %[[B:.+]] = arc.state_read %in_b : <i32>
  // CHECK-NEXT:   %[[RESULT:.+]] = func.call @func(%6, %7) : (i32, i32) -> i32
  hw.output %1 : i32
}
