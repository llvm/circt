// RUN: circt-opt %s --convert-to-arcs | FileCheck %s

// CHECK-LABEL: hw.module @Empty
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }
hw.module @Empty() {
}


// CHECK-LABEL: hw.module @Passthrough(
// CHECK-SAME:    in [[TMP:%.+]] : i4, out z : i4) {
// CHECK-NEXT:    hw.output [[TMP]]
// CHECK-NEXT:  }
hw.module @Passthrough(in %a: i4, out z: i4) {
  hw.output %a : i4
}


// CHECK-LABEL: arc.define @CombOnly_arc(
// CHECK-NEXT:    comb.add
// CHECK-NEXT:    comb.xor
// CHECK-NEXT:    comb.xor
// CHECK-NEXT:    comb.mul
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @CombOnly
hw.module @CombOnly(in %i0: i4, in %i1: i4, out z: i4) {
  // CHECK-NEXT: [[TMP:%.+]] = arc.call @CombOnly_arc(%i0, %i1)
  // CHECK-NEXT: hw.output [[TMP]]
  %0 = comb.add %i0, %i1 : i4
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %3 = comb.mul %1, %2 : i4
  hw.output %3 : i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @SplitAtConstants_arc(
// CHECK-NEXT:    comb.add
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @SplitAtConstants
hw.module @SplitAtConstants(out z: i4) {
  // CHECK-NEXT: %c1_i4 = hw.constant 1
  // CHECK-NEXT: [[TMP:%.+]] = arc.call @SplitAtConstants_arc(%c1_i4)
  // CHECK-NEXT: hw.output [[TMP]]
  %c1_i4 = hw.constant 1 : i4
  %0 = comb.add %c1_i4, %c1_i4 : i4
  hw.output %0 : i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @Pipeline_arc(
// CHECK-NEXT:    comb.add
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @Pipeline_arc_0(
// CHECK-NEXT:    comb.xor
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @Pipeline_arc_1(
// CHECK-NEXT:    comb.mul
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @Pipeline
hw.module @Pipeline(in %clock: !seq.clock, in %i0: i4, in %i1: i4, out z: i4) {
  // CHECK-NEXT: [[S0:%.+]] = arc.state @Pipeline_arc(%i0, %i1) clock %clock latency 1
  // CHECK-NEXT: [[S1:%.+]] = arc.state @Pipeline_arc_0([[S0]], %i0) clock %clock latency 1
  // CHECK-NEXT: [[S2:%.+]] = arc.call @Pipeline_arc_1([[S1]], %i1)
  // CHECK-NEXT: hw.output [[S2]]
  %0 = comb.add %i0, %i1 : i4
  %1 = seq.compreg %0, %clock : i4
  %2 = comb.xor %1, %i0 : i4
  %3 = seq.compreg %2, %clock : i4
  %4 = comb.mul %3, %i1 : i4
  hw.output %4 : i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @Reshuffling_arc(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    arc.output %arg0, %arg1
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @Reshuffling_arc_0(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    arc.output %arg0, %arg1
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @Reshuffling
hw.module @Reshuffling(in %clockA: !seq.clock, in %clockB: !seq.clock, out z0: i4, out z1: i4, out z2: i4, out z3: i4) {
  // CHECK-NEXT: hw.instance "x" @Reshuffling2()
  // CHECK-NEXT: arc.state @Reshuffling_arc(%x.z0, %x.z1) clock %clockA latency 1
  // CHECK-NEXT: arc.state @Reshuffling_arc_0(%x.z2, %x.z3) clock %clockB latency 1
  // CHECK-NEXT: hw.output
  %x.z0, %x.z1, %x.z2, %x.z3 = hw.instance "x" @Reshuffling2() -> (z0: i4, z1: i4, z2: i4, z3: i4)
  %4 = seq.compreg %x.z0, %clockA : i4
  %5 = seq.compreg %x.z1, %clockA : i4
  %6 = seq.compreg %x.z2, %clockB : i4
  %7 = seq.compreg %x.z3, %clockB : i4
  hw.output %4, %5, %6, %7 : i4, i4, i4, i4
}
// CHECK-NEXT: }

hw.module.extern private @Reshuffling2(out z0: i4, out z1: i4, out z2: i4, out z3: i4)

// CHECK-LABEL: arc.define @ReshufflingInit_arc(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    arc.output %arg0, %arg1
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @ReshufflingInit_arc_0(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    arc.output %arg0, %arg1
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @ReshufflingInit
hw.module @ReshufflingInit(in %clockA: !seq.clock, in %clockB: !seq.clock, out z0: i4, out z1: i4, out z2: i4, out z3: i4) {
  // CHECK-NEXT: hw.instance "x" @Reshuffling2()
  // CHECK-NEXT:  %[[INITIAL:.+]]:3 = seq.initial() {
  // CHECK-NEXT:    %c1_i4 = hw.constant 1 : i4
  // CHECK-NEXT:    %c2_i4 = hw.constant 2 : i4
  // CHECK-NEXT:    %c3_i4 = hw.constant 3 : i4
  // CHECK-NEXT:    seq.yield %c1_i4, %c2_i4, %c3_i4 : i4, i4, i4
  // CHECK-NEXT:  } : () -> (!seq.immutable<i4>, !seq.immutable<i4>, !seq.immutable<i4>)
  // CHECK-NEXT: %[[C1:.+]] = seq.from_immutable %[[INITIAL]]#0
  // CHECK-NEXT: %[[C2:.+]] = seq.from_immutable %[[INITIAL]]#1
  // CHECK-NEXT: %[[C3:.+]] = seq.from_immutable %[[INITIAL]]#2
  // CHECK-NEXT: %[[C0:.+]] = hw.constant 0 : i4
  // CHECK-NEXT: arc.state @ReshufflingInit_arc(%x.z0, %x.z1) clock %clockA initial (%[[C0]], %[[C1]] : i4, i4) latency 1
  // CHECK-NEXT: arc.state @ReshufflingInit_arc_0(%x.z2, %x.z3) clock %clockB initial (%[[C2]], %[[C3]] : i4, i4) latency 1
  // CHECK-NEXT: hw.output

  %x.z0, %x.z1, %x.z2, %x.z3 = hw.instance "x" @Reshuffling2() -> (z0: i4, z1: i4, z2: i4, z3: i4)
  %4 = seq.compreg %x.z0, %clockA : i4
  %init0, %init1, %init2 = seq.initial () {
    %cst1 = hw.constant 1 : i4
    %cst2 = hw.constant 2 : i4
    %cst3 = hw.constant 3 : i4
    seq.yield %cst1, %cst2, %cst3 : i4, i4, i4
  } : () -> (!seq.immutable<i4>, !seq.immutable<i4>, !seq.immutable<i4>)
  %5 = seq.compreg %x.z1, %clockA initial %init0 : i4
  %6 = seq.compreg %x.z2, %clockB initial %init1 : i4
  %7 = seq.compreg %x.z3, %clockB initial %init2 : i4
  hw.output %4, %5, %6, %7 : i4, i4, i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @FactorOutCommonOps_arc(
// CHECK-NEXT:    comb.xor
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @FactorOutCommonOps_arc_0(
// CHECK-NEXT:    comb.mul
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @FactorOutCommonOps_arc_1(
// CHECK-NEXT:    comb.add
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @FactorOutCommonOps
hw.module @FactorOutCommonOps(in %clock: !seq.clock, in %i0: i4, in %i1: i4, out o0: i4, out o1: i4) {
  // CHECK-DAG: [[T0:%.+]] = arc.call @FactorOutCommonOps_arc_1(%i0, %i1)
  %0 = comb.add %i0, %i1 : i4
  // CHECK-DAG: [[T1:%.+]] = arc.state @FactorOutCommonOps_arc([[T0]], %i0) clock %clock latency 1
  // CHECK-DAG: [[T2:%.+]] = arc.state @FactorOutCommonOps_arc_0([[T0]], %i1) clock %clock latency 1
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.mul %0, %i1 : i4
  %3 = seq.compreg %1, %clock : i4
  %4 = seq.compreg %2, %clock : i4
  // CHECK-NEXT: hw.output [[T1]], [[T2]]
  hw.output %3, %4 : i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: arc.define @SplitAtInstance_arc(
// CHECK-NEXT:    comb.mul
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: arc.define @SplitAtInstance_arc_0(
// CHECK-NEXT:    comb.shl
// CHECK-NEXT:    arc.output
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @SplitAtInstance(
hw.module @SplitAtInstance(in %a: i4, out z: i4) {
  // CHECK-DAG: [[T0:%.+]] = arc.call @SplitAtInstance_arc(%a)
  // CHECK-DAG: [[T1:%.+]] = hw.instance "x" @SplitAtInstance2(a: [[T0]]: i4)
  // CHECK-DAG: [[T2:%.+]] = arc.call @SplitAtInstance_arc_0([[T1]])
  %0 = comb.mul %a, %a : i4
  %1 = hw.instance "x" @SplitAtInstance2(a: %0: i4) -> (z: i4)
  %2 = comb.shl %1, %1 : i4
  // CHECK-NEXT: hw.output [[T2]]
  hw.output %2 : i4
}
// CHECK-NEXT: }

hw.module.extern private @SplitAtInstance2(in %a: i4, out z: i4)


// CHECK-LABEL: hw.module @AbsorbNames
hw.module @AbsorbNames(in %clock: !seq.clock) {
  // CHECK-NEXT: %x.z0, %x.z1 = hw.instance "x" @AbsorbNames2()
  // CHECK-NEXT: arc.state @AbsorbNames_arc(%x.z0, %x.z1) clock %clock latency 1
  // CHECK-SAME:   {names = ["myRegA", "myRegB"]}
  // CHECK-NEXT: hw.output
  %x.z0, %x.z1 = hw.instance "x" @AbsorbNames2() -> (z0: i4, z1: i4)
  %myRegA = seq.compreg %x.z0, %clock : i4
  %myRegB = seq.compreg %x.z1, %clock : i4
}
// CHECK-NEXT: }

hw.module.extern @AbsorbNames2(out z0: i4, out z1: i4)

// CHECK:   arc.define @[[TRIVIAL_ARC:.+]]([[ARG0:%.+]]: i4)
// CHECK-NEXT:     arc.output [[ARG0]]
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @Trivial(
hw.module @Trivial(in %clock: !seq.clock, in %i0: i4, in %reset: i1, out out: i4) {
  // CHECK: [[RES0:%.+]] = arc.state @[[TRIVIAL_ARC]](%i0) clock %clock reset %reset latency 1 {names = ["foo"]
  // CHECK-NEXT: hw.output [[RES0:%.+]]
  %0 = hw.constant 0 : i4
  %foo = seq.compreg %i0, %clock reset %reset, %0 : i4
  hw.output %foo : i4
}
// CHECK-NEXT: }

// CHECK:   arc.define @[[TRIVIALINIT_ARC:.+]]([[ARG0:%.+]]: i4)
// CHECK-NEXT:     arc.output [[ARG0]]
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @TrivialWithInit(
hw.module @TrivialWithInit(in %clock: !seq.clock, in %i0: i4, in %reset: i1, out out: i4) {
  // CHECK: %[[INIT:.+]] = seq.initial() {
  // CHECK: %[[CAST:.+]] = seq.from_immutable %[[INIT]]
  // CHECK: [[RES0:%.+]] = arc.state @[[TRIVIALINIT_ARC]](%i0) clock %clock reset %reset initial (%[[CAST]] : i4) latency 1 {names = ["foo"]
  // CHECK-NEXT: hw.output [[RES0:%.+]]
  %0 = hw.constant 0 : i4
  %init = seq.initial() {
    %cst2 = hw.constant 2 : i4
    seq.yield %cst2: i4
  } : () -> !seq.immutable<i4>
  %foo = seq.compreg %i0, %clock reset %reset, %0 initial %init: i4
  hw.output %foo : i4
}
// CHECK-NEXT: }

// CHECK-NEXT:   arc.define @[[NONTRIVIAL_ARC_0:.+]]([[ARG0_1:%.+]]: i4)
// CHECK-NEXT:     arc.output [[ARG0_1]]
// CHECK-NEXT:  }

// CHECK-NEXT:   arc.define @[[NONTRIVIAL_ARC_1:.+]]([[ARG0_2:%.+]]: i4)
// CHECK-NEXT:     arc.output [[ARG0_2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @NonTrivial(
hw.module @NonTrivial(in %clock: !seq.clock, in %i0: i4, in %reset1: i1, in %reset2: i1, out out1: i4, out out2: i4) {
  // CHECK: [[RES2:%.+]] = arc.state @[[NONTRIVIAL_ARC_0]](%i0) clock %clock reset %reset1 latency 1 {names = ["foo"]
  // CHECK-NEXT: [[RES3:%.+]] = arc.state @[[NONTRIVIAL_ARC_1]](%i0) clock %clock reset %reset2 latency 1 {names = ["bar"]
  // CHECK-NEXT: hw.output [[RES2]], [[RES3]]
  %0 = hw.constant 0 : i4
  %foo = seq.compreg %i0, %clock reset %reset1, %0 : i4
  %bar = seq.compreg %i0, %clock reset %reset2, %0 : i4
  hw.output %foo, %bar : i4, i4
}
// CHECK-NEXT: }

hw.module @ObserveWires(in %in1: i32, in %in2: i32, out out: i32) {
  %c-1_i32 = hw.constant -1 : i32
  %0 = comb.and %in1, %in2 : i32
  arc.tap %0 {name = "z"} : i32
  %1 = comb.xor %0, %c-1_i32 : i32
  hw.output %1 : i32
}
//       CHECK: arc.define @[[ARC_NAME:[a-zA-Z0-9_]+]]
//       CHECK:   [[V0:%.+]] = comb.and
//       CHECK:   [[V1:%.+]] = comb.xor [[V0]], {{.*}} :
//       CHECK:   arc.output [[V0]], [[V1]] :
//       CHECK: hw.module @ObserveWires
//   CHECK-DAG:   [[RES:%.+]]:2 = arc.call @[[ARC_NAME]]({{.*}}) :
//   CHECK-DAG:   arc.tap [[RES]]#0 {name = "z"} : i32
//       CHECK:   hw.output %0#1 : i32

// CHECK: arc.define [[ARC_ADD:@OpsWithRegions.+]](
// CHECK-NEXT: comb.add

// CHECK: arc.define [[ARC_SUB:@OpsWithRegions.+]](
// CHECK-NEXT: comb.sub

// CHECK: hw.module @OpsWithRegions
hw.module @OpsWithRegions(in %a: i42, in %b: i42, in %c: i42, in %d: i42, out z: i42) {
  // CHECK-DAG: [[ADD:%.+]] = arc.call [[ARC_ADD]](%a, %b)
  %0 = comb.add %a, %b : i42
  // CHECK-DAG: [[COMB:%.+]] = llhd.combinational -> i42 {
  // CHECK-DAG:   [[MUL:%.+]] = comb.mul [[ADD]], %c
  // CHECK-DAG:   llhd.yield [[MUL]]
  %1 = llhd.combinational -> i42 {
    %3 = comb.mul %0, %c : i42
    llhd.yield %3 : i42
  }
  // CHECK-DAG: [[SUB:%.+]] = arc.call [[ARC_SUB]]([[COMB]], %d)
  %2 = comb.sub %1, %d : i42
  // CHECK: hw.output [[SUB]]
  hw.output %2 : i42
}
