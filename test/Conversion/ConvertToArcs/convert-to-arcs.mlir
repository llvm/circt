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
  arc.tap %0 {names = ["z"]} : i32
  %1 = comb.xor %0, %c-1_i32 : i32
  hw.output %1 : i32
}
//       CHECK: arc.define @[[ARC_NAME:[a-zA-Z0-9_]+]]
//       CHECK:   [[V0:%.+]] = comb.and
//       CHECK:   [[V1:%.+]] = comb.xor [[V0]], {{.*}} :
//       CHECK:   arc.output [[V0]], [[V1]] :
//       CHECK: hw.module @ObserveWires
//   CHECK-DAG:   [[RES:%.+]]:2 = arc.call @[[ARC_NAME]]({{.*}}) :
//   CHECK-DAG:   arc.tap [[RES]]#0 {names = ["z"]} : i32
//       CHECK:   hw.output %0#1 : i32

// CHECK: arc.define [[ARC_ADD:@OpsWithRegions.+]](
// CHECK-NEXT: comb.add

// CHECK: arc.define [[ARC_SUB:@OpsWithRegions.+]](
// CHECK-NEXT: comb.sub

// CHECK: hw.module @OpsWithRegions
hw.module @OpsWithRegions(in %a: i42, in %b: i42, in %c: i42, in %d: i42, out z: i42) {
  // CHECK-DAG: [[ADD:%.+]] = arc.call [[ARC_ADD]](%a, %b)
  %0 = comb.add %a, %b : i42
  // CHECK-DAG: [[COMB:%.+]] = arc.execute ([[ADD]], %c : i42, i42) -> (i42) {
  // CHECK-DAG: ^bb0(%arg0: i42, %arg1: i42):
  // CHECK-DAG:   [[MUL:%.+]] = comb.mul %arg0, %arg1
  // CHECK-DAG:   arc.output [[MUL]]
  %1 = llhd.combinational -> i42 {
    %3 = comb.mul %0, %c : i42
    llhd.yield %3 : i42
  }
  // CHECK-DAG: [[SUB:%.+]] = arc.call [[ARC_SUB]]([[COMB]], %d)
  %2 = comb.sub %1, %d : i42
  // CHECK: hw.output [[SUB]]
  hw.output %2 : i42
}

// LLHD time ops survive `convert-to-arcs` unchanged. The arc outliner may
// pull purely combinational time ops into separate arcs.
// CHECK-LABEL: arc.define @TimeOpsPassthrough_arc
//       CHECK:   llhd.time_to_int
// CHECK-LABEL: hw.module @TimeOpsPassthrough
//   CHECK-DAG:   llhd.constant_time <42fs, 0d>
//   CHECK-DAG:   llhd.int_to_time
//   CHECK-DAG:   llhd.current_time
hw.module @TimeOpsPassthrough(in %a : i64, out t : i64) {
  %0 = llhd.constant_time <42fs, 0d>
  %1 = llhd.int_to_time %a
  %2 = llhd.current_time
  %3 = llhd.time_to_int %0
  hw.output %3 : i64
}

// comb.and uses a non-pure operation coming from llhd.prb
// CHECK-LABEL: arc.define @ProbeIsNonPure_arc
//   CHECK:   comb.and
//   CHECK:   arc.output
// CHECK-LABEL: hw.module @ProbeIsNonPure
//   CHECK:       [[RES:%.+]] = arc.call @ProbeIsNonPure
//   CHECK:   hw.output [[RES]]
hw.module @ProbeIsNonPure(in %in : i1, out out : i1) {
  %false = hw.constant false
  %r1 = llhd.sig %false : <i1>
  %1 = llhd.prb %r1 : i1
  %2 = comb.and %in, %1 : i1
  hw.output %2 : i1
}

// An assignment-only cycle must be recognized without any llhd.drv. Preserve
// the immediate assignments when committing the stable results.
// CHECK-LABEL: hw.module @SignalFixedPoint(
// CHECK: %[[INIT:.+]] = hw.constant 0 : i8
// CHECK: %[[TRUE:.+]] = hw.constant true
// CHECK: %[[A:.+]] = llhd.sig %[[INIT]] : <i8>
// CHECK: %[[B:.+]] = llhd.sig %[[INIT]] : <i8>
// CHECK: %[[OLD_B:.+]] = llhd.prb %[[B]] : i8
// CHECK: %[[OLD_A:.+]] = llhd.prb %[[A]] : i8
// CHECK: %[[RESULT:.+]]:2 = arc.fixed_point iter_args(%[[OLD_B]], %[[OLD_A]]) inputs(%[[TRUE]]) : (i8, i8, i1) -> (i8, i8) {
// CHECK: ^bb0(%[[PREV_B:.+]]: i8, %[[PREV_A:.+]]: i8, %[[ENABLE:.+]]: i1):
// CHECK:   %[[NEXT_B:.+]] = comb.mux %[[ENABLE]], %[[PREV_A]], %[[PREV_B]] : i8
// CHECK:   %[[NEXT_A:.+]] = comb.mux %[[ENABLE]], %[[PREV_B]], %[[PREV_A]] : i8
// CHECK:   arc.output %[[NEXT_B]], %[[NEXT_A]] : i8, i8
// CHECK: }
// CHECK: llhd.assign %[[B]], %[[RESULT]]#0 : i8
// CHECK: llhd.assign %[[A]], %[[RESULT]]#1 : i8
hw.module @SignalFixedPoint() {
  %false = hw.constant 0 : i8
  %true = hw.constant true
  %a = llhd.sig %false : <i8>
  %b = llhd.sig %false : <i8>
  %read_a = llhd.prb %a : i8
  %read_b = llhd.prb %b : i8
  llhd.assign %b, %read_a if %true : i8
  llhd.assign %a, %read_b if %true : i8
}

// Module inputs are block arguments without defining operations. Capture both
// a computation input and a drive enable as invariant region arguments. Export
// the computed drive value as an additional result for its external user.
// CHECK-LABEL: hw.module @SignalFixedPointModuleInputs(
// CHECK: %[[SIGNAL:.+]] = llhd.sig %{{.+}} : <i8>
// CHECK: %[[OLD:.+]] = llhd.prb %[[SIGNAL]] : i8
// CHECK: %[[RESULT:.+]]:2 = arc.fixed_point iter_args(%[[OLD]]) inputs(%input, %enable) : (i8, i8, i1) -> (i8, i8) {
// CHECK-NEXT: ^bb0(%[[PREV:.+]]: i8, %[[INPUT:.+]]: i8, %[[ENABLE:.+]]: i1):
// CHECK-NEXT:   %[[VALUE:.+]] = comb.and %[[PREV]], %[[INPUT]] : i8
// CHECK-NEXT:   %[[NEXT:.+]] = comb.mux %[[ENABLE]], %[[VALUE]], %[[PREV]] : i8
// CHECK-NEXT:   arc.output %[[NEXT]], %[[VALUE]] : i8, i8
// CHECK-NEXT: }
// CHECK: llhd.drv %[[SIGNAL]], %[[RESULT]]#0 after %{{.+}} : i8
// CHECK: hw.output %[[RESULT]]#0, %[[RESULT]]#1 : i8, i8
hw.module @SignalFixedPointModuleInputs(in %input: i8, in %enable: i1, out result: i8, out computed: i8) {
  %zero = hw.constant 0 : i8
  %delay = llhd.constant_time <0ns, 0d>
  %signal = llhd.sig %zero : <i8>
  %previous = llhd.prb %signal : i8
  %next = comb.and %previous, %input : i8
  llhd.drv %signal, %next after %delay if %enable : i8
  hw.output %previous, %next : i8, i8
}

// A graph region permits uses before definitions. Once signal feedback becomes
// block arguments, order both the feedback computation and its downstream
// observer by SSA dependencies before placing them in an SSACFG region.
// CHECK-LABEL: hw.module @SignalFixedPointReversedOrder
// CHECK: %[[ONE:.+]] = hw.constant 1 : i8
// CHECK: %[[RESULT:.+]] = arc.fixed_point iter_args(%{{.+}}) inputs(%[[ONE]]) : (i8, i8) -> i8 {
// CHECK-NEXT: ^bb0(%[[PREV:.+]]: i8, %[[CAPTURE:.+]]: i8):
// CHECK-NEXT: %[[SUM:.+]] = comb.add %[[PREV]], %[[CAPTURE]] : i8
// CHECK-NEXT: %[[NEXT:.+]] = comb.and %[[SUM]], %[[CAPTURE]] : i8
// CHECK-NEXT: %[[OBSERVED:.+]] = comb.xor %[[NEXT]], %[[CAPTURE]] : i8
// CHECK-NEXT: arc.coroutine.instance @ObserveOne(%[[OBSERVED]]) sensitive [true] : (i8) -> ()
// CHECK-NEXT: arc.output %[[NEXT]] : i8
// CHECK-NEXT: }
// CHECK: llhd.drv %{{.+}}, %[[RESULT]] after %{{.+}} : i8
// CHECK: hw.output %[[RESULT]] : i8
hw.module @SignalFixedPointReversedOrder(out result: i8) {
  %zero = hw.constant 0 : i8
  %one = hw.constant 1 : i8
  %delay = llhd.constant_time <0ns, 0d>
  %signal = llhd.sig %zero : <i8>
  %previous = llhd.prb %signal : i8
  arc.coroutine.instance @ObserveOne(%observed) sensitive [true] : (i8) -> ()
  %observed = comb.xor %next, %one : i8
  %next = comb.and %sum, %one : i8
  %sum = comb.add %previous, %one : i8
  llhd.drv %signal, %next after %delay : i8
  hw.output %previous : i8
}

// Two separate signal cycles share a downstream observer. Merge their
// evaluation regions so the observer sees both feedback values on every
// iteration and is instantiated only once. One cycle uses a zero-delay drive
// and the other an immediate assignment.
// CHECK-LABEL: hw.module @SharedObserver
// CHECK-NOT: arc.coroutine.instance
// CHECK: %[[RESULT:.+]]:2 = arc.fixed_point iter_args(%{{.+}}, %{{.+}}) : (i8, i8) -> (i8, i8) {
// CHECK-NEXT: ^bb0(%[[A:.+]]: i8, %[[B:.+]]: i8):
// CHECK-NEXT: arc.coroutine.instance @ObserveTwo(%[[A]], %[[B]]) sensitive [true, true] : (i8, i8) -> ()
// CHECK-NEXT: arc.output %[[A]], %[[B]] : i8, i8
// CHECK-NEXT: }
// CHECK-NOT: arc.fixed_point
// CHECK-NOT: arc.coroutine.instance
// CHECK: hw.output %[[RESULT]]#0, %[[RESULT]]#1 : i8, i8
hw.module @SharedObserver(out a: i8, out b: i8) {
  %zero = hw.constant 0 : i8
  %delay = llhd.constant_time <0ns, 0d>
  %a = llhd.sig %zero : <i8>
  %b = llhd.sig %zero : <i8>
  %read_a = llhd.prb %a : i8
  %read_b = llhd.prb %b : i8
  arc.coroutine.instance @ObserveTwo(%read_a, %read_b) sensitive [true, true] : (i8, i8) -> ()
  llhd.drv %a, %read_a after %delay : i8
  llhd.assign %b, %read_b : i8
  hw.output %read_a, %read_b : i8, i8
}

// Independent observers do not force unrelated cycles into one region.
// CHECK-LABEL: hw.module @IndependentObservers
// CHECK-NOT: arc.coroutine.instance
// CHECK: arc.fixed_point iter_args(%{{.+}}) : (i8) -> i8 {
// CHECK-NEXT: ^bb0(%[[A:.+]]: i8):
// CHECK-NEXT: arc.coroutine.instance @ObserveOne(%[[A]]) sensitive [true] : (i8) -> ()
// CHECK-NEXT: arc.output %[[A]] : i8
// CHECK-NEXT: }
// CHECK-NOT: arc.coroutine.instance
// CHECK: arc.fixed_point iter_args(%{{.+}}) : (i8) -> i8 {
// CHECK-NEXT: ^bb0(%[[B:.+]]: i8):
// CHECK-NEXT: arc.coroutine.instance @ObserveOne(%[[B]]) sensitive [true] : (i8) -> ()
// CHECK-NEXT: arc.output %[[B]] : i8
// CHECK-NEXT: }
// CHECK-NOT: arc.fixed_point
// CHECK-NOT: arc.coroutine.instance
// CHECK: hw.output
hw.module @IndependentObservers() {
  %zero = hw.constant 0 : i8
  %delay = llhd.constant_time <0ns, 0d>
  %a = llhd.sig %zero : <i8>
  %b = llhd.sig %zero : <i8>
  %read_a = llhd.prb %a : i8
  %read_b = llhd.prb %b : i8
  arc.coroutine.instance @ObserveOne(%read_a) sensitive [true] : (i8) -> ()
  arc.coroutine.instance @ObserveOne(%read_b) sensitive [true] : (i8) -> ()
  llhd.drv %a, %read_a after %delay : i8
  llhd.drv %b, %read_b after %delay : i8
  hw.output
}

arc.coroutine.define @ObserveTwo(%a: i8, %b: i8) -> (i2, i64) {
  %mask = hw.constant 0 : i2
  %never = hw.constant -1 : i64
  arc.coroutine.halt %mask, %never : i2, i64
}

arc.coroutine.define @ObserveOne(%a: i8) -> (i1, i64) {
  %mask = hw.constant false
  %never = hw.constant -1 : i64
  arc.coroutine.halt %mask, %never : i1, i64
}

// A register reset by another register that is converted in the same batch of
// shuffling arcs: the reset operand must refer to the converted register's new
// value, not to a stale/erased reference.
// CHECK-LABEL: hw.module @ResetFromRegister(
hw.module @ResetFromRegister(in %clock: !seq.clock, in %in: i1, in %rst: i1, out out: i1) {
  // CHECK-DAG: [[RST_D:%.+]] = arc.state @{{.+}}(%rst) clock %clock latency 1
  // CHECK-DAG: [[DATA:%.+]] = arc.state @{{.+}}(%in) clock %clock reset [[RST_D]] latency 1
  // CHECK: hw.output [[DATA]]
  %false = hw.constant false
  %rst_d = seq.compreg %rst, %clock : i1
  %data = seq.compreg %in, %clock reset %rst_d, %false : i1
  hw.output %data : i1
}
