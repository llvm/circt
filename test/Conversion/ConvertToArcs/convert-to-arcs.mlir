// RUN: circt-opt %s --convert-to-arcs | FileCheck %s

// CHECK-LABEL: hw.module @Empty
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }
hw.module @Empty() {
}


// CHECK-LABEL: hw.module @Passthrough(
// CHECK-SAME:    [[TMP:%.+]]: i4) -> (z: i4) {
// CHECK-NEXT:    hw.output [[TMP]]
// CHECK-NEXT:  }
hw.module @Passthrough(%a: i4) -> (z: i4) {
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
hw.module @CombOnly(%i0: i4, %i1: i4) -> (z: i4) {
  // CHECK-NEXT: [[TMP:%.+]] = arc.state @CombOnly_arc(%i0, %i1) lat 0
  // CHECK-NEXT: hw.output [[TMP]]
  %0 = comb.add %i0, %i1 : i4
  %1 = comb.xor %0, %i0 : i4
  %2 = comb.xor %0, %i1 : i4
  %3 = comb.mul %1, %2 : i4
  hw.output %3 : i4
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
hw.module @Pipeline(%clock: i1, %i0: i4, %i1: i4) -> (z: i4) {
  // CHECK-NEXT: [[S0:%.+]] = arc.state @Pipeline_arc(%i0, %i1) clock %clock lat 1
  // CHECK-NEXT: [[S1:%.+]] = arc.state @Pipeline_arc_0([[S0]], %i0) clock %clock lat 1
  // CHECK-NEXT: [[S2:%.+]] = arc.state @Pipeline_arc_1([[S1]], %i1) lat 0
  // CHECK-NEXT: hw.output [[S2]]
  %0 = comb.add %i0, %i1 : i4
  %1 = seq.compreg %0, %clock : i4
  %2 = comb.xor %1, %i0 : i4
  %3 = seq.compreg %2, %clock : i4
  %4 = comb.mul %3, %i1 : i4
  hw.output %4 : i4
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
hw.module @FactorOutCommonOps(%clock: i1, %i0: i4, %i1: i4) -> (o0: i4, o1: i4) {
  // CHECK-DAG: [[T0:%.+]] = arc.state @FactorOutCommonOps_arc_1(%i0, %i1) lat 0
  %0 = comb.add %i0, %i1 : i4
  // CHECK-DAG: [[T1:%.+]] = arc.state @FactorOutCommonOps_arc([[T0]], %i0) clock %clock lat 1
  // CHECK-DAG: [[T2:%.+]] = arc.state @FactorOutCommonOps_arc_0([[T0]], %i1) clock %clock lat 1
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
hw.module @SplitAtInstance(%a: i4) -> (z: i4) {
  // CHECK-DAG: [[T0:%.+]] = arc.state @SplitAtInstance_arc(%a) lat 0
  // CHECK-DAG: [[T1:%.+]] = hw.instance "x" @SplitAtInstance2(a: [[T0]]: i4)
  // CHECK-DAG: [[T2:%.+]] = arc.state @SplitAtInstance_arc_0([[T1]]) lat 0
  %0 = comb.mul %a, %a : i4
  %1 = hw.instance "x" @SplitAtInstance2(a: %0: i4) -> (z: i4)
  %2 = comb.shl %1, %1 : i4
  // CHECK-NEXT: hw.output [[T2]]
  hw.output %2 : i4
}
// CHECK-NEXT: }

hw.module.extern private @SplitAtInstance2(%a: i4) -> (z: i4)
