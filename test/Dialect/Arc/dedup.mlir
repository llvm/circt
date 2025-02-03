// RUN: circt-opt %s --arc-dedup | FileCheck %s

// CHECK-LABEL: arc.define @SimpleA
arc.define @SimpleA(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {Simple} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @SimpleB
arc.define @SimpleB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {Simple} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @Simple
hw.module @Simple(in %x: i4, in %y: i4, in %clock: !seq.clock) {
  // CHECK-NEXT: arc.call @SimpleA(%x, %y)
  // CHECK-NEXT: arc.call @SimpleA(%y, %x)
  %0 = arc.call @SimpleA(%x, %y) : (i4, i4) -> i4
  %1 = arc.call @SimpleB(%y, %x) : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @MismatchA
arc.define @MismatchA(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.or %arg0, %arg1 {Mismatch} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: arc.define @MismatchB
arc.define @MismatchB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.xor %arg0, %arg1 {Mismatch} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @Mismatch
hw.module @Mismatch(in %x: i4, in %y: i4) {
  // CHECK-NEXT: arc.call @MismatchA(%x, %y)
  // CHECK-NEXT: arc.call @MismatchB(%y, %x)
  %0 = arc.call @MismatchA(%x, %y) : (i4, i4) -> i4
  %1 = arc.call @MismatchB(%y, %x) : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @OutlineConstA
// CHECK-SAME:    %arg0: i4
// CHECK-SAME:    %arg1: i4
arc.define @OutlineConstA(%arg0: i4) -> i4 {
  // CHECK-NEXT: comb.and %arg0, %arg1
  // CHECK-NEXT: arc.output
  %c3_i4 = hw.constant 3 : i4
  %0 = comb.and %arg0, %c3_i4 {OutlineConst} : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @OutlineConstB
arc.define @OutlineConstB(%arg0: i4) -> i4 {
  %c7_i4 = hw.constant 7 : i4
  %0 = comb.and %arg0, %c7_i4 {OutlineConst} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @OutlineConst
hw.module @OutlineConst(in %x: i4, in %y: i4) {
  // CHECK-DAG: %c3_i4 = hw.constant 3 : i4
  // CHECK-DAG: %c7_i4 = hw.constant 7 : i4
  // CHECK-DAG: arc.call @OutlineConstA(%x, %c3_i4)
  // CHECK-DAG: arc.call @OutlineConstA(%y, %c7_i4)
  %0 = arc.call @OutlineConstA(%x) : (i4) -> i4
  %1 = arc.call @OutlineConstB(%y) : (i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @OutlineNonUniformConstsA
// CHECK-SAME:    %arg0: i4
// CHECK-SAME:    %arg1: i4
arc.define @OutlineNonUniformConstsA(%arg0: i4) -> i4 {
  // CHECK-NEXT: comb.mul %arg0, %arg1
  // CHECK-NEXT: arc.output
  %c3_i4 = hw.constant 3 : i4
  %0 = comb.mul %arg0, %c3_i4 {OutlineNonUniformConsts} : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @OutlineNonUniformConstsB
arc.define @OutlineNonUniformConstsB(%arg0: i4) -> i4 {
  %c7_i4 = hw.constant 7 : i4
  %0 = comb.mul %c7_i4, %arg0 {OutlineNonUniformConsts} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @OutlineNonUniformConstsC
arc.define @OutlineNonUniformConstsC() -> i4 {
  %c5_i4 = hw.constant 5 : i4
  %c4_i4 = hw.constant 4 : i4
  %0 = comb.mul %c5_i4, %c4_i4 {OutlineNonUniformConsts} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @OutlineNonUniformConstsD
arc.define @OutlineNonUniformConstsD(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.mul %arg0, %arg1 {OutlineNonUniformConsts} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @OutlineNonUniformConsts
hw.module @OutlineNonUniformConsts(in %x: i4) {
  // CHECK-DAG: %c3_i4 = hw.constant 3 : i4
  // CHECK-DAG: %c7_i4 = hw.constant 7 : i4
  // CHECK-DAG: %c5_i4 = hw.constant 5 : i4
  // CHECK-DAG: %c4_i4 = hw.constant 4 : i4
  // CHECK-DAG: arc.call @OutlineNonUniformConstsA(%x, %c3_i4)
  // CHECK-DAG: arc.call @OutlineNonUniformConstsA(%c7_i4, %x)
  // CHECK-DAG: arc.call @OutlineNonUniformConstsA(%c5_i4, %c4_i4)
  // CHECK-DAG: arc.call @OutlineNonUniformConstsA(%x, %x)
  %0 = arc.call @OutlineNonUniformConstsA(%x) : (i4) -> i4
  %1 = arc.call @OutlineNonUniformConstsB(%x) : (i4) -> i4
  %2 = arc.call @OutlineNonUniformConstsC() : () -> i4
  %3 = arc.call @OutlineNonUniformConstsD(%x, %x) : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @SplitArgumentsA
// CHECK-SAME:    %arg0: i4
// CHECK-SAME:    %arg1: i4
arc.define @SplitArgumentsA(%arg0: i4) -> i4 {
  // CHECK-NEXT: comb.and %arg0, %arg1
  // CHECK-NEXT: arc.output
  %0 = comb.and %arg0, %arg0 {SplitArguments} : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @SplitArgumentsB
arc.define @SplitArgumentsB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {SplitArguments} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @SplitArgumentsC
arc.define @SplitArgumentsC(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg1, %arg0 {SplitArguments} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @SplitArguments
hw.module @SplitArguments(in %x: i4, in %y: i4) {
  // CHECK-DAG: arc.call @SplitArgumentsA(%x, %x)
  // CHECK-DAG: arc.call @SplitArgumentsA(%x, %y)
  // CHECK-DAG: arc.call @SplitArgumentsA(%y, %x)
  %0 = arc.call @SplitArgumentsA(%x) : (i4) -> i4
  %1 = arc.call @SplitArgumentsB(%x, %y) : (i4, i4) -> i4
  %2 = arc.call @SplitArgumentsC(%x, %y) : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @WeirdSplitArgumentsA
// CHECK-SAME:    %arg0: i4
// CHECK-SAME:    %arg1: i4
// CHECK-SAME:    %arg2: i4
// CHECK-SAME:    %arg3: i4
arc.define @WeirdSplitArgumentsA(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: comb.and %arg0, %arg0, %arg1, %arg2, %arg3, %arg3
  // CHECK-NEXT: arc.output
  %0 = comb.and %arg0, %arg0, %arg0, %arg0, %arg1, %arg1 {WeirdSplitArguments} : i4
  arc.output %0 : i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @WeirdSplitArgumentsB
arc.define @WeirdSplitArgumentsB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg0, %arg0, %arg1, %arg1, %arg1 {WeirdSplitArguments} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @WeirdSplitArgumentsC
arc.define @WeirdSplitArgumentsC(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg0, %arg1, %arg0, %arg1, %arg1 {WeirdSplitArguments} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @WeirdSplitArguments
hw.module @WeirdSplitArguments(in %x: i4, in %y: i4) {
  // CHECK-DAG: arc.call @WeirdSplitArgumentsA(%x, %x, %x, %y)
  // CHECK-DAG: arc.call @WeirdSplitArgumentsA(%x, %x, %y, %y)
  // CHECK-DAG: arc.call @WeirdSplitArgumentsA(%x, %y, %x, %y)
  %0 = arc.call @WeirdSplitArgumentsA(%x, %y) : (i4, i4) -> i4
  %1 = arc.call @WeirdSplitArgumentsB(%x, %y) : (i4, i4) -> i4
  %2 = arc.call @WeirdSplitArgumentsC(%x, %y) : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @VariadicDiffsDontDedupA
arc.define @VariadicDiffsDontDedupA(%arg0: i4, %arg1: i4, %arg2: i4) -> i4 {
  %0 = comb.and %arg0, %arg1, %arg2 {VariadicDiffsDontDedup} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: arc.define @VariadicDiffsDontDedupB
arc.define @VariadicDiffsDontDedupB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {VariadicDiffsDontDedup} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @VariadicDiffsDontDedup
hw.module @VariadicDiffsDontDedup(in %x: i4, in %y: i4, in %z: i4) {
  // CHECK-DAG: arc.call @VariadicDiffsDontDedupA(%x, %y, %z)
  // CHECK-DAG: arc.call @VariadicDiffsDontDedupB(%x, %y)
  %0 = arc.call @VariadicDiffsDontDedupA(%x, %y, %z) : (i4, i4, i4) -> i4
  %1 = arc.call @VariadicDiffsDontDedupB(%x, %y) : (i4, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @DedupWithRegionsA
arc.define @DedupWithRegionsA(%arg0: i4, %arg1: i1) -> i4 {
  %0 = scf.if %arg1 -> i4 {
    %1 = comb.and %arg0, %arg0 {DedupWithRegions} : i4
    scf.yield %1 : i4
  } else {
    scf.yield %arg0 : i4
  }
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @DedupWithRegionsB
arc.define @DedupWithRegionsB(%arg0: i1, %arg1: i4) -> i4 {
  %0 = scf.if %arg0 -> i4 {
    %1 = comb.and %arg1, %arg1 {DedupWithRegions} : i4
    scf.yield %1 : i4
  } else {
    scf.yield %arg1 : i4
  }
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @DedupWithRegions
hw.module @DedupWithRegions(in %x: i4, in %y: i1) {
  // CHECK-NEXT: arc.call @DedupWithRegionsA(%x, %y)
  // CHECK-NEXT: arc.call @DedupWithRegionsA(%x, %y)
  %0 = arc.call @DedupWithRegionsA(%x, %y) : (i4, i1) -> i4
  %1 = arc.call @DedupWithRegionsB(%y, %x) : (i1, i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @DiffAttrsBlockDedupA
arc.define @DiffAttrsBlockDedupA(%arg0: i4) -> i4 {
  %0 = comb.and %arg0, %arg0 {DiffAttrsBlockDedup, foo = "a"} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: arc.define @DiffAttrsBlockDedupB
arc.define @DiffAttrsBlockDedupB(%arg0: i4) -> i4 {
  %0 = comb.and %arg0, %arg0 {DiffAttrsBlockDedup, foo = "b"} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: hw.module @DiffAttrsBlockDedup
hw.module @DiffAttrsBlockDedup(in %x: i4) {
  // CHECK-NEXT: arc.call @DiffAttrsBlockDedupA(%x)
  // CHECK-NEXT: arc.call @DiffAttrsBlockDedupB(%x)
  %0 = arc.call @DiffAttrsBlockDedupA(%x) : (i4) -> i4
  %1 = arc.call @DiffAttrsBlockDedupB(%x) : (i4) -> i4
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @DiffTypesBlockDedupA
arc.define @DiffTypesBlockDedupA(%arg0: i4) -> i1 {
  %0 = comb.extract %arg0 from 0 {DiffTypesBlockDedup} : (i4) -> i2
  %1 = comb.extract %0 from 0 : (i2) -> i1
  arc.output %1 : i1
}

// CHECK-LABEL: arc.define @DiffTypesBlockDedupB
arc.define @DiffTypesBlockDedupB(%arg0: i4) -> i1 {
  %0 = comb.extract %arg0 from 0 {DiffTypesBlockDedup} : (i4) -> i3
  %1 = comb.extract %0 from 0 : (i3) -> i1
  arc.output %1 : i1
}

// CHECK-LABEL: hw.module @DiffTypesBlockDedup
hw.module @DiffTypesBlockDedup(in %x: i4) {
  // CHECK-NEXT: arc.call @DiffTypesBlockDedupA(%x)
  // CHECK-NEXT: arc.call @DiffTypesBlockDedupB(%x)
  %0 = arc.call @DiffTypesBlockDedupA(%x) : (i4) -> i1
  %1 = arc.call @DiffTypesBlockDedupB(%x) : (i4) -> i1
  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @CallA
arc.define @CallA(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {RootCallArc} : i4
  arc.output %0 : i4
}

// CHECK-NOT: arc.define @CallB
arc.define @CallB(%arg0: i4, %arg1: i4) -> i4 {
  %0 = comb.and %arg0, %arg1 {RootCallArc} : i4
  arc.output %0 : i4
}

// CHECK-LABEL: arc.define @RootCallArc
arc.define @RootCallArc(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK-NEXT: arc.call @CallA(%arg1, %arg0)
  %0 = arc.call @CallA(%arg1, %arg0) {RootCallArc} : (i4, i4) -> i4
  // CHECK-NEXT: arc.call @CallA(%arg0, %arg1)
  %1 = arc.call @CallB(%arg0, %arg1) {RootCallArc} : (i4, i4) -> i4
  %2 = comb.and %0, %1 {RootCallArc} : i4
  arc.output %2 : i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: arc.define @OutlineRegressionA
arc.define @OutlineRegressionA(%arg0: i1, %arg1: i3) -> (i3, i3) {
  %c0_i3 = hw.constant 0 : i3
  %0 = comb.mux %arg0, %arg1, %c0_i3 {OutlineRegression} : i3
  arc.output %0, %c0_i3 : i3, i3
}

// CHECK-NOT: arc.define @OutlineRegressionB
arc.define @OutlineRegressionB(%arg0: i1, %arg1: i3) -> (i3, i3) {
  %c0_i3 = hw.constant 0 : i3
  %0 = comb.mux %arg0, %c0_i3, %arg1 {OutlineRegression} : i3
  arc.output %0, %c0_i3 : i3, i3
}

// CHECK-LABEL: hw.module @OutlineRegression
hw.module @OutlineRegression(in %a: i1, in %b: i3) {
  // CHECK-NEXT: [[K0:%.+]] = hw.constant 0 : i3
  // CHECK-NEXT: arc.call @OutlineRegressionA(%a, %b, [[K0]]) :
  // CHECK-NEXT: [[K1:%.+]] = hw.constant 0 : i3
  // CHECK-NEXT: arc.call @OutlineRegressionA(%a, [[K1]], %b) :
  arc.call @OutlineRegressionA(%a, %b) : (i1, i3) -> (i3, i3)
  arc.call @OutlineRegressionB(%a, %b) : (i1, i3) -> (i3, i3)
}
