// RUN: circt-opt %s --arc-split-loops | FileCheck %s

// CHECK-LABEL: hw.module @Simple(
hw.module @Simple(in %clock: !seq.clock, in %a: i4, in %b: i4, out x: i4, out y: i4) {
  // CHECK-NEXT: %0 = arc.call @SimpleArc_split_0(%a, %b)
  // CHECK-NEXT: %1 = arc.call @SimpleArc_split_1(%0, %a)
  // CHECK-NEXT: %2 = arc.call @SimpleArc_split_2(%0, %b)
  // CHECK-NEXT: hw.output %1, %2
  %0:2 = arc.call @SimpleArc(%a, %b) : (i4, i4) -> (i4, i4)
  hw.output %0#0, %0#1 : i4, i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @SimpleArc_split_0(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.and %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @SimpleArc_split_1(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.add %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @SimpleArc_split_2(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-NOT:   arc.define @SimplerArc(
arc.define @SimpleArc(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %0 = comb.and %arg0, %arg1 : i4
  %1 = comb.add %0, %arg0 : i4
  %2 = comb.mul %0, %arg1 : i4
  arc.output %1, %2 : i4, i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @Unchanged(
hw.module @Unchanged(in %a: i4, out x: i4, out y0: i4, out y1: i4) {
  // CHECK-NEXT: %0 = arc.call @UnchangedArc1(%a)
  // CHECK-NEXT: %1:2 = arc.call @UnchangedArc2(%a)
  // CHECK-NEXT: hw.output %0, %1#0, %1#1
  %0 = arc.call @UnchangedArc1(%a) : (i4) -> i4
  %1:2 = arc.call @UnchangedArc2(%a) : (i4) -> (i4, i4)
  hw.output %0, %1#0, %1#1 : i4, i4, i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @UnchangedArc1(%arg0: i4)
arc.define @UnchangedArc1(%arg0: i4) -> i4 {
  %0 = comb.mul %arg0, %arg0 : i4
  arc.output %0 : i4
}

// CHECK-LABEL: arc.define @UnchangedArc2(%arg0: i4)
arc.define @UnchangedArc2(%arg0: i4) -> (i4, i4) {
  %true = hw.constant true
  %0, %1 = scf.if %true -> (i4, i4) {
    scf.yield %arg0, %arg0 : i4, i4
  } else {
    scf.yield %arg0, %arg0 : i4, i4
  }
  arc.output %0, %1 : i4, i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @Passthrough(
hw.module @Passthrough(in %a: i4, in %b: i4, out x0: i4, out x1: i4, out y0: i4, out y1: i4) {
  // CHECK-NEXT: %0 = arc.call @PassthroughArc2(%a)
  // CHECK-NEXT: hw.output %a, %b, %0, %b
  %0:2 = arc.call @PassthroughArc1(%a, %b) : (i4, i4) -> (i4, i4)
  %1:2 = arc.call @PassthroughArc2(%a, %b) : (i4, i4) -> (i4, i4)
  hw.output %0#0, %0#1, %1#0, %1#1 : i4, i4, i4, i4
}
// CHECK-NEXT: }

// CHECK-NOT: arc.define @PassthroughArc1(
arc.define @PassthroughArc1(%arg0: i4, %arg1: i4) -> (i4, i4) {
  arc.output %arg0, %arg1 : i4, i4
}

// CHECK-LABEL: arc.define @PassthroughArc2(%arg0: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
arc.define @PassthroughArc2(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %0 = comb.mul %arg0, %arg0 : i4
  arc.output %0, %arg1 : i4, i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @NestedRegions(
hw.module @NestedRegions(in %a: i4, in %b: i4, in %c: i4, out x: i4, out y: i4) {
  // CHECK-NEXT: %0:3 = arc.call @NestedRegionsArc_split_0(%a, %b, %c)
  // CHECK-NEXT: %1 = arc.call @NestedRegionsArc_split_1(%0#0, %0#1)
  // CHECK-NEXT: %2 = arc.call @NestedRegionsArc_split_2(%0#2)
  // CHECK-NEXT: hw.output %1, %2
  %0, %1 = arc.call @NestedRegionsArc(%a, %b, %c) : (i4, i4, i4) -> (i4, i4)
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @NestedRegionsArc_split_0(%arg0: i4, %arg1: i4, %arg2: i4)
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %0:3 = scf.if %true -> (i4, i4, i4) {
// CHECK-NEXT:      scf.yield %arg0, %arg1, %arg2
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg2, %arg1, %arg0
// CHECK-NEXT:    }
// CHECK-NEXT:    arc.output %0#0, %0#1, %0#2
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @NestedRegionsArc_split_1(%arg0: i4, %arg1: i4)
// CHECK-NEXT:    %0 = comb.add %arg0, %arg1
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @NestedRegionsArc_split_2(%arg0: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-NOT:   arc.define @NestedRegionsArc(
arc.define @NestedRegionsArc(%arg0: i4, %arg1: i4, %arg2: i4) -> (i4, i4) {
  %true = hw.constant true
  %0, %1, %2 = scf.if %true -> (i4, i4, i4) {
    scf.yield %arg0, %arg1, %arg2 : i4, i4, i4
  } else {
    scf.yield %arg2, %arg1, %arg0 : i4, i4, i4
  }
  %3 = comb.add %0, %1 : i4
  %4 = comb.mul %2, %2 : i4
  arc.output %3, %4 : i4, i4
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @BreakFalseLoops(
hw.module @BreakFalseLoops(in %a: i4, out x: i4, out y: i4) {
  // CHECK-NEXT: %0 = arc.call @BreakFalseLoopsArc_split_0(%a)
  // CHECK-NEXT: %1 = arc.call @BreakFalseLoopsArc_split_1(%0)
  // CHECK-NEXT: %2 = arc.call @BreakFalseLoopsArc_split_0(%3)
  // CHECK-NEXT: %3 = arc.call @BreakFalseLoopsArc_split_1(%a)
  // CHECK-NEXT: hw.output %1, %2
  %0, %1 = arc.call @BreakFalseLoopsArc(%a, %0) : (i4, i4) -> (i4, i4)
  %2, %3 = arc.call @BreakFalseLoopsArc(%3, %a) : (i4, i4) -> (i4, i4)
  hw.output %1, %2 : i4, i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @BreakFalseLoopsArc_split_0(%arg0: i4)
// CHECK-NEXT:    %0 = comb.add %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @BreakFalseLoopsArc_split_1(%arg0: i4)
// CHECK-NEXT:    %0 = comb.mul %arg0, %arg0
// CHECK-NEXT:    arc.output %0
// CHECK-NEXT:  }
// CHECK-NOT:   arc.define @BreakFalseLoopsArc(
arc.define @BreakFalseLoopsArc(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %0 = comb.add %arg0, %arg0 : i4
  %1 = comb.mul %arg1, %arg1 : i4
  arc.output %0, %1 : i4, i4
}

//===----------------------------------------------------------------------===//
// COM: https://github.com/llvm/circt/issues/4862

// CHECK-LABEL: @SplitDependencyModule
hw.module @SplitDependencyModule(in %a: i1, out x: i1, out y: i1) {
  // CHECK-NEXT: %0 = arc.call @SplitDependency_split_1(%a, %a) : (i1, i1) -> i1
  // CHECK-NEXT: %1 = arc.call @SplitDependency_split_0(%a, %a, %0) : (i1, i1, i1) -> i1
  // CHECK-NEXT: hw.output %0, %1 : i1, i1
  %0, %1 = arc.call @SplitDependency(%a, %a, %a) : (i1, i1, i1) -> (i1, i1)
  hw.output %0, %1 : i1, i1
}
// CHECK-NEXT: }

// CHECK-NEXT: arc.define @SplitDependency_split_0(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
// CHECK-NEXT:   %0 = comb.xor %arg0, %arg1 : i1
// CHECK-NEXT:   %1 = comb.xor %0, %arg2 : i1
// CHECK-NEXT:   arc.output %1 : i1
// CHECK-NEXT: }
// CHECK-NEXT: arc.define @SplitDependency_split_1(%arg0: i1, %arg1: i1) -> i1 {
// CHECK-NEXT:   %0 = comb.xor %arg0, %arg1 : i1
// CHECK-NEXT:   arc.output %0 : i1
// CHECK-NEXT: }
// CHECK-NOT:  arc.define @SplitDependency(
arc.define @SplitDependency(%arg0: i1, %arg1: i1, %arg2: i1) -> (i1, i1) {
  %0 = comb.xor %arg0, %arg1 : i1
  %1 = comb.xor %arg2, %arg1 : i1
  %2 = comb.xor %0, %1 : i1
  arc.output %1, %2 : i1, i1
}
