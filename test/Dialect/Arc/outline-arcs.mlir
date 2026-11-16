// RUN: circt-opt --arc-outline %s | FileCheck %s
// RUN: circt-opt --arc-outline --arc-outline %s | FileCheck %s

// CHECK-LABEL: hw.module @Empty
// CHECK-NEXT:    hw.output
// CHECK-NEXT:  }
hw.module @Empty() {
  hw.output
}

// CHECK-LABEL: hw.module @Passthrough
// CHECK-NEXT:    hw.output %a :
// CHECK-NEXT:  }
hw.module @Passthrough(in %a: i42, out b: i42) {
  hw.output %a : i42
}

// CHECK-LABEL: arc.define @SimpleDAG_arc0
// CHECK-NEXT:    [[T0:%.+]] = comb.add %arg0, %arg1
// CHECK-NEXT:    [[T3:%.+]] = comb.xor [[T0]], %arg2
// CHECK-NEXT:    arc.output [[T3]]
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @SimpleDAG_arc1
// CHECK-NEXT:    [[T1:%.+]] = comb.sub %arg0, %arg1
// CHECK-NEXT:    [[T4:%.+]] = comb.xor [[T1]], %arg2
// CHECK-NEXT:    arc.output [[T4]]
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @SimpleDAG_arc2
// CHECK-NEXT:    [[T2:%.+]] = comb.mul %arg0, %arg1
// CHECK-NEXT:    arc.output [[T2]]
// CHECK-NEXT:  }
// CHECK-LABEL: hw.module @SimpleDAG
// CHECK-DAG:     [[T2:%.+]] = arc.call @SimpleDAG_arc2(%a, %b)
// CHECK-DAG:     [[T3:%.+]] = arc.call @SimpleDAG_arc0(%a, %b, [[T2]])
// CHECK-DAG:     [[T4:%.+]] = arc.call @SimpleDAG_arc1(%a, %b, [[T2]])
// CHECK-NEXT:    hw.output [[T3]], [[T4]] :
// CHECK-NEXT:  }
hw.module @SimpleDAG(in %a: i42, in %b: i42, out u: i42, out v: i42) {
  %3 = comb.xor %0, %2 : i42
  %4 = comb.xor %1, %2 : i42
  %0 = comb.add %a, %b : i42
  %1 = comb.sub %a, %b : i42
  %2 = comb.mul %a, %b : i42
  hw.output %3, %4 : i42, i42
}

// CHECK-LABEL: hw.module @Regs
// CHECK-DAG:     [[T1:%.+]] = seq.compreg [[T0:.+]], %clock
// CHECK-DAG:     [[T3:%.+]] = seq.firreg [[T2:.+]] clock %clock
// CHECK-DAG:     [[T0:%.+]] = arc.call @Regs_arc{{.+}}(%a, %b)
// CHECK-DAG:     [[T2:%.+]] = arc.call @Regs_arc{{.+}}(%a, [[T1]])
// CHECK-DAG:     [[T4:%.+]] = arc.call @Regs_arc{{.+}}([[T3]], %b)
// CHECK-NEXT:    hw.output [[T4]] :
// CHECK-NEXT:  }
hw.module @Regs(in %clock: !seq.clock, in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add %a, %b : i42
  %1 = seq.compreg %0, %clock : i42
  %2 = comb.sub %a, %1 : i42
  %3 = seq.firreg %2 clock %clock : i42
  %4 = comb.mul %3, %b : i42
  hw.output %4 : i42
}

// CHECK-LABEL: hw.module @Instance(
// CHECK-DAG:     [[T0:%.+]] = arc.call @Instance_arc{{.+}}(%a, %b)
// CHECK-DAG:     [[T1:%.+]] = hw.instance "sub" @InstanceSub(a: [[T0]]: i42)
// CHECK-DAG:     [[T2:%.+]] = arc.call @Instance_arc{{.+}}(%a, [[T1]])
// CHECK-NEXT:    hw.output [[T2]] :
// CHECK-NEXT:  }
hw.module @Instance(in %a: i42, in %b: i42, out z: i42) {
  %0 = comb.add %a, %b : i42
  %1 = hw.instance "sub" @InstanceSub(a: %0: i42) -> (z: i42)
  %2 = comb.sub %a, %1 : i42
  hw.output %2 : i42
}
hw.module.extern private @InstanceSub(in %a: i42, out z: i42)

// CHECK-LABEL: arc.define @TightLoop_arc0
// CHECK-NEXT:    [[T0:%.+]] = comb.add %arg0, %arg1
// CHECK-NEXT:    arc.output [[T0]]
// CHECK-NEXT:  }
// CHECK-LABEL: hw.module @TightLoop(
// CHECK-NEXT:    [[T0:%.+]] = arc.call @TightLoop_arc0(%a, [[T0]])
// CHECK-NEXT:    hw.output [[T0]] :
// CHECK-NEXT:  }
hw.module @TightLoop(in %a: i42, out z: i42) {
  %0 = comb.add %a, %0 : i42
  hw.output %0 : i42
}

// CHECK-LABEL: arc.define @WideLoop1_arc0
// CHECK-NEXT:    [[T0:%.+]] = comb.add %arg0, %arg1
// CHECK-NEXT:    [[T1:%.+]] = comb.sub %arg0, [[T0]]
// CHECK-NEXT:    arc.output [[T1]]
// CHECK-NEXT:  }
// CHECK-LABEL: hw.module @WideLoop1(
// CHECK-NEXT:    [[T1:%.+]] = arc.call @WideLoop1_arc0(%a, [[T1]])
// CHECK-NEXT:    hw.output [[T1]] :
// CHECK-NEXT:  }
hw.module @WideLoop1(in %a: i42, out z: i42) {
  %0 = comb.add %a, %1 : i42
  %1 = comb.sub %a, %0 : i42
  hw.output %1 : i42
}

// CHECK-LABEL: arc.define @WideLoop2_arc0
// CHECK-NEXT:    [[T0:%.+]] = comb.add %arg0, %arg1
// CHECK-NEXT:    [[T1:%.+]] = comb.sub %arg0, [[T0]]
// CHECK-NEXT:    arc.output [[T0]], [[T1]]
// CHECK-NEXT:  }
// CHECK-LABEL: hw.module @WideLoop2(
// CHECK-NEXT:    [[T01:%.+]]:2 = arc.call @WideLoop2_arc0(%a, [[T01]]#1)
// CHECK-NEXT:    hw.output [[T01]]#0 :
// CHECK-NEXT:  }
hw.module @WideLoop2(in %a: i42, out z: i42) {
  %0 = comb.add %a, %1 : i42
  %1 = comb.sub %a, %0 : i42
  hw.output %0 : i42
}

// CHECK-LABEL: arc.define @CloneConsts_arc0
// CHECK-NEXT:    [[CST:%.+]] = hw.constant 9001 :
// CHECK-NEXT:    [[T0:%.+]] = comb.add %arg0, [[CST]]
// CHECK-NEXT:    arc.output [[T0]]
// CHECK-NEXT:  }
// CHECK-LABEL: arc.define @CloneConsts_arc1
// CHECK-NEXT:    [[CST:%.+]] = hw.constant 9001 :
// CHECK-NEXT:    [[T1:%.+]] = comb.sub %arg0, [[CST]]
// CHECK-NEXT:    arc.output [[T1]]
// CHECK-NEXT:  }
// CHECK-LABEL: hw.module @CloneConsts
// CHECK-DAG:     [[T0:%.+]] = arc.call @CloneConsts_arc0(%a)
// CHECK-DAG:     [[T1:%.+]] = arc.call @CloneConsts_arc1(%b)
// CHECK-NEXT:    hw.output [[T0]], [[T1]] :
// CHECK-NEXT:  }
hw.module @CloneConsts(in %a: i42, in %b: i42, out u: i42, out v: i42) {
  %c9001_i42 = hw.constant 9001 : i42
  %0 = comb.add %a, %c9001_i42 : i42
  %1 = comb.sub %b, %c9001_i42 : i42
  hw.output %0, %1 : i42, i42
}

// CHECK-LABEL: arc.define @Regions_arc0
// CHECK-NEXT:    [[T0:%.+]] = comb.add %arg0, %arg1
// CHECK-NEXT:    [[T1:%.+]] = scf.execute_region
// CHECK:         arc.output [[T1]]
// CHECK-NEXT:  }
// CHECK-LABEL: hw.module @Regions
// CHECK-DAG:     [[T1:%.+]] = arc.call @Regions_arc0(%a, %b)
// CHECK-NEXT:    hw.output [[T1]] :
// CHECK-NEXT:  }
hw.module @Regions(in %a: i42, in %b: i42, out z: i42) {
  %1 = scf.execute_region -> i42 {
    scf.yield %0 : i42
  }
  %0 = comb.add %a, %b : i42
  hw.output %1 : i42
}
