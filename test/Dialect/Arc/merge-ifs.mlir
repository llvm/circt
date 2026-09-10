// RUN: circt-opt --arc-merge-ifs %s | FileCheck %s

func.func private @Blocker()

// CHECK-LABEL: func.func @DontMoveUnusedOps
func.func @DontMoveUnusedOps(%arg0: !arc.state<i42>) {
  // CHECK-NEXT: arc.state_read %arg0 : <i42>
  arc.state_read %arg0 : <i42>
  // CHECK-NEXT: hw.constant false
  hw.constant false
  // CHECK-NEXT: hw.constant true
  hw.constant true
  return
}

// CHECK-LABEL: func.func @SinkReads
func.func @SinkReads(%arg0: !arc.state<i42>, %arg1: !arc.memory<2 x i42, i1>, %arg2: i1) {
  %0 = arc.state_read %arg0 : <i42>
  %1 = arc.memory_read %arg1[%arg2] : <2 x i42, i1>
  // CHECK-NEXT: hw.constant false
  hw.constant false
  // CHECK-NEXT: scf.if
  scf.if %arg2 {
    // CHECK-NEXT: hw.constant true
    hw.constant true
    // CHECK-NEXT: arc.state_read
    // CHECK-NEXT: arc.memory_read
    // CHECK-NEXT: comb.xor
    comb.xor %0, %1 : i42
  }
  return
}

// CHECK-LABEL: func.func @MoveReads
func.func @MoveReads(%arg0: !arc.state<i42>, %arg1: !arc.memory<2 x i42, i1>, %arg2: i1) {
  %0 = arc.state_read %arg0 : <i42>
  %1 = arc.memory_read %arg1[%arg2] : <2 x i42, i1>
  // CHECK-NEXT: hw.constant false
  hw.constant false
  // CHECK-NEXT: arc.state_read
  // CHECK-NEXT: arc.memory_read
  // CHECK-NEXT: scf.if
  scf.if %arg2 {
    comb.xor %0, %1 : i42
  }
  comb.xor %0, %1 : i42
  return
}

// CHECK-LABEL: func.func @SinkAndMoveReads
func.func @SinkAndMoveReads(%arg0: !arc.state<i42>, %arg1: !arc.memory<2 x i42, i1>, %arg2: i1, %arg3: i1) {
  %0 = arc.state_read %arg0 {a} : <i42>
  %1 = arc.state_read %arg0 {b} : <i42>
  %2 = arc.state_read %arg0 {c} : <i42>
  %3 = arc.memory_read %arg1[%arg2] {x} : <2 x i42, i1>
  %4 = arc.memory_read %arg1[%arg2] {y} : <2 x i42, i1>
  %5 = arc.memory_read %arg1[%arg2] {z} : <2 x i42, i1>
  // CHECK-NEXT: scf.if
  scf.if %arg2 {
    // CHECK-NEXT: hw.constant false
    hw.constant false
    // CHECK-NEXT: arc.state_read %arg0 {a}
    // CHECK-NEXT: arc.memory_read %arg1[%arg2] {x}
    // CHECK-NEXT: comb.xor
    comb.xor %0, %3 : i42
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: arc.state_read %arg0 {b}
  // CHECK-NEXT: arc.memory_read %arg1[%arg2] {y}
  // CHECK-NEXT: scf.if
  scf.if %arg3 {
    // CHECK-NEXT: hw.constant false
    hw.constant false
    // CHECK-NEXT: comb.xor
    comb.xor %1, %4 : i42
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: arc.state_read %arg0 {c}
  // CHECK-NEXT: arc.memory_read %arg1[%arg2] {z}
  // CHECK-NEXT: comb.xor
  comb.xor %1, %2, %4, %5 : i42
  return
}

// CHECK-LABEL: func.func @WriteBlocksReadMove
func.func @WriteBlocksReadMove(
  %arg0: !arc.state<i42>,
  %arg1: !arc.state<i42>,
  %arg2: !arc.memory<2 x i42, i1>,
  %arg3: !arc.memory<2 x i42, i1>,
  %arg4: i1,
  %arg5: i42
) {
  %0 = arc.state_read %arg0 {blocked} : <i42>
  %1 = arc.state_read %arg1 {free} : <i42>
  %2 = arc.memory_read %arg2[%arg4] {blocked} : <2 x i42, i1>
  %3 = arc.memory_read %arg3[%arg4] {free} : <2 x i42, i1>
  // CHECK-NEXT: hw.constant false
  hw.constant false
  // CHECK-NEXT: arc.state_read %arg0 {blocked}
  // CHECK-NEXT: arc.memory_read %arg2[%arg4] {blocked}
  // CHECK-NEXT: scf.if
  scf.if %arg4 {
    // CHECK-NEXT: arc.state_write
    // CHECK-NEXT: arc.memory_write
    arc.state_write %arg0 = %arg5 : <i42>
    arc.memory_write %arg2[%arg4], %arg5 : <2 x i42, i1>
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: arc.state_read %arg1 {free}
  // CHECK-NEXT: arc.memory_read %arg3[%arg4] {free}
  // CHECK-NEXT: comb.xor
  comb.xor %0, %1, %2, %3 : i42
  return
}

// CHECK-LABEL: func.func @MovedOpsRetainOrder
func.func @MovedOpsRetainOrder(%arg0: i42, %arg1: i1) {
  // CHECK-NEXT: hw.constant false {ka}
  // CHECK-NEXT: hw.constant false {kb}
  // CHECK-NEXT: hw.constant false {kc}
  // CHECK-NEXT: comb.xor {{%.+}} {a0}
  // CHECK-NEXT: comb.xor {{%.+}} {a1}
  // CHECK-NEXT: comb.xor {{%.+}} {b0}
  // CHECK-NEXT: comb.xor {{%.+}} {b1}
  // CHECK-NEXT: comb.xor {{%.+}} {c0}
  // CHECK-NEXT: comb.xor {{%.+}} {c1}
  %a0 = comb.xor %arg0 {a0} : i42
  %a1 = comb.xor %a0 {a1} : i42
  hw.constant false {ka}
  %b0 = comb.xor %arg0 {b0} : i42
  %b1 = comb.xor %b0 {b1} : i42
  hw.constant false {kb}
  %c0 = comb.xor %arg0 {c0} : i42
  %c1 = comb.xor %c0 {c1} : i42
  hw.constant false {kc}
  // CHECK-NEXT: scf.if
  scf.if %arg1 {
    comb.xor %a1 {ia} : i42
    comb.xor %b1 {ib} : i42
    comb.xor %c1 {ic} : i42
  }
  comb.xor %a1 {xa} : i42
  comb.xor %b1 {xb} : i42
  comb.xor %c1 {xc} : i42
  return
}

// CHECK-LABEL: func.func @MergeAdjacentIfs
func.func @MergeAdjacentIfs(%arg0: i1, %arg1: i1) {
  // CHECK-NEXT: scf.if %arg0 {
  // CHECK-NEXT:   hw.constant false {a}
  // CHECK-NEXT:   hw.constant false {b}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {a}
  }
  scf.if %arg0 {
    hw.constant false {b}
  }
  // CHECK-NEXT: scf.if %arg1 {
  // CHECK-NEXT:   hw.constant false {c}
  // CHECK-NEXT: }
  scf.if %arg1 {
    hw.constant false {c}
  }
  return
}

// CHECK-LABEL: func.func @MergeIfsAcrossOps
func.func @MergeIfsAcrossOps(
  %arg0: i1,
  %arg1: !arc.state<i42>,
  %arg2: !arc.memory<2 x i42, i1>,
  %arg3: i42
) {
  // CHECK-NEXT: arc.state_read
  // CHECK-NEXT: arc.state_write
  // CHECK-NEXT: arc.memory_read
  // CHECK-NEXT: arc.memory_write
  // CHECK-NEXT: scf.if %arg0 {
  // CHECK-NEXT:   hw.constant false {a}
  // CHECK-NEXT:   hw.constant false {b}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {a}
  }
  arc.state_read %arg1 : <i42>
  arc.state_write %arg1 = %arg3 : <i42>
  arc.memory_read %arg2[%arg0] : <2 x i42, i1>
  arc.memory_write %arg2[%arg0], %arg3 : <2 x i42, i1>
  scf.if %arg0 {
    hw.constant false {b}
  }
  return
}

// CHECK-LABEL: func.func @DontMergeIfsAcrossSideEffects
func.func @DontMergeIfsAcrossSideEffects(
  %arg0: i1,
  %arg1: !arc.state<i42>,
  %arg2: !arc.memory<2 x i42, i1>,
  %arg3: i42
) {
  // CHECK-NEXT: scf.if %arg0 {
  // CHECK-NEXT:   hw.constant false {a}
  // CHECK-NEXT:   func.call @Blocker() {blockerA}
  // CHECK-NEXT:   hw.constant false {b}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {a}
    func.call @Blocker() {blockerA} : () -> ()
  }
  scf.if %arg0 {
    hw.constant false {b}
  }
  // CHECK-NEXT: call @Blocker() {cantMoveAcrossA}
  call @Blocker() {cantMoveAcrossA} : () -> ()
  // CHECK-NEXT: scf.if %arg0 {
  // CHECK-NEXT:   hw.constant false {c}
  // CHECK-NEXT:   arc.state_write %arg1 = %arg3 {blockerB}
  // CHECK-NEXT:   hw.constant false {d}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {c}
    arc.state_write %arg1 = %arg3 {blockerB} : <i42>
  }
  scf.if %arg0 {
    hw.constant false {d}
  }
  // CHECK-NEXT: arc.state_read %arg1 {cantMoveAcrossB}
  arc.state_read %arg1 {cantMoveAcrossB} : <i42>
  // CHECK-NEXT: scf.if %arg0 {
  // CHECK-NEXT:   hw.constant false {e}
  // CHECK-NEXT:   arc.memory_write %arg2[%arg0], %arg3 {blockerC}
  // CHECK-NEXT:   hw.constant false {f}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {e}
    arc.memory_write %arg2[%arg0], %arg3 {blockerC} : <2 x i42, i1>
  }
  scf.if %arg0 {
    hw.constant false {f}
  }
  // CHECK-NEXT: arc.memory_read %arg2[%arg0] {cantMoveAcrossC}
  arc.memory_read %arg2[%arg0] {cantMoveAcrossC} : <2 x i42, i1>
  // CHECK-NEXT: scf.if %arg0 {
  // CHECK-NEXT:   hw.constant false {g}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {g}
  }
  return
}

// CHECK-LABEL: func.func @MergeNestedIfs
func.func @MergeNestedIfs(%arg0: i42, %arg1: i1, %arg2: i1) {
  // CHECK-NEXT: scf.if %arg1 {
  // CHECK-NEXT:   hw.constant false {a}
  // CHECK-NEXT:   hw.constant false {b}
  // CHECK-NEXT:   hw.constant false {c}
  // CHECK-NEXT:   scf.if %arg2 {
  // CHECK-NEXT:     hw.constant false {x}
  // CHECK-NEXT:     hw.constant false {y}
  // CHECK-NEXT:   }
  // CHECK-NEXT:   hw.constant false {d}
  // CHECK-NEXT: }
  scf.if %arg1 {
    hw.constant false {a}
    scf.if %arg2 {
      hw.constant false {x}
    }
    hw.constant false {b}
  }
  scf.if %arg1 {
    hw.constant false {c}
    scf.if %arg2 {
      hw.constant false {y}
    }
    hw.constant false {d}
  }
  return
}

// Check that ops containing a write aren't sunk
// CHECK-LABEL: func.func @DontNestWrites
func.func @DontNestWrites(%arg0: !arc.state<i1>, %arg1: i1, %arg2: i1) {
  // We just want to check that the first if hasn't been moved into the second
  // CHECK-NEXT:  {{%.+}} = scf.if %arg1 -> (i1) {
  // CHECK:  } else {
  // CHECK:  }
  // CHECK-NEXT:  scf.if %arg2 {
  // CHECK:  }
  // CHECK-NEXT:  return

  %1 = scf.if %arg1 -> (i1) {
    %0 = hw.constant true
    arc.state_write %arg0 = %0 : <i1>
    scf.yield %0 : i1
  } else {
    %0 = hw.constant false
    scf.yield %0 : i1
  }
  scf.if %arg2 {
    %0 = comb.or %1, %1 : i1
  }
  return
}
