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

// CHECK-LABEL: func.func @SinkStateReads
func.func @SinkStateReads(%arg0: !arc.state<i42>, %arg1: i1) {
  %0 = arc.state_read %arg0 : <i42>
  // CHECK-NEXT: hw.constant false {dummy0}
  hw.constant false {dummy0}
  // CHECK-NEXT: scf.if
  scf.if %arg1 {
    // CHECK-NEXT: hw.constant false {dummy1}
    hw.constant false {dummy1}
    // CHECK-NEXT: arc.state_read %arg0
    // CHECK-NEXT: comb.xor
    comb.xor %0 : i42
  }
  return
}

// CHECK-LABEL: func.func @MoveStateReads
func.func @MoveStateReads(%arg0: !arc.state<i42>, %arg1: i1) {
  %0 = arc.state_read %arg0 : <i42>
  // CHECK-NEXT: hw.constant false
  hw.constant false
  // CHECK-NEXT: arc.state_read %arg0
  // CHECK-NEXT: scf.if
  scf.if %arg1 {
    comb.xor %0 : i42
  }
  comb.xor %0 : i42
  return
}

// CHECK-LABEL: func.func @SinkAndMoveStateReads
func.func @SinkAndMoveStateReads(%arg0: !arc.state<i42>, %arg1: i1, %arg2: i1) {
  %0 = arc.state_read %arg0 {a} : <i42>
  %1 = arc.state_read %arg0 {b} : <i42>
  %2 = arc.state_read %arg0 {c} : <i42>
  // CHECK-NEXT: scf.if %arg1
  scf.if %arg1 {
    // CHECK-NEXT: hw.constant false
    hw.constant false
    // CHECK-NEXT: arc.state_read %arg0 {a}
    // CHECK-NEXT: comb.xor
    comb.xor %0 : i42
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: arc.state_read %arg0 {b}
  // CHECK-NEXT: scf.if %arg2
  scf.if %arg2 {
    // CHECK-NEXT: hw.constant false
    hw.constant false
    // CHECK-NEXT: comb.xor
    comb.xor %1 : i42
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: arc.state_read %arg0 {c}
  // CHECK-NEXT: comb.xor
  comb.xor %1, %2 : i42
  return
}

// CHECK-LABEL: func.func @StateWriteBlocksReadMove
func.func @StateWriteBlocksReadMove(%arg0: !arc.state<i42>, %arg1: !arc.state<i42>, %arg2: i1, %arg3: i42) {
  %0 = arc.state_read %arg0 {blocked} : <i42>
  %1 = arc.state_read %arg1 {free} : <i42>
  // CHECK-NEXT: hw.constant false {dummy}
  hw.constant false {dummy}
  // CHECK-NEXT: arc.state_read %arg0 {blocked}
  // CHECK-NEXT: scf.if
  scf.if %arg2 {
    // CHECK-NEXT: arc.state_write
    arc.state_write %arg0 = %arg3 : <i42>
  }
  // CHECK-NEXT: }
  // CHECK-NEXT: arc.state_read %arg1 {free}
  // CHECK-NEXT: comb.xor
  comb.xor %0, %1 : i42
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
func.func @MergeIfsAcrossOps(%arg0: i1, %arg1: !arc.state<i42>, %arg2: i42) {
  // CHECK-NEXT: arc.state_read
  // CHECK-NEXT: arc.state_write
  // CHECK-NEXT: scf.if %arg0 {
  // CHECK-NEXT:   hw.constant false {a}
  // CHECK-NEXT:   hw.constant false {b}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {a}
  }
  arc.state_read %arg1 : <i42>
  arc.state_write %arg1 = %arg2 : <i42>
  scf.if %arg0 {
    hw.constant false {b}
  }
  return
}

// CHECK-LABEL: func.func @DontMergeIfsAcrossSideEffects
func.func @DontMergeIfsAcrossSideEffects(%arg0: i1, %arg1: !arc.state<i42>, %arg2: i42) {
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
  // CHECK-NEXT:   arc.state_write %arg1 = %arg2 {blockerB}
  // CHECK-NEXT:   hw.constant false {d}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {c}
    arc.state_write %arg1 = %arg2 {blockerB} : <i42>
  }
  scf.if %arg0 {
    hw.constant false {d}
  }
  // CHECK-NEXT: arc.state_read %arg1 {cantMoveAcrossB}
  arc.state_read %arg1 {cantMoveAcrossB} : <i42>
  // CHECK-NEXT: scf.if %arg0 {
  // CHECK-NEXT:   hw.constant false {e}
  // CHECK-NEXT: }
  scf.if %arg0 {
    hw.constant false {e}
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
