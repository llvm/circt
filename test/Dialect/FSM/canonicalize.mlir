// RUN: circt-opt --canonicalize %s | FileCheck %s

// No-op variable updates (var <- var) are removed.
// CHECK-LABEL: fsm.machine @foo
// CHECK-NOT: fsm.update
fsm.machine @foo(%arg0: i1) attributes {initialState = "A"} {
  %var = fsm.variable "var" {initValue = 0 : i16} : i16
  fsm.state @A transitions {
    fsm.transition @A action {
        fsm.update %var, %var : i16
    }
  }
}

// Mutually exclusive transition elimination (#3577), comb form: a state with
// exactly two transitions whose second guard is the logical complement of the
// first (`!cond` via comb.xor) has its second guard stripped, dropping the
// redundant complement logic.
// CHECK-LABEL: fsm.machine @mutex_comb
// CHECK: fsm.transition @B guard
// CHECK-NEXT: fsm.return %arg0
// CHECK: fsm.transition @C
// CHECK-NOT: comb.xor
// CHECK-LABEL: fsm.machine @mutex_arith
fsm.machine @mutex_comb(%cond: i1) attributes {initialState = "A"} {
  fsm.state @A transitions {
    fsm.transition @B guard {
      fsm.return %cond
    }
    fsm.transition @C guard {
      %true = hw.constant true
      %ncond = comb.xor %cond, %true : i1
      fsm.return %ncond
    }
  }
  fsm.state @B transitions {
    fsm.transition @A
  }
  fsm.state @C transitions {
    fsm.transition @A
  }
}

// Same elimination for the arith spelling of the complement (arith.xori).
// CHECK: fsm.transition @B guard
// CHECK-NEXT: fsm.return %arg0
// CHECK: fsm.transition @C
// CHECK-NOT: arith.xori
// CHECK-LABEL: fsm.machine @not_complementary
fsm.machine @mutex_arith(%cond: i1) attributes {initialState = "A"} {
  fsm.state @A transitions {
    fsm.transition @B guard {
      fsm.return %cond
    }
    fsm.transition @C guard {
      %true = arith.constant true
      %ncond = arith.xori %cond, %true : i1
      fsm.return %ncond
    }
  }
  fsm.state @B transitions {
    fsm.transition @A
  }
  fsm.state @C transitions {
    fsm.transition @A
  }
}

// Negative: the second guard (`!b`) is not the complement of the first (`a`),
// so nothing is eliminated and the comb.xor remains.
// CHECK-LABEL: fsm.machine @not_complementary
// CHECK: comb.xor
// CHECK-LABEL: fsm.machine @three_transitions
fsm.machine @not_complementary(%a: i1, %b: i1) attributes {initialState = "A"} {
  fsm.state @A transitions {
    fsm.transition @B guard {
      fsm.return %a
    }
    fsm.transition @C guard {
      %true = hw.constant true
      %nb = comb.xor %b, %true : i1
      fsm.return %nb
    }
  }
  fsm.state @B transitions {
    fsm.transition @A
  }
  fsm.state @C transitions {
    fsm.transition @A
  }
}

// Negative: the elimination is conservatively scoped to states with exactly two
// transitions, so a state with three transitions is left untouched even when
// the first two are complementary.
// CHECK-LABEL: fsm.machine @three_transitions
// CHECK: comb.xor
fsm.machine @three_transitions(%cond: i1) attributes {initialState = "A"} {
  fsm.state @A transitions {
    fsm.transition @B guard {
      fsm.return %cond
    }
    fsm.transition @C guard {
      %true = hw.constant true
      %ncond = comb.xor %cond, %true : i1
      fsm.return %ncond
    }
    fsm.transition @D guard {
      fsm.return %cond
    }
  }
  fsm.state @B transitions {
    fsm.transition @A
  }
  fsm.state @C transitions {
    fsm.transition @A
  }
  fsm.state @D transitions {
    fsm.transition @A
  }
}
