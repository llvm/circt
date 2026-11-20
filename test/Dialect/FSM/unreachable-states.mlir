// RUN: circt-opt --split-input-file --fsm-remove-unreachable-states %s | FileCheck %s

// A state with no incoming transition is removed.

// CHECK-LABEL: fsm.machine @unreferenced
// CHECK:         fsm.state @IDLE
// CHECK:         fsm.state @B
// CHECK-NOT:     fsm.state
fsm.machine @unreferenced(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  fsm.state @IDLE output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @B
  }

  fsm.state @A output {
    fsm.output %arg0 : i1
  } transitions {}

  fsm.state @B output {
    fsm.output %arg0 : i1
  } transitions {}

  fsm.state @C output {
    fsm.output %arg0 : i1
  } transitions {}
}

// -----

// A cycle of states that cannot be reached is removed, and so are its
// transitions.

// CHECK-LABEL: fsm.machine @cycle
// CHECK:         fsm.state @IDLE
// CHECK:         fsm.state @A
// CHECK-NOT:     fsm.state
fsm.machine @cycle(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  fsm.state @IDLE output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @A
  }

  fsm.state @A output {
    fsm.output %arg0 : i1
  } transitions {}

  fsm.state @B output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @C
  }

  fsm.state @C output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @B
  }
}

// -----

// The result does not depend on the order the states are declared in. Here
// the states are reached in a different order than they appear in the IR.

// CHECK-LABEL: fsm.machine @order
// CHECK:         fsm.state @IDLE
// CHECK:         fsm.state @Y
// CHECK:         fsm.state @Z
// CHECK-NOT:     fsm.state
fsm.machine @order(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  fsm.state @IDLE output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @Z
  }

  fsm.state @X output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @Y
  }

  fsm.state @Y output {
    fsm.output %arg0 : i1
  } transitions {}

  fsm.state @Z output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @Y
  }
}

// -----

// A machine where every state is reachable is left alone.

// CHECK-LABEL: fsm.machine @all_reachable
// CHECK:         fsm.state @IDLE
// CHECK:         fsm.state @A
// CHECK-NOT:     fsm.state
fsm.machine @all_reachable(%arg0: i1) -> i1 attributes {initialState = "IDLE"} {
  fsm.state @IDLE output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @A
  }

  fsm.state @A output {
    fsm.output %arg0 : i1
  } transitions {
    fsm.transition @IDLE
  }
}
