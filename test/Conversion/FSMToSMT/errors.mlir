// RUN: circt-opt -split-input-file -convert-fsm-to-smt -verify-diagnostics %s

// expected-error @below {{'fsm.machine' op initial state 'nonexistent' was not defined in the machine}}
fsm.machine @bad_initial(%arg0: i8) -> (i8) attributes {initialState = "nonexistent"} {
  %c0_i8 = hw.constant 0 : i8
  fsm.state @S0 output {
    fsm.output %c0_i8 : i8
  }
}

// -----

fsm.machine @bad_transition(%arg0: i8) -> (i8) attributes {initialState = "S0"} {
  %c0_i8 = hw.constant 0 : i8
  fsm.state @S0 output {
    fsm.output %c0_i8 : i8
  } transitions {
    // expected-error @below {{'fsm.transition' op cannot find the definition of the next state `nonexistent`}}
    fsm.transition @nonexistent
  }
}

// -----

fsm.machine @missing_init(%arg0: i8) -> (i8) attributes {initialState = "S0"} {
  %c0_i8 = hw.constant 0 : i8
  // expected-error @below {{'fsm.variable' op requires attribute 'initValue'}}
  %counter = fsm.variable "counter" : i32
  fsm.state @S0 output {
    fsm.output %c0_i8 : i8
  }
}


// -----

fsm.machine @comb_out(%arg0: i8) -> (i8) attributes {initialState = "S0"} {
  %c0_i8 = hw.constant 0 : i8
  // expected-error @below {{Operations other than hw.constants are not supported outside FSM output, guard, and action regions.}}
  %add = comb.add %arg0, %c0_i8 : i8
  fsm.state @S0 output {
    fsm.output %arg0 : i8
  }
}

// -----

// expected-error @below {{Operation hw.constant is declared outside of any FSM MachineOp}}
%c0_i8 = hw.constant 0 : i8
fsm.machine @comb_op(%arg0: i8) -> (i8) attributes {initialState = "S0"} {
  %add = comb.add %arg0, %c0_i8 : i8
  fsm.state @S0 output {
    fsm.output %arg0 : i8
  }
}

