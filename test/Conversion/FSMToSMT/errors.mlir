// RUN: circt-opt -split-input-file -convert-fsm-to-smt -verify-diagnostics %s

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
  // expected-error @below {{Only fsm operations and hw.constants are allowed in the top level of the FSM MachineOp.}}
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

// -----

fsm.machine @comb_op() -> () attributes {initialState = "A"} {
  // expected-error @below {{Only integer variables are supported in FSMs.}}
  %var = fsm.variable "var" {initValue = 0.0 : f32} : f32
  fsm.state @A output  {
  } transitions {
  }
}

// -----

// expected-error @below {{Only integer arguments are supported in FSMs.}}
fsm.machine @comb_op(%x0 : f32) -> () attributes {initialState = "A"} {
  fsm.state @A output  {
  } transitions {
  }
}

// -----

// expected-error @below {{Only integer outputs are supported in FSMs.}}
fsm.machine @comb_op() -> (!hw.enum<A, B, C>) attributes {initialState = "A"} {
  fsm.state @A output  {
    %e = hw.enum.constant A : !hw.enum<A, B, C>
    fsm.output %e : !hw.enum<A, B, C>
  } transitions {
  }
}

// -----
