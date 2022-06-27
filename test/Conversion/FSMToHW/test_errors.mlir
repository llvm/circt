// RUN: circt-opt -split-input-file -convert-fsm-to-hw -verify-diagnostics %s


fsm.machine @foo(%arg0: i1) -> () attributes {initialState = "A"} {
  // expected-error@+1 {{'fsm.variable' op FSM variables not yet supported for HW lowering.}}
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state "A" output  {
    fsm.output
  } transitions {
    fsm.transition @A
  }

}

// -----

fsm.machine @foo(%arg0: i1) -> (i1) attributes {initialState = "A"} {
  // expected-error@+1 {{'arith.constant' op is unsupported (op from the arith dialect).}}
  %true = arith.constant true
  fsm.state "A" output  {
    fsm.output %true : i1
  } transitions {
    fsm.transition @A
  }

  fsm.state "B" output  {
    fsm.output %true : i1
  } transitions {
    fsm.transition @A
  }
}
