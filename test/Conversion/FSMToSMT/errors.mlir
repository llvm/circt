// RUN: circt-opt -split-input-file -convert-fsm-to-smt -verify-diagnostics %s

// -----

fsm.machine @comb_out(%arg0: i8) -> (i8) attributes {initialState = "S0"} {
  %c0_i8 = hw.constant 0 : i8
  // expected-error @below {{Only fsm operations and hw.constants are allowed in the top level of the fsm.machine op.}}
  %add = comb.add %arg0, %c0_i8 : i8
  fsm.state @S0 output {
    fsm.output %arg0 : i8
  }
}

// -----

// expected-error @below {{Operation hw.constant is declared outside of any fsm.machine op}}
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

fsm.machine @out() -> (i1) attributes {initialState = "A"} {
  fsm.state @A output  {
    %x = hw.constant true
    // expected-error @below {{Only fsm, comb, hw, and verif.assert operations are handled in the output region of a state.}}
    %tc = seq.to_clock %x
    %fc = seq.from_clock %tc
    verif.assert %fc : i1
    fsm.output %x : i1
  } transitions {
    fsm.transition @A
  }
}

// -----

fsm.machine @guard() -> () attributes {initialState = "A"} {
  %x = hw.constant true
  fsm.state @A output  {
  } transitions {
    fsm.transition @A guard {
      // expected-error @below {{Only fsm, comb, hw, and verif.assert operations are handled in the guard region of a transition.}}
      %tc = seq.to_clock %x
      %fc = seq.from_clock %tc
      verif.assert %fc : i1
      fsm.return %x
    } action {
      
    }
  }
}

// -----

fsm.machine @action() -> () attributes {initialState = "A"} {
  %x = hw.constant true
  fsm.state @A output  {
  } transitions {
    fsm.transition @A guard {
    } action {
      // expected-error @below {{Only fsm, comb, hw, and verif.assert operations are handled in the action region of a transition.}}
      %tc = seq.to_clock %x
      %fc = seq.from_clock %tc
      verif.assert %fc : i1
    }
  }
}
