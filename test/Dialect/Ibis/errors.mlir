// RUN: circt-opt %s --split-input-file --verify-diagnostics

ibis.class @C {
  %this = ibis.this @C
  ibis.method @typeMismatch1() -> ui32 {
    // expected-error @+1 {{must return a value}}
    ibis.return
  }
}

// -----

ibis.class @C {
  %this = ibis.this @C
  ibis.method @typeMismatch2() {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{cannot return a value from a function with no result type}}
    ibis.return %c : i8
  }
}

// -----
ibis.class @C {
  %this = ibis.this @C
  ibis.method @typeMismatch3() -> ui32 {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{return type ('i8') must match function return type ('ui32')}}
    ibis.return %c : i8
  }
}

// -----

ibis.class @MissingPort {
  %this = ibis.this @MissingPort
  // expected-error @+1 {{'ibis.get_port' op port '@C_in' does not exist in MissingPort}}
  %c_in = ibis.get_port %this, @C_in : !ibis.scoperef<@MissingPort> -> !ibis.portref<i1>
}

// -----

ibis.class @PortTypeMismatch {
  %this = ibis.this @PortTypeMismatch
  ibis.port.input @in : i1
  // expected-error @+1 {{'ibis.get_port' op symbol '@in' refers to a port of type 'i1', but this op has type 'i2'}}
  %c_in = ibis.get_port %this, @in : !ibis.scoperef<@PortTypeMismatch> -> !ibis.portref<i2>
}

// -----

// expected-error @+1 {{'ibis.class' op must contain only one 'ibis.this' operation}}
ibis.class @MultipleThis {
  %this = ibis.this @MultipleThis
  %this2 = ibis.this @MultipleThis
}

// -----

// expected-error @+1 {{'ibis.container' op must contain a 'ibis.this' operation}}
ibis.container @NoThis {
}
