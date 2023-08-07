// RUN: circt-opt %s --split-input-file --verify-diagnostics

ibis.class @C(%this) {
  ibis.method @typeMismatch1() -> ui32 {
    // expected-error @+1 {{must return a value}}
    ibis.return
  }
}

// -----

ibis.class @C(%this) {
  ibis.method @typeMismatch2() {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{cannot return a value from a function with no result type}}
    ibis.return %c : i8
  }
}

// -----
ibis.class @C(%this) {
  ibis.method @typeMismatch3() -> ui32 {
    %c = hw.constant 1 : i8
    // expected-error @+1 {{return type ('i8') must match function return type ('ui32')}}
    ibis.return %c : i8
  }
}

// -----

ibis.class @MissingPort(%this) {
  // expected-error @+1 {{'ibis.get_port' op port '@C_in' does not exist in MissingPort}}
  %c_in = ibis.get_port %this, @C_in : !ibis.classref<@MissingPort> -> !ibis.portref<i1>
}

// -----

ibis.class @PortTypeMismatch(%this) {
  ibis.port.input @in : i1
  // expected-error @+1 {{'ibis.get_port' op symbol '@in' refers to a port of type 'i1', but this op has type 'i2'}}
  %c_in = ibis.get_port %this, @in : !ibis.classref<@PortTypeMismatch> -> !ibis.portref<i2>
}
