// RUN: circt-opt --hw-verify-irn --split-input-file --verify-diagnostics %s

ibis.class @MissingPort {
  %this = ibis.this @MissingPort
  // expected-error @+1 {{'ibis.get_port' op port '@C_in' does not exist in @MissingPort}}
  %c_in = ibis.get_port %this, @C_in : !ibis.scoperef<@MissingPort> -> !ibis.portref<in i1>
}

// -----

ibis.class @InvalidGetVar2 {
  %this = ibis.this @InvalidGetVar2
  ibis.var @var : memref<i32>
  ibis.method @foo()  {
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@InvalidGetVar2>>
    ]
    // expected-error @+1 {{'ibis.get_var' op dereferenced type ('memref<i1>') must match variable type ('memref<i32>')}}
    %var = ibis.get_var %parent, @var : !ibis.scoperef<@InvalidGetVar2> -> memref<i1>
    ibis.return
  }
}

// -----

ibis.class @InvalidGetVar {
  %this = ibis.this @InvalidGetVar
  ibis.var @var : memref<i32>
  ibis.method @foo()  {
    %parent = ibis.path [
      #ibis.step<parent : !ibis.scoperef<@InvalidGetVar>>
    ]
    // expected-error @+1 {{'ibis.get_var' op result #0 must be memref of any type values, but got 'i32'}}
    %var = ibis.get_var %parent, @var : !ibis.scoperef<@InvalidGetVar> -> i32
    ibis.return
  }
}

// -----

ibis.class @MissingPort {
  %this = ibis.this @MissingPort
  // expected-error @+1 {{'ibis.get_port' op port '@C_in' does not exist in @MissingPort}}
  %c_in = ibis.get_port %this, @C_in : !ibis.scoperef<@MissingPort> -> !ibis.portref<in i1>
}

// -----

ibis.class @PortTypeMismatch {
  %this = ibis.this @PortTypeMismatch
  ibis.port.input @in : i1
  // expected-error @+1 {{'ibis.get_port' op symbol '@in' refers to a port of type 'i1', but this op has type 'i2'}}
  %c_in = ibis.get_port %this, @in : !ibis.scoperef<@PortTypeMismatch> -> !ibis.portref<in i2>
}
