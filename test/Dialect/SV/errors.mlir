// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @test_instance_exist_error() {
  // expected-error @+1 {{Symbol not found: @noexist.}}
  %b = sv.interface.instance : !sv.interface<@noexist>
}

// -----

hw.module @foo () {  }
// expected-error @+1 {{Symbol @foo is not an InterfaceOp.}}
%b = sv.interface.instance : !sv.interface<@foo>

// -----

sv.interface @foo {
  sv.interface.signal @data : i32
}
%iface = sv.interface.instance : !sv.interface<@foo>
// expected-error @+1 {{Symbol @foo::@data is not an InterfaceModportOp.}}
%b = sv.modport.get %iface @data : !sv.interface<@foo> -> !sv.modport<@foo::@data>

// -----

hw.module @Aliasing(%a : !hw.inout<i42>, %b : !hw.inout<i42>,
                      %c : !hw.inout<i42>) {

  // ok
  sv.alias %a, %b     : !hw.inout<i42>, !hw.inout<i42>

  // expected-error @+1 {{'sv.alias' op alias must have at least two operands}}
  sv.alias %a : !hw.inout<i42>
}

// -----
hw.module @Fwrite() {
  // expected-error @+1 {{sv.fwrite should be in a procedural region}}
  sv.fwrite "error"
}

// -----
hw.module @Bpassign(%arg0: i1) {
  %reg = sv.reg : !hw.inout<i1>
  // expected-error @+1 {{sv.bpassign should be in a procedural region}}
  sv.bpassign %reg, %arg0 : i1
}

// -----
hw.module @Passign(%arg0: i1) {
  %reg = sv.reg : !hw.inout<i1>
  // expected-error @+1 {{sv.passign should be in a procedural region}}
  sv.passign %reg, %arg0 : i1
}

// -----
hw.module @IfOp(%arg0: i1) {
  // expected-error @+1 {{sv.if should be in a procedural region}}
  sv.if %arg0 {
    sv.fwrite "Foo"
  }
}

// -----
hw.module @Fatal() {
  // expected-error @+1 {{sv.fatal should be in a procedural region}}
  sv.fatal
}

// -----
hw.module @Fatal() {
  // expected-error @+1 {{sv.finish should be in a procedural region}}
  sv.finish
}

// -----
hw.module @CaseZ(%arg8: i8) {
  // expected-error @+1 {{sv.casez should be in a procedural region}}
  sv.casez %arg8 : i8
    case b0000001x: {
      sv.fwrite "x"
    }
    default: {
      sv.fwrite "z"
    }
}

// -----
hw.module @Initial() {
  sv.initial {
    // expected-error @+1 {{sv.initial should be in a non-procedural region}}
    sv.initial {}
  }
}

// -----
hw.module @IfDef() {
  sv.initial {
    // expected-error @+1 {{sv.ifdef should be in a non-procedural region}}
    sv.ifdef "SYNTHESIS" {}
  }
}

// -----
hw.module @Always(%arg0: i1) {
  sv.initial {
    // expected-error @+1 {{sv.always should be in a non-procedural region}}
    sv.always posedge %arg0 {}
  }
}

// -----
hw.module @AlwaysFF(%arg0: i1) {
  sv.initial {
    // expected-error @+1 {{sv.alwaysff should be in a non-procedural region}}
    sv.alwaysff (posedge %arg0) {}
  }
}

// -----
hw.module @Wire() {
  sv.initial {
    // expected-error @+1 {{sv.wire should be in a non-procedural region}}
    %wire = sv.wire : !hw.inout<i1>
  }
}

// -----
hw.module @Assert(%arg0: i1) {
  // expected-error @+1 {{sv.assert should be in a procedural region}}
  sv.assert %arg0: i1
}

// -----
hw.module @Assume(%arg0: i1) {
  // expected-error @+1 {{sv.assume should be in a procedural region}}
  sv.assume %arg0: i1
}

// -----
hw.module @Cover(%arg0: i1) {
  // expected-error @+1 {{sv.cover should be in a procedural region}}
  sv.cover %arg0: i1
}