// RUN: circt-opt %s -split-input-file -verify-diagnostics

rtl.module @test_instance_exist_error() {
  // expected-error @+1 {{Symbol not found: @noexist.}}
  %b = sv.interface.instance : !sv.interface<@noexist>
}

// -----

rtl.module @foo () {  }
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

rtl.module @Aliasing(%a : !rtl.inout<i42>, %b : !rtl.inout<i42>,
                      %c : !rtl.inout<i42>) {

  // ok
  sv.alias %a, %b     : !rtl.inout<i42>, !rtl.inout<i42>

  // expected-error @+1 {{'sv.alias' op alias must have at least two operands}}
  sv.alias %a : !rtl.inout<i42>
}

// -----
rtl.module @Fwrite() {
  // expected-error @+1 {{sv.fwrite should be in a procedural region}}
  sv.fwrite "error"
}

// -----
rtl.module @Bpassign(%arg0: i1) {
  %reg = sv.reg : !rtl.inout<i1>
  // expected-error @+1 {{sv.bpassign should be in a procedural region}}
  sv.bpassign %reg, %arg0 : i1
}

// -----
rtl.module @Passign(%arg0: i1) {
  %reg = sv.reg : !rtl.inout<i1>
  // expected-error @+1 {{sv.passign should be in a procedural region}}
  sv.passign %reg, %arg0 : i1
}

// -----
rtl.module @IfOp(%arg0: i1) {
  // expected-error @+1 {{sv.if should be in a procedural region}}
  sv.if %arg0 {
    sv.fwrite "Foo"
  }
}

// -----
rtl.module @Fatal() {
  // expected-error @+1 {{sv.fatal should be in a procedural region}}
  sv.fatal
}

// -----
rtl.module @Fatal() {
  // expected-error @+1 {{sv.finish should be in a procedural region}}
  sv.finish
}

// -----
rtl.module @CaseZ(%arg8: i8) {
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
rtl.module @Initial() {
  sv.initial {
    // expected-error @+1 {{sv.initial should be in a non-procedural region}}
    sv.initial {}
  }
}

// -----
rtl.module @IfDef() {
  sv.initial {
    // expected-error @+1 {{sv.ifdef should be in a non-procedural region}}
    sv.ifdef "SYNTHESIS" {}
  }
}

// -----
rtl.module @Always(%arg0: i1) {
  sv.initial {
    // expected-error @+1 {{sv.always should be in a non-procedural region}}
    sv.always posedge %arg0 {}
  }
}

// -----
rtl.module @AlwaysFF(%arg0: i1) {
  sv.initial {
    // expected-error @+1 {{sv.alwaysff should be in a non-procedural region}}
    sv.alwaysff (posedge %arg0) {}
  }
}

// -----
rtl.module @Wire() {
  sv.initial {
    // expected-error @+1 {{sv.wire should be in a non-procedural region}}
    %wire = sv.wire : !rtl.inout<i1>
  }
}

// -----
rtl.module @Assert(%arg0: i1) {
  // expected-error @+1 {{sv.assert should be in a procedural region}}
  sv.assert %arg0: i1
}

// -----
rtl.module @Assume(%arg0: i1) {
  // expected-error @+1 {{sv.assume should be in a procedural region}}
  sv.assume %arg0: i1
}

// -----
rtl.module @Cover(%arg0: i1) {
  // expected-error @+1 {{sv.cover should be in a procedural region}}
  sv.cover %arg0: i1
}

// -----
rtl.module.extern @test1(%arg0: i1, %arg1: i1, %arg8: i8)
func @test2(%arg0: i1, %arg1: i1, %arg8: i8) { return }
// expected-error @+1 {{'sv.bind' op attribute 'child' failed to satisfy constraint: flat symbol reference attribute is module like}}
sv.bind "testinst" @test1 @test2
