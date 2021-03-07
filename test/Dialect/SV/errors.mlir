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
      sv.yield
    }
    default: {
      sv.fwrite "z"
      sv.yield
    }
}