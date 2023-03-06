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
  %fd = hw.constant 0x80000002 : i32
  // expected-error @+1 {{sv.fwrite should be in a procedural region}}
  sv.fwrite %fd, "error"
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
hw.module @ForcePassign(%arg0: i1) {
  %reg = sv.reg : !hw.inout<i1>
  // expected-error @+1 {{sv.force should be in a procedural region}}
  sv.force %reg, %arg0 : i1
}

// -----
hw.module @ReleasePassign(%arg0: i1) {
  %reg = sv.reg : !hw.inout<i1>
  // expected-error @+1 {{sv.release should be in a procedural region}}
  sv.release %reg : !hw.inout<i1>
}

// -----
hw.module @IfOp(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32
  // expected-error @+1 {{sv.if should be in a procedural region}}
  sv.if %arg0 {
    sv.fwrite %fd, "Foo"
  }
}

// -----
hw.module @Fatal() {
  // expected-error @+1 {{sv.fatal should be in a procedural region}}
  sv.fatal 1
}

// -----
hw.module @Finish() {
  // expected-error @+1 {{sv.finish should be in a procedural region}}
  sv.finish 1
}

// -----
hw.module @CaseZ(%arg8: i8) {
  %fd = hw.constant 0x80000002 : i32
  // expected-error @+1 {{sv.case should be in a procedural region}}
  sv.case %arg8 : i8
    case b0000001x: {
      sv.fwrite %fd, "x"
    }
    default: {
      sv.fwrite %fd, "z"
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
  sv.assert %arg0, immediate
}

// -----
hw.module @Assume(%arg0: i1) {
  // expected-error @+1 {{sv.assume should be in a procedural region}}
  sv.assume %arg0, immediate
}

// -----
hw.module @Cover(%arg0: i1) {
  // expected-error @+1 {{sv.cover should be in a procedural region}}
  sv.cover %arg0, immediate
}

// -----
// expected-error @+1 {{Referenced instance doesn't exist}}
sv.bind #hw.innerNameRef<@assume::@A>
hw.module @assume() {
  hw.output
}

// -----
// expected-error @+1 {{Referenced module doesn't exist}}
sv.bind #hw.innerNameRef<@NotAModule::@A>

// -----
hw.module.extern @ExternDestMod()
hw.module @InternSrcMod() {
  hw.instance "whatever" sym @A @ExternDestMod() -> ()
  hw.output
}
// expected-error @+1 {{Referenced instance isn't marked as doNotPrint}}
sv.bind #hw.innerNameRef<@InternSrcMod::@A>

// -----

hw.module @test() {
  // expected-error @+1 {{op invalid parameter value @test}}
  %param_x = sv.localparam {value = @test} : i42
}

// -----

hw.module @part_select1() {
  %selWire = sv.wire : !hw.inout<i10>
  %c2 = hw.constant 2 : i3
  // expected-error @+1 {{slice width should not be greater than input width}}
  %xx1 = sv.indexed_part_select_inout %selWire[%c2:11] :  !hw.inout<i10>, i3
}

// -----

hw.module @part_select1() {
  %selWire = sv.wire : !hw.inout<i10>
  %c2 = hw.constant 2 : i3
  %r1 = sv.read_inout %selWire : !hw.inout<i10>
  // expected-error @+1 {{slice width should not be greater than input width}}
  %c = sv.indexed_part_select %r1[%c2 : 20] : i10,i3
}

// -----

hw.module @ZeroWidthConstantX() {
  // expected-error @+1 {{unsupported type}}
  %0 = sv.constantX : !hw.struct<>
}

// -----

hw.module @ZeroWidthConstantZ() {
  // expected-error @+1 {{unsupported type}}
  %0 = sv.constantZ : !hw.struct<>
}

// -----

hw.module @CaseEnum() {
  %0 = hw.enum.constant A : !hw.enum<A, B, C>
  // expected-error @+1 {{custom op 'sv.case' case value 'D' is not a member of enum type '!hw.enum<A, B, C>'}}
  sv.case %0 : !hw.enum<A, B, C>
    case D: {
      sv.fwrite %fd, "x"
    }
}

// -----

hw.module @DPINoFunction(%clk: i1) -> () {
  sv.alwaysff(posedge %clk) {
    // expected-error @+1 {{Cannot find function definition 'no_declared_function'}}
    sv.dpi.call @no_declared_function() : () -> ()
  }
}

// -----

sv.dpi.import @func(%arg0: i32) -> (res0: i5)

hw.module @dpi_invalid_result_count(%clk: i1) -> () {
  %arg0 = hw.constant 0 : i32
  %arg1 = hw.constant 1 : i64
  %fd = hw.constant 0x80000002 : i32
  sv.alwaysff(posedge %clk) {
    // expected-error @+1 {{2 results present, expected 1}}
    %res0, %res1 = sv.dpi.call @func(%arg0) : (i32) -> (i5, i8)
  }
}

// -----

sv.dpi.import @func(%arg0: i32) -> (res0: i5)

hw.module @dpi_invalid_result(%clk: i1) -> () {
  %arg0 = hw.constant 0 : i32
  %arg1 = hw.constant 1 : i64
  %fd = hw.constant 0x80000002 : i32
  sv.alwaysff(posedge %clk) {
    // expected-error @+1 {{invalid result #0: expected 'i5', got 'i7'}}
    %res2 = sv.dpi.call @func(%arg0) : (i32) -> (i7)
  }
}

// -----

sv.dpi.import @func(%arg0: i32) -> (res0: i5)

hw.module @dpi_invalid_argument(%clk: i1) -> () {
  %arg0 = hw.constant 0 : i32
  %arg1 = hw.constant 1 : i64
  %fd = hw.constant 0x80000002 : i32
  sv.alwaysff(posedge %clk) {
    // expected-error @+1 {{invalid argument #0: expected 'i32', got 'i64'}}
    %res3 = sv.dpi.call @func(%arg1) : (i64) -> (i5)
  }
}

// -----

sv.dpi.import @func(%arg0: i32) -> (res0: i5)

hw.module @dpi_invalid_argument_count(%clk: i1) -> () {
  %arg0 = hw.constant 0 : i32
  %arg1 = hw.constant 1 : i64
  %fd = hw.constant 0x80000002 : i32
  sv.alwaysff(posedge %clk) {
    // expected-error @+1 {{2 arguments present, expected 1}}
    %res4 = sv.dpi.call @func(%arg0, %arg1) : (i32, i64) -> (i5)
  }
}

