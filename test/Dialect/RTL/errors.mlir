// RUN: circt-opt %s -split-input-file -verify-diagnostics

func private @test_constant() -> i32 {
  // expected-error @+1 {{firrtl.constant attribute bitwidth doesn't match return type}}
  %a = rtl.constant(42 : i12) : i32
  return %a : i32
}

// -----

func private @test_extend(%arg0: i4) -> i4 {
  // expected-error @+1 {{extension must increase bitwidth of operand}}
  %a = rtl.sext %arg0 : (i4) -> i4
  return %a : i4
}

// -----

func private @test_extract(%arg0: i4) {
  // expected-error @+1 {{'rtl.extract' op from bit too large for input}}
  %a = rtl.extract %arg0 from 6 : (i4) -> i3
}

// -----

func private @test_extract(%arg0: i4) {
  // expected-error @+1 {{'rtl.extract' op from bit too large for input}}
  %b = rtl.extract %arg0 from 2 : (i4) -> i3
}

// -----

func private @test_and() {
  // expected-error @+1 {{'rtl.and' op expected 1 or more operands}}
  %b = rtl.and : i111
}

// -----

func private @notModule () {
  return
}

rtl.module @A(%arg0: i1) {
  // expected-error @+1 {{Symbol resolved to 'func' which is not a RTL[Ext]ModuleOp}}
  rtl.instance "foo" @notModule(%arg0) : (i1) -> ()
}

// -----

rtl.module @A(%arg0: i1) {
  // expected-error @+1 {{Cannot find module definition 'doesNotExist'}}
  rtl.instance "b1" @doesNotExist(%arg0) : (i1) -> ()
}

// -----

// expected-error @+1 {{'rtl.output' op must have same number of operands as region results}}
rtl.module @A() -> (i1) { }

// -----

rtl.module @A () {}

rtl.module @B() {
  // expected-error @+1 {{has unknown extmodule parameter value 'width' = @Foo}}
  rtl.instance "foo" @A() { parameters = { width = @Foo } }: () -> ()
}

// -----

// expected-error @+1 {{rtl.array only supports one dimension}}
func private @arrayDims(%a: !rtl.array<3 x 4 x i5>) { }

// -----

// expected-error @+1 {{invalid element for rtl.inout type}}
func private @invalidInout(%arg0: !rtl.inout<tensor<*xf32>>) { }

// -----

rtl.module @inout(%a: i42) {
  // expected-error @+1 {{'input' must be InOutType, but got 'i42'}}
  %aget = sv.read_inout %a: i42
}

// -----

rtl.module @wire(%a: i42) {
  // expected-error @+1 {{'sv.wire' op result #0 must be InOutType, but got 'i42'}}
  %aget = sv.wire: i42
}

// -----

rtl.module @struct(%a: i42) {
  // expected-error @+1 {{custom op 'rtl.struct_create' invalid kind of type specified}}
  %aget = rtl.struct_create(%a) : i42
}

// -----

rtl.module @struct(%a: !rtl.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'rtl.struct_explode' invalid kind of type specified}}
  %aget = rtl.struct_explode %a : i42
}

// -----

rtl.module @struct(%a: !rtl.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'rtl.struct_extract' invalid kind of type specified}}
  %aget = rtl.struct_extract %a["foo"] : i42
}

// -----

rtl.module @struct(%a: !rtl.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'rtl.struct_extract' invalid field name specified}}
  %aget = rtl.struct_extract %a["bar"] : !rtl.struct<foo: i42>
}

// -----

rtl.module @struct(%a: !rtl.struct<foo: i42>, %b: i42) {
  // expected-error @+1 {{custom op 'rtl.struct_inject' invalid kind of type specified}}
  %aget = rtl.struct_inject %a["foo"], %b : i42
}

// -----

rtl.module @struct(%a: !rtl.struct<foo: i42>, %b: i42) {
  // expected-error @+1 {{custom op 'rtl.struct_inject' invalid field name specified}}
  %aget = rtl.struct_inject %a["bar"], %b : !rtl.struct<foo: i42>
}

// -----

rtl.module @invalid_add(%a: i0) {  // i0 ports are ok.
  // expected-error @+1 {{'rtl.add' op operand #0 must be an integer bitvector of one or more bits, but got 'i0'}}
  %b = rtl.add %a, %a: i0
}