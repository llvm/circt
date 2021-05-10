// RUN: circt-opt %s -split-input-file -verify-diagnostics

func private @test_extend(%arg0: i4) -> i3 {
  // expected-error @+1 {{extension must increase bitwidth of operand}}
  %a = comb.sext %arg0 : (i4) -> i3
  return %a : i3
}

// -----

func private @test_extract(%arg0: i4) {
  // expected-error @+1 {{'comb.extract' op from bit too large for input}}
  %a = comb.extract %arg0 from 6 : (i4) -> i3
}

// -----

func private @test_extract(%arg0: i4) {
  // expected-error @+1 {{'comb.extract' op from bit too large for input}}
  %b = comb.extract %arg0 from 2 : (i4) -> i3
}

// -----

func private @test_and() {
  // expected-error @+1 {{'comb.and' op expected 1 or more operands}}
  %b = comb.and : i111
}

// -----

func private @notModule () {
  return
}

rtl.module @A(%arg0: i1) {
  // expected-error @+1 {{'rtl.instance' op attribute 'moduleName' failed to satisfy constraint: flat symbol reference attribute is module like}}
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

rtl.module @union(%b: i42) {
  // expected-error @+1 {{custom op 'rtl.union_create' cannot find union field 'bar'}}
  %u = rtl.union_create "bar", %a : !rtl.union<foo: i42>
}

// -----

rtl.module @invalid_add(%a: i0) {  // i0 ports are ok.
  // expected-error @+1 {{'comb.add' op operand #0 must be an integer bitvector of one or more bits, but got 'i0'}}
  %b = comb.add %a, %a: i0
}

// -----

// expected-note @+1 {{original module declared here}}
rtl.module @empty() -> () {
  rtl.output
}

rtl.module @test() -> () {
  // expected-error @+1 {{'rtl.instance' op has a wrong number of results; expected 0 but got 3}}
  %0, %1, %3 = rtl.instance "test" @empty() : () -> (i2, i2, i2)
  rtl.output
}

// -----

// expected-note @+1 {{original module declared here}}
rtl.module @f() -> (i2) {
  %a = rtl.constant 1 : i2
  rtl.output %a : i2
}

rtl.module @test() -> () {
  // expected-error @+1 {{'rtl.instance' op #0 result type must be 'i2', but got 'i1'}}
  %0 = rtl.instance "test" @f() : () -> (i1)
  rtl.output
}

// -----

// expected-note @+1 {{original module declared here}}
rtl.module @empty() -> () {
  rtl.output
}

rtl.module @test(%a: i1) -> () {
  // expected-error @+1 {{'rtl.instance' op has a wrong number of operands; expected 0 but got 1}}
  rtl.instance "test" @empty(%a) : (i1) -> ()
  rtl.output
}

// -----

// expected-note @+1 {{original module declared here}}
rtl.module @f(%a: i1) -> () {
  rtl.output
}

rtl.module @test(%a: i2) -> () {
  // expected-error @+1 {{'rtl.instance' op #0 operand type must be 'i1', but got 'i2'}}
  rtl.instance "test" @f(%a) : (i2) -> ()
  rtl.output
}

// -----

// expected-error @+1 {{'rtl.module' op incorrect number of argument names}}
rtl.module @invalidNames(%clock: i1, %a: i1) 
  attributes { argNames = ["x", "y", "z"] } {
}
