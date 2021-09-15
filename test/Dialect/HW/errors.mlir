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

hw.module @A(%arg0: i1) {
  // expected-error @+1 {{symbol reference 'notModule' isn't a module}}
  hw.instance "foo" @notModule(a: %arg0: i1) -> ()
}

// -----

hw.module @A(%arg0: i1) {
  // expected-error @+1 {{Cannot find module definition 'doesNotExist'}}
  hw.instance "b1" @doesNotExist(a: %arg0: i1) -> ()
}

// -----

// expected-error @+1 {{'hw.output' op must have same number of operands as region results}}
hw.module @A() -> ("": i1) { }

// -----

// expected-error @+1 {{hw.array only supports one dimension}}
func private @arrayDims(%a: !hw.array<3 x 4 x i5>) { }

// -----

// expected-error @+1 {{invalid element for hw.inout type}}
func private @invalidInout(%arg0: !hw.inout<tensor<*xf32>>) { }

// -----

hw.module @inout(%a: i42) {
  // expected-error @+1 {{'input' must be InOutType, but got 'i42'}}
  %aget = sv.read_inout %a: i42
}

// -----

hw.module @wire(%a: i42) {
  // expected-error @+1 {{'sv.wire' op result #0 must be InOutType, but got 'i42'}}
  %aget = sv.wire: i42
}

// -----

hw.module @struct(%a: i42) {
  // expected-error @+1 {{custom op 'hw.struct_create' expected !hw.struct type or alias}}
  %aget = hw.struct_create(%a) : i42
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_explode' invalid kind of type specified}}
  %aget = hw.struct_explode %a : i42
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_extract' invalid kind of type specified}}
  %aget = hw.struct_extract %a["foo"] : i42
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_extract' invalid field name specified}}
  %aget = hw.struct_extract %a["bar"] : !hw.struct<foo: i42>
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>, %b: i42) {
  // expected-error @+1 {{custom op 'hw.struct_inject' invalid kind of type specified}}
  %aget = hw.struct_inject %a["foo"], %b : i42
}

// -----

hw.module @struct(%a: !hw.struct<foo: i42>, %b: i42) {
  // expected-error @+1 {{custom op 'hw.struct_inject' invalid field name specified}}
  %aget = hw.struct_inject %a["bar"], %b : !hw.struct<foo: i42>
}

// -----

hw.module @union(%b: i42) {
  // expected-error @+1 {{custom op 'hw.union_create' cannot find union field 'bar'}}
  %u = hw.union_create "bar", %a : !hw.union<foo: i42>
}

// -----

hw.module @invalid_add(%a: i0) {  // i0 ports are ok.
  // expected-error @+1 {{'comb.add' op operand #0 must be an integer bitvector of one or more bits, but got 'i0'}}
  %b = comb.add %a, %a: i0
}

// -----

// expected-note @+1 {{original module declared here}}
hw.module @empty() -> () {
  hw.output
}

hw.module @test() -> () {
  // expected-error @+1 {{'hw.instance' op has a wrong number of results; expected 0 but got 3}}
  %0, %1, %3 = hw.instance "test" @empty() -> (a: i2, b: i2, c: i2)
  hw.output
}

// -----

// expected-note @+1 {{original module declared here}}
hw.module @f() -> (a: i2) {
  %a = hw.constant 1 : i2
  hw.output %a : i2
}

hw.module @test() -> () {
  // expected-error @+1 {{'hw.instance' op result type #0 must be 'i2', but got 'i1'}}
  %0 = hw.instance "test" @f() -> (a: i1)
  hw.output
}

// -----

// expected-note @+1 {{original module declared here}}
hw.module @empty() -> () {
  hw.output
}

hw.module @test(%a: i1) -> () {
  // expected-error @+1 {{'hw.instance' op has a wrong number of operands; expected 0 but got 1}}
  hw.instance "test" @empty(a: %a: i1) -> ()
  hw.output
}

// -----

// expected-note @+1 {{original module declared here}}
hw.module @f(%a: i1) -> () {
  hw.output
}

hw.module @test(%a: i2) -> () {
  // expected-error @+1 {{'hw.instance' op operand type #0 must be 'i1', but got 'i2'}}
  hw.instance "test" @f(a: %a: i2) -> ()
  hw.output
}


// -----

// expected-note @+1 {{original module declared here}}
hw.module @f(%a: i1) -> () {
  hw.output
}

hw.module @test(%a: i1) -> () {
  // expected-error @+1 {{'hw.instance' op input label #0 must be "a", but got "b"}}
  hw.instance "test" @f(b: %a: i1) -> ()
  hw.output
}
