// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module private @test_extract(in %arg0: i4) {
  // expected-error @+1 {{'comb.extract' op from bit too large for input}}
  %a = comb.extract %arg0 from 6 : (i4) -> i3
}

// -----

hw.module private @test_extract(in %arg0: i4) {
  // expected-error @+1 {{'comb.extract' op from bit too large for input}}
  %b = comb.extract %arg0 from 2 : (i4) -> i3
}

// -----

hw.module private @test_and() {
  // expected-error @+1 {{'comb.and' op expected 1 or more operands}}
  %b = comb.and : i111
}

// -----

hw.module @InnerSymVisibility() {
  // expected-error @+1 {{expected 'public', 'private', or 'nested'}}
  %wire = hw.wire %wire sym [<@x, 1, oops>] : i1
}

// -----

func.func @notModule () {
  return
}

hw.module @A(in %arg0: i1) {
  // expected-error @+1 {{symbol reference 'notModule' isn't a module}}
  hw.instance "foo" @notModule(a: %arg0: i1) -> ()
}

// -----

hw.module @A(in %arg0: i1) {
  // expected-error @+1 {{Cannot find module definition 'doesNotExist'}}
  hw.instance "b1" @doesNotExist(a: %arg0: i1) -> ()
}

// -----

hw.generator.schema @S, "Test Schema", ["test"]
// expected-error @+1 {{Cannot find generator definition 'S2'}}
hw.module.generated @A, @S2(in %arg0: i1, out a: i1) attributes { test = 1 }

// -----

hw.module @S() { }
// expected-error @+1 {{which is not a HWGeneratorSchemaOp}}
hw.module.generated @A, @S(in %arg0: i1, out a: i1) attributes { test = 1 }


// -----

// expected-error @+1 {{'hw.output' op must have same number of operands as region results}}
hw.module @A(out "": i1) { }

// -----

// expected-error @+1 {{expected non-function type}}
hw.module private @arrayDims(in %a: !hw.array<3 x 4 x i5>) { }

// -----

// expected-error @+1 {{invalid element for hw.inout type}}
hw.module private @invalidInout(in %arg0: !hw.inout<tensor<*xf32>>) { }

// -----

hw.module @inout(in %a: i42) {
  // expected-error @+1 {{'input' must be InOutType, but got 'i42'}}
  %aget = sv.read_inout %a: i42
}

// -----

hw.module @wire(in %a: i42) {
  // expected-error @+1 {{'sv.wire' op result #0 must be InOutType, but got 'i42'}}
  %aget = sv.wire: i42
}

// -----

hw.module @struct(in %a: i42) {
  // expected-error @+1 {{custom op 'hw.struct_create' expected !hw.struct type or alias}}
  %aget = hw.struct_create(%a) : i42
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_explode' invalid kind of type specified}}
  %aget = hw.struct_explode %a : i42
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_extract' invalid kind of type specified}}
  %aget = hw.struct_extract %a["foo"] : i42
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.struct_extract' field name 'bar' not found in aggregate type}}
  %aget = hw.struct_extract %a["bar"] : !hw.struct<foo: i42>
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i32, bar: i18>) {
  // expected-error @+1 {{'hw.struct_extract' op field index 2 exceeds element count of aggregate type}}
  %0 = "hw.struct_extract"(%a) {fieldIndex = 2 : i32} : (!hw.struct<foo: i32, bar: i18>) -> i18
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i32, bar: i18>) {
  // expected-error @+1 {{'hw.struct_extract' op type 'i18' of accessed field in aggregate at index 1 does not match expected type 'i19'}}
  %0 = "hw.struct_extract"(%a) {fieldIndex = 1 : i32} : (!hw.struct<foo: i32, bar: i18>) -> i19
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i42>, in %b: i42) {
  // expected-error @+1 {{custom op 'hw.struct_inject' invalid kind of type specified}}
  %aget = hw.struct_inject %a["foo"], %b : i42
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i42>, in %b: i42) {
  // expected-error @+1 {{custom op 'hw.struct_inject' field name 'bar' not found in aggregate type}}
  %aget = hw.struct_inject %a["bar"], %b : !hw.struct<foo: i42>
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i32, bar: i18>, in %b: i18) {
  // expected-error @+1 {{'hw.struct_inject' op field index 2 exceeds element count of aggregate type}}
  %0 = "hw.struct_inject"(%a, %b) {fieldIndex = 2 : i32} : (!hw.struct<foo: i32, bar: i18>, i18) -> !hw.struct<foo: i32, bar: i18>
}

// -----

hw.module @struct(in %a: !hw.struct<foo: i32, bar: i18>, in %b: i42) {
  // expected-error @+1 {{'hw.struct_inject' op type 'i18' of accessed field in aggregate at index 1 does not match expected type 'i42'}}
  %0 = "hw.struct_inject"(%a, %b) {fieldIndex = 1 : i32} : (!hw.struct<foo: i32, bar: i18>, i42) -> !hw.struct<foo: i32, bar: i18>
}
// -----

hw.module @union(in %a: i42) {
  // expected-error @+1 {{custom op 'hw.union_create' cannot find union field 'bar'}}
  %u = hw.union_create "bar", %a : !hw.union<foo: i42>
}

// -----

hw.module @union(in %a: i42) {
  // expected-error @+1 {{'hw.union_create' op field index 1 exceeds element count of aggregate type}}
  %0 = "hw.union_create"(%a) {fieldIndex = 1 : i32} : (i42) -> !hw.union<foo: i42>
}

// -----

hw.module @union(in %a: i12) {
  // expected-error @+1 {{'hw.union_create' op type 'i42' of accessed field in aggregate at index 0 does not match expected type 'i12'}}
  %0 = "hw.union_create"(%a) {fieldIndex = 0 : i32} : (i12) -> !hw.union<foo: i42, bar: i12>
}

// -----

hw.module @union(in %a: !hw.union<foo: i42>) {
  // expected-error @+1 {{custom op 'hw.union_extract' field name 'bar' not found in aggregate type}}
  %aget = hw.union_extract %a["bar"] : !hw.union<foo: i42>
}
// -----

hw.module @union(in %a: !hw.union<foo: i42>) {
  // expected-error @+2 {{'hw.union_extract' op failed to infer returned types}}
  // expected-error @+1 {{field index 1 exceeds element count of aggregate type}}
  %aget = "hw.union_extract"(%a) {fieldIndex = 1 : i32} : (!hw.union<foo: i42>) -> i42
}
// -----

hw.module @union(in %a: !hw.union<foo: i42, bar: i12>) {
  // expected-error @+2 {{'hw.union_extract' op failed to infer returned types}}
  // expected-error @+1 {{'hw.union_extract' op inferred type(s) 'i12' are incompatible with return type(s) of operation 'i42'}}
  %aget = "hw.union_extract"(%a) {fieldIndex = 1 : i32} : (!hw.union<foo: i42, bar: i12>) -> i42
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @empty() {
  hw.output
}

hw.module @test() {
  // expected-error @+1 {{'hw.instance' op has a wrong number of results; expected 0 but got 3}}
  %0, %1, %3 = hw.instance "test" @empty() -> (a: i2, b: i2, c: i2)
  hw.output
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @f(out a: i2) {
  %a = hw.constant 1 : i2
  hw.output %a : i2
}

hw.module @test() {
  // expected-error @+1 {{'hw.instance' op result type #0 must be 'i2', but got 'i1'}}
  %0 = hw.instance "test" @f() -> (a: i1)
  hw.output
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @empty() {
  hw.output
}

hw.module @test(in %a: i1) {
  // expected-error @+1 {{'hw.instance' op has a wrong number of operands; expected 0 but got 1}}
  hw.instance "test" @empty(a: %a: i1) -> ()
  hw.output
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @f(in %a: i1) {
  hw.output
}

hw.module @test(in %a: i2) {
  // expected-error @+1 {{'hw.instance' op operand type #0 must be 'i1', but got 'i2'}}
  hw.instance "test" @f(a: %a: i2) -> ()
  hw.output
}


// -----

// expected-note @+1 {{module declared here}}
hw.module @f(in %a: i1) {
  hw.output
}

hw.module @test(in %a: i1) {
  // expected-error @+1 {{'hw.instance' op input label #0 must be "a", but got "b"}}
  hw.instance "test" @f(b: %a: i1) -> ()
  hw.output
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @p<p1: i42 = 17, p2: i1>(in %arg0: i8, out out: i8)

hw.module @Use(in %a: i8, out xx: i8) {
  // expected-error @+1 {{op expected 2 parameters but had 1}}
  %r0 = hw.instance "inst1" @p<p1: i42 = 4>(arg0: %a: i8) -> (out: i8)
  hw.output %r0: i8
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @p<p1: i42 = 17, p2: i1>(in %arg0: i8, out out: i8)

hw.module @Use(in %a: i8, out xx: i8) {
  // expected-error @+1 {{op parameter #1 should have name "p2" but has name "p3"}}
  %r0 = hw.instance "inst1" @p<p1: i42 = 4, p3: i1 = 0>(arg0: %a: i8) -> (out: i8)
  hw.output %r0: i8
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @p<p1: i42 = 17, p2: i1>(in %arg0: i8, out out: i8)

hw.module @Use(in %a: i8, out xx: i8) {
  // expected-error @+1 {{op parameter "p2" should have type 'i1' but has type 'i2'}}
  %r0 = hw.instance "inst1" @p<p1: i42 = 4, p2: i2 = 0>(arg0: %a: i8) -> (out: i8)
  hw.output %r0: i8
}

// -----

hw.module.extern @p<p1: i42 = 17, p2: i1>(in %arg0: i8, out out: i8)

hw.module @Use(in %a: i8, out xx: i8) {
  // expected-error @+1 {{op parameter "p2" must have a value}}
  %r0 = hw.instance "inst1" @p<p1: i42 = 4, p2: i1>(arg0: %a: i8) -> (out: i8)
  hw.output %r0: i8
}

// -----
// Check attribute validity for parameters.

hw.module.extern @p<p: i42>()

// expected-note @+1 {{module declared here}}
hw.module @Use() {
  // expected-error @+1 {{op use of unknown parameter "FOO"}}
  hw.instance "inst1" @p<p: i42 = #hw.param.decl.ref<"FOO">>() -> ()
}

// -----
// Check attribute validity for parameters.

hw.module.extern @p<p: i42>()

// expected-note @+1 {{module declared here}}
hw.module @Use<xx: i41>() {
  // expected-error @+1 {{op parameter "xx" used with type 'i42'; should have type 'i41'}}
  hw.instance "inst1" @p<p: i42 = #hw.param.decl.ref<"xx">>() -> ()
}


// -----
// Check attribute validity for module parameters.

// expected-error @+1 {{op parameter "p" cannot be used as a default value for a parameter}}
hw.module.extern @p<p: i42 = #hw.param.decl.ref<"p">>()

// -----

// expected-note @+1 {{module declared here}}
hw.module @Use<xx: i41>() {
  // expected-error @+1 {{'hw.param.value' op parameter "xx" used with type 'i40'; should have type 'i41'}}
  %0 = hw.param.value i40 = #hw.param.decl.ref<"xx">
}

// -----

// expected-error @+1 {{parameter #hw.param.decl<"xx": i41> : i41 has the same name as a previous parameter}}
hw.module @Use<xx: i41, xx: i41>() {}

// -----

// expected-error @+1 {{parameter #hw.param.decl<"xx": i41 = 1> : i41 has the same name as a previous parameter}}
hw.module @Use<xx: i41, xx: i41 = 1>() {}

// -----

// expected-error @+1 {{parameter #hw.param.decl<"xx": none> has the same name as a previous parameter}}
hw.module @Use<xx: none, xx: none>() {}

// -----

module {
  hw.module @A(in %a : !hw.int<41>, out out: !hw.int<42>) {
// expected-error @+1 {{'hw.instance' op operand type #0 must be 'i42', but got 'i41'}}
    %r0 = hw.instance "inst1" @parameters<p1: i42 = 42>(arg0: %a: !hw.int<41>) -> (out: !hw.int<42>)
    hw.output %r0: !hw.int<42>
  }
// expected-note @+1 {{module declared here}}
  hw.module.extern @parameters<p1: i42>(in %arg0: !hw.int<#hw.param.decl.ref<"p1">>, out out: !hw.int<#hw.param.decl.ref<"p1">>)
}

// -----

module {
  hw.module @A(in %a : !hw.int<42>, out out: !hw.int<41>) {
// expected-error @+1 {{'hw.instance' op result type #0 must be 'i42', but got 'i41'}}
    %r0 = hw.instance "inst1" @parameters<p1: i42 = 42>(arg0: %a: !hw.int<42>) -> (out: !hw.int<41>)
    hw.output %r0: !hw.int<41>
  }
// expected-note @+1 {{module declared here}}
  hw.module.extern @parameters<p1: i42>(in %arg0: !hw.int<#hw.param.decl.ref<"p1">>, out out: !hw.int<#hw.param.decl.ref<"p1">>)
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @submodule (out out0: i32)

hw.module @wrongResultLabel() {
  // expected-error @+1 {{result label #0 must be "out0", but got "o"}}
  %inst0.out0 = hw.instance "inst0" @submodule () -> (o: i32)
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @submodule (out out0: i32)

hw.module @wrongNumberOfResultNames() {
  // expected-error @+1 {{has a wrong number of results port labels; expected 1 but got 0}}
  "hw.instance"() {instanceName="inst0", moduleName=@submodule, argNames=[], resultNames=[], parameters=[]} : () -> i32
}

// -----

// expected-note @+1 {{module declared here}}
hw.module.extern @submodule (in %arg0: i32)

hw.module @wrongNumberOfInputNames(in %arg0: i32) {
  // expected-error @+1 {{has a wrong number of input port names; expected 1 but got 0}}
  "hw.instance"(%arg0) {instanceName="inst0", moduleName=@submodule, argNames=[], resultNames=[], parameters=[]} : (i32) -> ()
}

// -----

// expected-error @+1 {{unsupported dimension kind in hw.array}}
hw.module @bab<param: i32, N: i32> (in %array2d: !hw.array<i3 x i4>) {}

// -----

hw.module @aggConstDimArray() {
  // expected-error @+1 {{'hw.aggregate_constant' op array attribute (1) has wrong size for array constant (2)}}
  %0 = hw.aggregate_constant [42 : i8] : !hw.array<2xi8>
}

// -----

hw.module @aggConstDimUArray() {
  // expected-error @+1 {{'hw.aggregate_constant' op array attribute (1) has wrong size for unpacked array constant (2)}}
  %0 = hw.aggregate_constant [42 : i8] : !hw.uarray<2xi8>
}

// -----

hw.module @aggConstDimStruct() {
  // expected-error @+1 {{'hw.aggregate_constant' op array attribute (1) has wrong size for struct constant (2)}}
  %0 = hw.aggregate_constant [42 : i8] : !hw.struct<foo: i8, bar: i8>
}

// -----

hw.module @foo() {
  // expected-error @+1 {{enum value 'D' is not a member of enum type '!hw.enum<A, B, C>'}}
  %0 = hw.enum.constant D : !hw.enum<A, B, C>
  hw.output
}

// -----

hw.module @foo() {
  // expected-error @+1 {{return type '!hw.enum<A, B>' does not match attribute type #hw.enum.field<A, !hw.enum<A>>}}
  %0 = "hw.enum.constant"() {field = #hw.enum.field<A, !hw.enum<A>>} : () -> !hw.enum<A, B>
  hw.output
}

// -----

hw.module @foo() {
  %0 = hw.enum.constant A : !hw.enum<A>
  %1 = hw.enum.constant B : !hw.enum<B>
  // expected-error @+1 {{types do not match}}
  %2 = hw.enum.cmp %0, %1 : !hw.enum<A>, !hw.enum<B>
  hw.output
}

// -----

// expected-error @+2 {{duplicate field name 'foo'}}
// expected-error @+1 {{duplicate field name 'bar'}}
hw.module @struct(in %a: !hw.struct<foo: i8, bar: i8, foo: i8, baz: i8, bar: i8>) {}

// -----

// expected-error @+2 {{duplicate field name 'foo' in hw.union type}}
// expected-error @+1 {{duplicate field name 'bar' in hw.union type}}
hw.module @union(in %a: !hw.union<foo: i8, bar: i8, foo: i8, baz: i8, bar: i8>) {}

// -----

// Test that nested symbol tables fail in the correct way when trying to use an
// instance to escape its containing table.

// If you make a change and this check isn't triggered, then you are breaking
// SymbolTable semantics.

hw.module @Foo () {}

builtin.module @Nested {
  hw.module @Bar () {
    // expected-error @+1 {{Cannot find module definition 'Foo'}}
    hw.instance "inst" @Foo () -> ()
  }
}

// -----

hw.module @Foo () {
  // expected-error @+1 {{Cannot find module definition 'DoesNotExist'}}
  hw.instance_choice "inst" option "foo" @DoesNotExist () -> ()
}

// -----

// Don't crash if hw.array attribute fails to parse as integer
// expected-error @below {{floating point value not valid for specified type}}
hw.module @arrayTypeError(in %in: !hw.array<44.44axi0>) { }

// -----

// Don't crash if element type of inout fails to parse.
hw.module @elementTypeError() {
  // expected-error @below {{expected ':'}}
  "builtin.unrealized_conversion_cast"() : () -> !hw.inout<struct<foo>>
}

// -----

hw.module @elementTypeError() {
  // expected-error @below {{'hw.aggregate_constant' op unknown element type '!seq.clock'}}
  %0 = hw.aggregate_constant [#hw.output_file<"dummy.sv">] : !hw.array<1x!seq.clock>
}

// -----

hw.module @elementTypeError() {
  // expected-error @below {{'hw.aggregate_constant' op typed attr doesn't match the return type '!seq.clock'}}
  %0 = hw.aggregate_constant [32: i16] : !hw.array<1x!seq.clock>
}

// -----

// expected-error @+1 {{inner reference must have exactly one nested reference}}
#innerRef = #hw.innerNameRef<@innerRef>

// -----
%0 = unrealized_conversion_cast to !hw.array<1000xi42>
%1 = hw.constant 0 : i9
// expected-error @below {{index bit width equals ceil(log2(length(input))), or 0 or 1 if input contains only one element}}
hw.array_get %0[%1] : !hw.array<1000xi42>, i9

// -----
%0 = unrealized_conversion_cast to !hw.array<1000xi42>
%1 = hw.constant 0 : i9
%2 = hw.constant 0 : i42
// expected-error @below {{index bit width equals ceil(log2(length(input))), or 0 or 1 if input contains only one element}}
hw.array_inject %0[%1], %2 : !hw.array<1000xi42>, i9
