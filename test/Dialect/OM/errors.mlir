// RUN: circt-opt -om-verify-object-fields %s -verify-diagnostics -split-input-file

om.class @Class() {
  // expected-error @+1 {{'om.object' op result type ("Bar") does not match referred to class ("Foo")}}
  %0 = om.object @Foo() : () -> !om.class.type<@Bar>
}

// -----

om.class @Class() {
  // expected-error @+1 {{'om.object' op refers to non-existant class ("NonExistant")}}
  %0 = om.object @NonExistant() : () -> !om.class.type<@NonExistant>
}

// -----

// expected-note @+1 {{formal parameters:}}
om.class @Class1(%param : i1) {}

om.class @Class2() {
  // expected-error @+2 {{'om.object' op actual parameter list doesn't match formal parameter list}}
  // expected-note @+1 {{actual parameters:}}
  %0 = om.object @Class1() : () -> !om.class.type<@Class1>
}

// -----

om.class @Class1(%param : i1) {}

om.class @Class2(%param : i2) {
  // expected-error @+1 {{'om.object' op actual parameter type ('i2') doesn't match formal parameter type ('i1')}}
  %1 = om.object @Class1(%param) : (i2) -> !om.class.type<@Class1>
}

// -----

// expected-note @+1 {{class defined here}}
om.class @Class1() {}

om.class @Class2() {
  %0 = om.object @Class1() : () -> !om.class.type<@Class1>
  // expected-error @+1 {{'om.object.field' op referenced non-existent field @foo}}
  om.object.field %0, [@foo] : (!om.class.type<@Class1>) -> i1
}

// -----

om.class @Class1() {
  %0 = om.constant 1 : i1
  om.class.field @foo, %0 : i1
}

om.class @Class2() {
  %0 = om.object @Class1() : () -> !om.class.type<@Class1>
  // expected-error @+1 {{'om.object.field' op nested field access into @foo requires a ClassType, but found 'i1'}}
  om.object.field %0, [@foo, @bar] : (!om.class.type<@Class1>) -> i1
}

// -----

om.class @Class1() {
  %0 = om.constant 1 : i1
  om.class.field @foo, %0 : i1
}

om.class @Class2(%arg0: i1) {
  %0 = om.object @Class1() : () -> !om.class.type<@Class1>
  // expected-error @+1 {{'om.object.field' op expected type 'i2', but accessed field has type 'i1'}}
  om.object.field %0, [@foo] : (!om.class.type<@Class1>) -> i2
}

// -----

om.class.extern @Extern(%param1: i1) {
  // expected-error @+1 {{'om.constant' op not allowed in external class}}
  %0 = om.constant 0 : i1
  om.class.extern.field @field1 : i1
}

// -----

// CHECK-LABEL: @List
om.class @List() {
  // expected-error @+1 {{an element of a list attribute must have a type 'i32' but got 'i64'}}
  %0 = om.constant #om.list< i32, [42 : i64]> : !om.list<i32>
  om.class.field @list, %0 : !om.list<i32>
}

// -----

// CHECK-LABEL: @ListCreate
om.class @ListCreate() {
  %0 = om.constant 0 : i64
  %1 = om.constant 1 : i32
  // expected-error @+2 {{use of value '%1' expects different type than prior uses: 'i64' vs 'i32'}}
  // expected-note @-2 {{prior use here}}
  %lst = om.list_create %0, %1 : i64
}

// -----

// expected-error @+1 {{map key type must be either string or integer but got '!om.list<!om.string>'}}
om.class @Map(%map: !om.map<!om.list<!om.string>, !om.string>) {
}

// -----

om.class @Tuple(%tuple: tuple<i1, !om.string>) {
  // expected-error @+1 {{tuple index out-of-bounds, must be less than 2 but got 2}}
  %val = om.tuple_get %tuple[2]  : tuple<i1, !om.string>
}

// -----

om.class @MapConstant() {
  // expected-error @+1 {{a value of a map attribute must have a type 'i64' but field "b" has '!om.list<i32>'}}
  %0 = om.constant #om.map<i64, {a = 42, b = #om.list<i32, []>}> : !om.map<!om.string, i64>
}

// -----

om.class @Thing() { }
om.class @BadPath(%basepath: !om.basepath) {
  // expected-error @below {{invalid symbol reference}}
  %0 = om.path_create reference %basepath @Thing
}


// -----

om.class @DupField(%0: i1) {
  // expected-note @+1 {{previous definition is here}}
  om.class.field @foo, %0 : i1
  // expected-error @+1 {{'om.class.field' op field "foo" is defined twice}}
  om.class.field @foo, %0 : i1
}

// -----

om.class @UnknownClass(%arg: !om.class.type<@Unknwon>) {
  // expected-error @+1 {{class @Unknwon was not found}}
  om.object.field %arg, [@unknown]: (!om.class.type<@Unknwon>) -> i1
}
