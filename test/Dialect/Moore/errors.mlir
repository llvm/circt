// RUN: circt-opt %s --verify-diagnostics --split-input-file

// expected-error @below {{references unknown symbol @doesNotExist}}
moore.instance "b1" @doesNotExist() -> ()

// -----
// expected-error @below {{must reference a 'moore.module', but @Foo is a 'func.func'}}
moore.instance "foo" @Foo() -> ()
func.func @Foo() { return }

// -----
// expected-error @below {{has 0 operands, but target module @Foo has 1 inputs}}
moore.instance "foo" @Foo() -> ()
moore.module @Foo(in %a: !moore.i42) {}

// -----
// expected-error @below {{has 0 results, but target module @Foo has 1 outputs}}
moore.instance "foo" @Foo() -> ()
moore.module @Foo(out a: !moore.i42) {
  %0 = moore.constant 42 : i42
  moore.output %0 : !moore.i42
}

// -----
%0 = moore.constant 42 : i32
// expected-error @below {{operand 0 ('!moore.i32') does not match input type ('!moore.string') of module @Foo}}
moore.instance "foo" @Foo(a: %0: !moore.i32) -> ()
moore.module @Foo(in %a: !moore.string) {}

// -----
// expected-error @below {{result 0 ('!moore.i32') does not match output type ('!moore.i42') of module @Foo}}
moore.instance "foo" @Foo() -> (a: !moore.i32)
moore.module @Foo(out a: !moore.i42) {
  %0 = moore.constant 42 : i42
  moore.output %0 : !moore.i42
}

// -----

moore.module @Foo() {
  %0 = moore.constant 42 : i32
  // expected-error @below {{op has 1 operands, but enclosing module @Foo has 0 outputs}}
  moore.output %0 : !moore.i32
}

// -----

moore.module @Foo(out a: !moore.string) {
  %0 = moore.constant 42 : i32
  // expected-error @below {{op operand 0 ('!moore.i32') does not match output type ('!moore.string') of module @Foo}}
  moore.output %0 : !moore.i32
}

// -----

// expected-error @below {{value requires 6 bits, but result type only has 1}}
moore.constant 42 : !moore.i1

// -----

// expected-error @below {{value requires 2 bits, but result type only has 1}}
moore.constant -2 : !moore.i1

// -----

// expected-error @below {{value contains X or Z bits, but result type '!moore.i4' only allows two-valued bits}}
moore.constant b10XZ : !moore.i4

// -----

// expected-error @below {{attribute width 9 does not match return type's width 8}}
"moore.constant" () {value = #moore.fvint<42 : 9>} : () -> !moore.i8

// -----

%0 = moore.constant 0 : i8
// expected-error @below {{'moore.yield' op expects parent op to be one of 'moore.conditional, moore.global_variable'}}
moore.yield %0 : i8

// -----

%0 = moore.constant 1 : i1
%1 = moore.constant 42 : i8
%2 = moore.constant 42 : i32

moore.conditional %0 : i1 -> i32 {
  // expected-error @below {{yields '!moore.i8', but parent expects '!moore.i32'}}
  moore.yield %1 : i8
} {
  moore.yield %2 : i32
}

// -----

%0 = moore.constant 1 : i1
%1 = moore.constant 42 : i32
%2 = moore.constant 42 : i8

moore.conditional %0 : i1 -> i32 {
  moore.yield %1 : i32
} {
  // expected-error @below {{yields '!moore.i8', but parent expects '!moore.i32'}}
  moore.yield %2 : i8
}

// -----

%0 = moore.constant 42 : i32
// expected-error @below {{op has 1 operands, but result type requires 2}}
moore.struct_create %0 : !moore.i32 -> struct<{a: i32, b: i32}>

// -----

%0 = moore.constant 42 : i32
// expected-error @below {{op operand #0 has type '!moore.i32', but struct field "a" requires '!moore.i1337'}}
moore.struct_create %0 : !moore.i32 -> struct<{a: i1337}>

// -----

%0 = unrealized_conversion_cast to !moore.struct<{a: i32}>
// expected-error @below {{op extracts field "b" which does not exist}}
moore.struct_extract %0, "b" : struct<{a: i32}> -> i9001

// -----

%0 = unrealized_conversion_cast to !moore.struct<{a: i32}>
// expected-error @below {{op result type '!moore.i9001' must match struct field type '!moore.i32'}}
moore.struct_extract %0, "a" : struct<{a: i32}> -> i9001

// -----

%0 = unrealized_conversion_cast to !moore.ref<struct<{a: i32}>>
// expected-error @below {{op extracts field "b" which does not exist}}
moore.struct_extract_ref %0, "b" : <struct<{a: i32}>> -> <i9001>

// -----

%0 = unrealized_conversion_cast to !moore.ref<struct<{a: i32}>>
// expected-error @below {{op result ref of type '!moore.i9001' must match struct field type '!moore.i32'}}
moore.struct_extract_ref %0, "a" : <struct<{a: i32}>> -> <i9001>

// -----

%0 = unrealized_conversion_cast to !moore.struct<{a: i32}>
%1 = moore.constant 42 : i32
// expected-error @below {{op injects field "b" which does not exist}}
moore.struct_inject %0, "b", %1 : struct<{a: i32}>, i32

// -----

%0 = unrealized_conversion_cast to !moore.struct<{a: i32}>
%1 = moore.constant 42 : i9001
// expected-error @below {{op injected value '!moore.i9001' must match struct field type '!moore.i32'}}
moore.struct_inject %0, "a", %1 : struct<{a: i32}>, i9001

// -----

// expected-error @below {{references unknown symbol @doesNotExist}}
moore.get_global_variable @doesNotExist : <i42>

// -----

// expected-error @below {{must reference a 'moore.global_variable', but @Foo is a 'func.func'}}
moore.get_global_variable @Foo : <i42>
func.func @Foo() { return }

// -----

// expected-error @below {{returns a '!moore.i42' reference, but @Foo is of type '!moore.i9001'}}
moore.get_global_variable @Foo : <i42>
moore.global_variable @Foo : !moore.i9001

// -----

// expected-error @below {{must have a 'moore.yield' terminator}}
moore.global_variable @Foo : !moore.i42 init {
  llvm.unreachable
}

// -----

// UnionCreateOp verifier: input type mismatch
%0 = moore.constant 42 : i16
// expected-error @below {{op input type '!moore.i16' does not match union field 'x' type '!moore.i32'}}
moore.union_create %0 {fieldName = "x"} : !moore.i16 -> union<{x: i32, y: i32}>

// -----

// UnionCreateOp verifier: field not found
%0 = moore.constant 42 : i32
// expected-error @below {{op field 'z' not found in union type}}
moore.union_create %0 {fieldName = "z"} : !moore.i32 -> union<{x: i32, y: i32}>
