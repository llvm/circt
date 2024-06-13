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

// expected-error @below {{constant out of range for result type '!moore.i1'}}
moore.constant 42 : !moore.i1

// -----

// expected-error @below {{constant out of range for result type '!moore.i1'}}
moore.constant -2 : !moore.i1

// -----

// expected-error @below {{attribute width 9 does not match return type's width 8}}
"moore.constant" () {value = 42 : i9} : () -> !moore.i8

// -----

%0 = moore.constant 0 : i8
// expected-error @below {{'moore.yield' op expects parent op 'moore.conditional'}}
moore.yield %0 : i8

// -----

%0 = moore.constant true : i1
%1 = moore.constant 42 : i8
%2 = moore.constant 42 : i32

moore.conditional %0 : i1 -> i32 {
  // expected-error @below {{yield type must match conditional. Expected '!moore.i32', but got '!moore.i8'}}
  moore.yield %1 : i8
} {
  moore.yield %2 : i32
}

// -----

%0 = moore.constant true : i1
%1 = moore.constant 42 : i32
%2 = moore.constant 42 : i8

moore.conditional %0 : i1 -> i32 {
  moore.yield %1 : i32
} {
  // expected-error @below {{yield type must match conditional. Expected '!moore.i32', but got '!moore.i8'}}
  moore.yield %2 : i8
}
