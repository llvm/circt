// RUN: circt-opt %s --verify-diagnostics --split-input-file

func.func @Foo() {
  return
}

moore.module @Bar {
  // expected-error @below {{symbol 'Foo' must reference a 'moore.module', but got a 'func.func' instead}}
  moore.instance "foo" @Foo
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
