// RUN: circt-opt %s --verify-diagnostics --split-input-file

func.func @Foo() {
  return
}

moore.module @Bar {
  // expected-error @below {{symbol 'Foo' must reference a 'moore.module', but got a 'func.func' instead}}
  moore.instance "foo" @Foo
}
