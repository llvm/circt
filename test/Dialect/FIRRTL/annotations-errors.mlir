// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -verify-diagnostics --split-input-file %s


// expected-error @+2 {{circuit name 'Foo' doesn't match annotation 'Fooo'}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{one, target = "~Fooo|Foo>bar", class="circt.test"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// expected-error @+2 {{circuit name 'Foo' doesn't match annotation 'Fooo'}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{one, target = "~Fooo", class="circt.test"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// expected-error @+2 {{cannot find instance 'bar' in '"Foo"'}}
// expected-error @+1 {{Unable to resolve target of annotation:}}
firrtl.circuit "Foo"  attributes {annotations = [{a = "a", target = "~Foo|Foo/bar:Bar/baz:Baz", class="circt.test"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// An empty target should report an error (and not crash)

// expected-error @+2 {{empty target string}}
// expected-error @+1 {{Unable to resolve target of annotation:}}
firrtl.circuit "Foo"  attributes {annotations = [{target = "", class="circt.test"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// A null target should report an error (and not crash)

// expected-error @+2 {{Target field in annotation doesn't contain string}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{target, class="circt.test"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// Invalid port reference should report errors

// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{one, target = "~Foo|Foo>bar[1]", class="circt.test"}]}  {
  firrtl.module @Bar(in %a: !firrtl.uint<1>) {
  }
  firrtl.module @Foo() {
// expected-error @+1 {{index access '1' into non-vector type}}
    %bar_a = firrtl.instance @Bar  {name = "bar"} : !firrtl.uint<1>
  }
}

// -----

// Invalid sub-target annotation should report an error

// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{one, target = "~Foo|Foo>bar[a].baz", class="circt.test"}, {target = "~Foo|Foo>bar[2].baz", two, class="circt.test"}]}  {
  firrtl.module @Foo() {
// expected-error @+1 {{non-integer array index}}
    %bar = firrtl.wire  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// -----

// Invalid sub-target annotation should report an error

// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{one, target = "~Foo|Foo>bar[1][0]", class="circt.test"}, {target = "~Foo|Foo>bar[1].qnx", two, class="circt.test"}]}  {
  firrtl.module @Foo(in %clock: !firrtl.clock) {
// expected-error @+2 {{cannot resolve field 'qnx' in subtype ''!firrtl.bundle<baz: uint<1>, qux: uint<1>>''}}
// expected-error @+1 {{index access '0' into non-vector type}}
    %bar = firrtl.reg %clock  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// -----

// Invalid sub-target annotation should report an error

// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{one, target = "~Foo|Foo>bar[1].baz[0]", class="circt.test"}]}  {
  firrtl.module @Foo(in %clock: !firrtl.clock) {
// expected-error @+1 {{index access '0' into non-vector type}}
    %bar = firrtl.reg %clock  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// -----

// A target pointing at a non-existent module should error.

// expected-error @+2 {{cannot find module 'Bar' in annotation}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{a, target = "~Foo|Bar", class="circt.test"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// A target pointing at a non-existent component should error.

// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{a, target = "~Foo|Foo>x", class="circt.test"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// A target pointing at a non-existent instance should error.

// expected-error @+2 {{cannot find instance 'baz' in '"Foo"'}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {annotations = [{a, target = "~Foo|Foo/baz:Bar", class="circt.test"}]}  {
  firrtl.module @Bar() {
    firrtl.skip
  }
  firrtl.module @Foo() {
    firrtl.instance @Bar  {name = "bar"}
  }
}
