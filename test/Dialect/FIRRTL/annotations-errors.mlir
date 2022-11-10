// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations))' -split-input-file %s -verify-diagnostics

// An unknown annotation should error.
//
// expected-error @+1 {{Unhandled annotation}}
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.unknown"
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// An incorrect circuit target should report an error.
//
// expected-error @+2 {{circuit name doesn't match annotation}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Fooo|Foo>bar"
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// An incorrect circuit name should report an error.
//
// expected-error @+2 {{circuit name doesn't match annotation}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Fooo"
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// An empty target string should be illegal.
//
// expected-error @+2 {{Cannot tokenize annotation path}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = ""
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// A target that does a subindex of an instance should be illegal.
//
// expected-error @+2 {{illegal target '~Foo|Foo>bar[0]' indexes into an instance}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Foo|Foo>bar[0]"
  }
]} {
  firrtl.module @Bar() {}
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
  }
}

// -----

// A target that uses a string for an index should be illegal.
//
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Foo|Foo>bar[a].baz"
  }
]} {
  firrtl.module @Foo() {
    // expected-error @+1 {{Cannot convert 'a' to an integer}}
    %bar = firrtl.wire : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// -----

// Invalid subindex or subfield targets are checked.
//
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Foo|Foo>bar[1][42]"
  },
  {
    class = "circt.test",
    target = "~Foo|Foo>bar[1].qnx"
  },
  {
    class = "circt.test",
    target = "~Foo|Foo>bar[1].baz[1337]"
  }
]} {
  firrtl.module @Foo(in %clock: !firrtl.clock) {
    // expected-error @+3 {{index access '42' into non-vector type}}
    // expected-error @+2 {{cannot resolve field 'qnx' in subtype}}
    // expected-error @+1 {{index access '1337' into non-vector type}}
    %bar = firrtl.reg %clock : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// -----

// A target on a non-existent module should error.
//
// expected-error @+2 {{module doesn't exist 'Bar'}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Foo|Bar"
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// A target on a non-existent component should error.
//
// expected-error @+2 {{cannot find name 'x' in Foo}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Foo|Foo>x"
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// A non-local annotation on a non-existent instance should error.
//
// expected-error @+2 {{cannot find instance 'baz' in 'Foo'}}
// expected-error @+1 {{Unable to resolve target of annotation}}
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {
    class = "circt.test",
    target = "~Foo|Foo/baz:Bar"
  }
]} {
  firrtl.module private @Bar() {}
  firrtl.module @Foo() {
    firrtl.instance bar interesting_name  @Bar()
  }
}

// -----

// expected-error @+1 {{Unable to apply annotation}}
firrtl.circuit "LocalOnlyAnnotation" attributes {
  rawAnnotations = [
    {class = "circt.testLocalOnly",
     target = "~LocalOnlyAnnotation|LocalOnlyAnnotation/foo:Foo>w"}
  ]} {
  firrtl.module @Foo() {
    // expected-error @+2 {{targeted by a non-local annotation}}
    // expected-note @+1 {{see current annotation}}
    %w = firrtl.wire : !firrtl.uint<1>
  }
  firrtl.module @LocalOnlyAnnotation() {
    firrtl.instance foo @Foo()
  }
}

// -----

// expected-error @+1 {{Unable to apply annotation}}
firrtl.circuit "DontTouchOnNonReferenceTarget" attributes {
  rawAnnotations = [
    {class = "firrtl.transforms.DontTouchAnnotation",
     target = "~DontTouchOnNonReferenceTarget|Submodule"},
    {class = "firrtl.transforms.DontTouchAnnotation",
     target = "~DontTouchOnNonReferenceTarget|DontTouchOnNonReferenceTarget>submodule"}]} {
  firrtl.module @Submodule() {}
  firrtl.module @DontTouchOnNonReferenceTarget() {
    firrtl.instance submodule @Submodule()
  }
}
