// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations{disable-annotation-classless=true})' -verify-diagnostics --split-input-file %s

module  {
  // expected-error @+2 {{circuit name doesn't match annotation}}
  // expected-error @+1 {{Unable to resolve target of annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Fooo|Foo>bar"}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  // expected-error @+2 {{circuit name doesn't match annotation}}
  // expected-error @+1 {{Unable to resolve target of annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Fooo"}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  // expected-error @+3 {{Annotation 'sifive.enterprise.grandcentral.DataTapsAnnotation' did not contain required key 'blackBox'.}}
  // expected-note @+2 {{The full Annotation is reproduced here}}
  // expected-error @+1 {{Unable to apply annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{class = "sifive.enterprise.grandcentral.DataTapsAnnotation"}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
// expected-error @+2 {{Target field in annotation is empty}}
// expected-error @+1 {{Unable to resolve target of annotation: circt.missing}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{target = ""}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  // expected-error @+2 {{Target field in annotation doesn't contain string}}
  // expected-error @+1 {{Unable to resolve target of annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{target}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  // expected-error @+1 {{Unable to resolve target of annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[1]"}]}  {
    firrtl.module @Bar(in %a: !firrtl.uint<1>) {
    }
    firrtl.module @Foo() {
  // expected-error @+1 {{index access '1' into non-vector type}}
      %bar_a = firrtl.instance bar  @Bar(in a: !firrtl.uint<1>)
    }
  }
}

// -----

module  {
  // expected-error @+1 {{Unable to resolve target of annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[a].baz"}, {target = "~Foo|Foo>bar[2].baz", two}]}  {
    firrtl.module @Foo() {
  // expected-error @+1 {{Cannot convert 'a' to an integer}}
      %bar = firrtl.wire  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    }
  }
}

// -----

module  {
  // expected-error @+1 {{Unable to resolve target of annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[1][0]"}, {target = "~Foo|Foo>bar[1].qnx", two}]}  {
    firrtl.module @Foo(in %clock: !firrtl.clock) {
  // expected-error @+2 {{index access '0' into non-vector type}}
  // expected-error @+1 {{cannot resolve field 'qnx' in subtype}}
      %bar = firrtl.reg %clock  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    }
  }
}

// -----

module  {
  // expected-error @+1 {{Unable to resolve target of annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[1].baz[0]"}]}  {
    firrtl.module @Foo(in %clock: !firrtl.clock) {
  // expected-error @+1 {{index access '0' into non-vector type}}
      %bar = firrtl.reg %clock  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    }
  }
}

// -----

module  {
  // expected-error @+2 {{Unable to resolve target of annotation}}
  // expected-error @+1 {{module doesn't exist 'Bar'}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{a, target = "~Foo|Bar"}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  // expected-error @+2 {{Unable to resolve target of annotation}}
  // expected-error @+1 {{cannot find instance 'baz' in 'Foo'}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{a, target = "~Foo|Foo/baz:Bar"}]}  {
    firrtl.module @Bar() {
      firrtl.skip
    }
    firrtl.module @Foo() {
      firrtl.instance bar  @Bar()
    }
  }
}
