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
  // expected-error @+2 {{No target field in annotation}}
  // expected-error @+1 {{Unable to resolve target of annotation}}
  firrtl.circuit "Foo"  attributes {raw_annotations = [{class = "sifive.enterprise.grandcentral.DataTapsAnnotation"}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{target = ""}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{target}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[1]"}]}  {
    firrtl.module @Bar(in %a: !firrtl.uint<1>) {
    }
    firrtl.module @Foo() {
      %bar_a = firrtl.instance bar  @Bar(in a: !firrtl.uint<1>)
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[a].baz"}, {target = "~Foo|Foo>bar[2].baz", two}]}  {
    firrtl.module @Foo() {
      %bar = firrtl.wire  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[1][0]"}, {target = "~Foo|Foo>bar[1].qnx", two}]}  {
    firrtl.module @Foo(in %clock: !firrtl.clock) {
      %bar = firrtl.reg %clock  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[1].baz[0]"}]}  {
    firrtl.module @Foo(in %clock: !firrtl.clock) {
      %bar = firrtl.reg %clock  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{a, target = "~Foo|Bar"}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{a, target = "~Foo|Foo>x"}]}  {
    firrtl.module @Foo() {
      firrtl.skip
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo"  attributes {raw_annotations = [{a, target = "~Foo|Foo/baz:Bar"}]}  {
    firrtl.module @Bar() {
      firrtl.skip
    }
    firrtl.module @Foo() {
      firrtl.instance bar  @Bar()
    }
  }
}

// -----

module  {
  firrtl.circuit "NLAParse"  attributes {raw_annotations = [{class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", companion = "~NLAParse|A_companion", name = "A", parent = "~NLAParse|DUT", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "B", elements = [{name = "C", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "D", elements = [{name = "clock", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "NLAParse", component = [], module = "NLAParse", path = [{_1 = {value = "dut"}, _2 = {value = "DUT"}}, {_1 = {value = "foobar"}, _2 = {value = "FooBar"}}], ref = "clock"}}}]}}]}}]}  {
    firrtl.module @FooBar() {
    }
    firrtl.module @DUT() {
      firrtl.instance foobar  @FooBar()
    }
    firrtl.module @NLAParse() {
      firrtl.instance dut  @DUT()
    }
  }
}
