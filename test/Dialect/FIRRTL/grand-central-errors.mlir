// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-grand-central)' -split-input-file -verify-diagnostics %s

firrtl.circuit "NonGroundType" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Foo", elements = [{name = "foo", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}]} {
  firrtl.module @NonGroundType() {
    // expected-error @+2 {{'firrtl.wire' op cannot be added to interface 'Foo', component 'foo' because it is not a ground type.}}
    // expected-note @+1 {{"sifive.enterprise.grandcentral.AugmentedGroundType"}}
    %a = firrtl.wire {annotations = [{a}, {class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Foo", name = "foo", target = []}]} : !firrtl.vector<uint<2>, 1>
  }
}

// -----

// expected-error @+2 {{'firrtl.circuit' op contained an 'AugmentedBundleType' Annotation which did not conform to the expected format}}
// expected-note @+1 {{the problematic 'AugmentedBundleType' is:}}
firrtl.circuit "NonGroundType" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType"}]} {
  firrtl.module @NonGroundType() {}
}

// -----

module  {
  firrtl.circuit "Foo" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "View", elements = [{name = "sub_port", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}]}  {
    firrtl.module @Bar(in %a: !firrtl.uint<1>) {}
    firrtl.module @Foo(in %a: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", id = 0 : i64, type = "parent"}]} {
      // expected-error @+2 {{'firrtl.instance' op is marked as a an interface element}}
      // expected-note @+1 {{{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "View", name = "sub_port", target = [".a"]}}}
      %bar_a = firrtl.instance @Bar  {annotations = [{a}, {class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "View", name = "sub_port", target = [".a"]}], name = "bar"} : !firrtl.flip<uint<1>>
      firrtl.connect %bar_a, %a : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    }
  }
}

// -----

module  {
  firrtl.circuit "Foo" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "View", elements = [{name = "sub_port", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}]}  {
    firrtl.module @Foo(in %a: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", id = 0 : i64, type = "parent"}]} {
      // expected-error @+2 {{'firrtl.mem' op is marked as a an interface element}}
      // expected-note @+1 {{{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "View", name = "some_mem", target = []}}}
      %memory_b_r = firrtl.mem Undefined  {annotations = [{a}, {class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "View", name = "some_mem", target = []}], depth = 16 : i64, name = "memory_b", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>>
    }
  }
}

// -----

module {
  // expected-error @+1 {{'firrtl.circuit' op contained a Grand Central Interface 'Bar' that had an element 'baz' which did not have a scattered companion annotation}}
  firrtl.circuit "Foo" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Bar", elements = [{name = "baz", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}]}  {
    firrtl.module @Foo() {}
  }
}
