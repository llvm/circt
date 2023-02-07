// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations))' --split-input-file %s | FileCheck %s

// Check added ports are real type
// CHECK-LABEL: firrtl.circuit "FooBar"
firrtl.circuit "FooBar" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Foo.io.out",
      pin = "foo_out"
    },
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "FooBar.FooBar.io.in",
      pin = "foo_out"
    }]} {
  // CHECK: firrtl.module @Foo
  // The real port type of the source should be bored
  // CHECK-SAME: in %io_out__bore: !firrtl.uint<1>
  firrtl.module@Foo(out %io: !firrtl.bundle<out: uint<1>>) {
      firrtl.skip
  }
  // CHECK: firrtl.module @Bar
  // The real port type of the source should be bored in the parent
  // CHECK-SAME: in %foo_io_out__bore: !firrtl.uint<1>
  firrtl.module @Bar(out %io: !firrtl.bundle<out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
      // CHECK: firrtl.instance foo
      // CHECK-SAME: in io_out__bore: !firrtl.uint<1>
      %foo_io = firrtl.instance foo interesting_name  @Foo(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %foo_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @FooBar
  firrtl.module @FooBar(out %io: !firrtl.bundle<in flip: uint<1>, out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<in flip: uint<1>, out: uint<1>>
      // CHECK: firrtl.instance bar
      // CHECK-SAME: in foo_io_out__bore: !firrtl.uint<1>
      %bar_io = firrtl.instance bar interesting_name  @Bar(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %bar_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
}

// -----

// Test the behaviour of single source, multiple sink
// CHECK-LABEL: firrtl.circuit "FooBar"
firrtl.circuit "FooBar" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "FooBar.FooBar.io.in",
      pin = "in"
    },
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Foo.io.out",
      pin = "in"
    },
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Foo_1.io.out",
      pin = "in"
    },
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Bar.io.out",
      pin = "in"
    }]} {
  // CHECK: firrtl.module @Foo
  // CHECK-SAME: in %io_out__bore: !firrtl.uint<1>
  firrtl.module @Foo(out %io: !firrtl.bundle<out: uint<1>>) {
    firrtl.skip
    // CHECK: %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.connect %0, %io_out__bore : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Foo_1
  // CHECK-SAME: in %io_out__bore: !firrtl.uint<1>
  firrtl.module @Foo_1(out %io: !firrtl.bundle<out: uint<1>>) {
    firrtl.skip
    // CHECK: %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.connect %0, %io_out__bore : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Bar
  // CHECK-SAME: in %io_out__bore: !firrtl.uint<1>
  firrtl.module @Bar(out %io: !firrtl.bundle<out: uint<1>>) {
    firrtl.skip
    // CHECK: %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.connect %0, %io_out__bore : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: firrtl.module @FooBar
  firrtl.module @FooBar(out %io: !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>) {
    // CHECK: %0 = firrtl.subfield %io[in] : !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>
    %0 = firrtl.subfield %io[out_bar] : !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>
    %1 = firrtl.subfield %io[out_foo1] : !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>
    %2 = firrtl.subfield %io[out_foo0] : !firrtl.bundle<in flip: uint<1>, out_foo0: uint<1>, out_foo1: uint<1>, out_bar: uint<1>>
    // CHECK: firrtl.instance foo0
    // CHECK-SAME: in io_out__bore: !firrtl.uint<1>
    %foo0_io = firrtl.instance foo0 interesting_name  @Foo(out io: !firrtl.bundle<out: uint<1>>)
    %3 = firrtl.subfield %foo0_io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.instance foo1
    // CHECK-SAME: in io_out__bore: !firrtl.uint<1>
    %foo1_io = firrtl.instance foo1 interesting_name  @Foo_1(out io: !firrtl.bundle<out: uint<1>>)
    %4 = firrtl.subfield %foo1_io[out] : !firrtl.bundle<out: uint<1>>
    // CHECK: firrtl.instance bar
    // CHECK-SAME: in io_out__bore: !firrtl.uint<1>
    %bar_io = firrtl.instance bar interesting_name  @Bar(out io: !firrtl.bundle<out: uint<1>>)
    %5 = firrtl.subfield %bar_io[out] : !firrtl.bundle<out: uint<1>>
    firrtl.strictconnect %2, %3 : !firrtl.uint<1>
    firrtl.strictconnect %1, %4 : !firrtl.uint<1>
    firrtl.strictconnect %0, %5 : !firrtl.uint<1>
    // CHECK: firrtl.connect %foo0_io_out__bore, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %foo1_io_out__bore, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.connect %bar_io_out__bore, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
