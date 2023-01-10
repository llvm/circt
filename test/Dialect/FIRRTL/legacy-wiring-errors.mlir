// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations))' -split-input-file %s -verify-diagnostics

// Every Wiring pin must have exactly one defined source
//
// expected-error @+1 {{Unable to resolve source for pin: "foo_out"}}
firrtl.circuit "FooBar" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "FooBar.Foo.io.out",
      pin = "foo_out"
    }]} {
  firrtl.module@Foo(out %io: !firrtl.bundle<out: uint<1>>) {
      firrtl.skip
  }
  firrtl.module @Bar(out %io: !firrtl.bundle<out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
      %foo_io = firrtl.instance foo interesting_name  @Foo(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %foo_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
  firrtl.module @FooBar(out %io: !firrtl.bundle<in flip: uint<1>, out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<in flip: uint<1>, out: uint<1>>
      %bar_io = firrtl.instance bar interesting_name  @Bar(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %bar_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
}

// -----

// Every Wiring pin must have at least one defined sink
//
// expected-error @+1 {{Unable to resolve sink(s) for pin: "foo_out"}}
firrtl.circuit "FooBar" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "FooBar.FooBar.io.in",
      pin = "foo_out"
    }]} {
  firrtl.module@Foo(out %io: !firrtl.bundle<out: uint<1>>) {
      firrtl.skip
  }
  firrtl.module @Bar(out %io: !firrtl.bundle<out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
      %foo_io = firrtl.instance foo interesting_name  @Foo(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %foo_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
  firrtl.module @FooBar(out %io: !firrtl.bundle<in flip: uint<1>, out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<in flip: uint<1>, out: uint<1>>
      %bar_io = firrtl.instance bar interesting_name  @Bar(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %bar_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
}

// -----

// Multiple SourceAnnotations for the same pin are forbidden
//
// expected-error @+2 {{Unable to apply annotation: {class = "firrtl.passes.wiring.SourceAnnotation", pin = "foo_out", target = "FooBar.FooBar.io.other_in"}}}
// expected-error @+1 {{More than one firrtl.passes.wiring.SourceAnnotation defined for pin "foo_out"}}
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
    },
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "FooBar.FooBar.io.other_in",
      pin = "foo_out"
    }]} {
  firrtl.module@Foo(out %io: !firrtl.bundle<out: uint<1>>) {
      firrtl.skip
  }
  firrtl.module @Bar(out %io: !firrtl.bundle<out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<out: uint<1>>
      %foo_io = firrtl.instance foo interesting_name  @Foo(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %foo_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
  firrtl.module @FooBar(out %io: !firrtl.bundle<in flip: uint<1>, other_in flip: uint<1>, out: uint<1>>) {
      %0 = firrtl.subfield %io[out] : !firrtl.bundle<in flip: uint<1>, other_in flip: uint<1>, out: uint<1>>
      %bar_io = firrtl.instance bar interesting_name  @Bar(out io: !firrtl.bundle<out: uint<1>>)
      %1 = firrtl.subfield %bar_io[out] : !firrtl.bundle<out: uint<1>>
      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
  }
}