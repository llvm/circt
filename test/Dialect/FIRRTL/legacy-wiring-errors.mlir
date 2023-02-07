// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-annotations))' -split-input-file %s -verify-diagnostics

// Every Wiring pin must have exactly one defined source
//
// expected-error @+1 {{Unable to resolve source for pin: "foo_out"}}
firrtl.circuit "Foo" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "Foo.Foo.out",
      pin = "foo_out"
    }]} {
  firrtl.module @Foo(out %out: !firrtl.uint<1>) {
      firrtl.skip
  }
}

// -----

// Every Wiring pin must have at least one defined sink
//
// expected-error @+1 {{Unable to resolve sink(s) for pin: "foo_in"}}
firrtl.circuit "Foo" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "Foo.Foo.in",
      pin = "foo_in"
    }]} {
  firrtl.module @Foo(in %in: !firrtl.uint<1>) {
      firrtl.skip
  }
}

// -----

// Multiple SourceAnnotations for the same pin are forbidden
//
// expected-error @+2 {{Unable to apply annotation: {class = "firrtl.passes.wiring.SourceAnnotation", pin = "foo_out", target = "Foo.Foo.b"}}}
// expected-error @+1 {{More than one firrtl.passes.wiring.SourceAnnotation defined for pin "foo_out"}}
firrtl.circuit "Foo" attributes {
  rawAnnotations = [
    {
      class = "firrtl.passes.wiring.SinkAnnotation",
      target = "Foo.Foo.out",
      pin = "foo_out"
    },
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "Foo.Foo.a",
      pin = "foo_out"
    },
    {
      class = "firrtl.passes.wiring.SourceAnnotation",
      target = "Foo.Foo.b",
      pin = "foo_out"
    }]} {
  firrtl.module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      firrtl.skip
  }
}
