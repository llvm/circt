// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -split-input-file %s -verify-diagnostics

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
