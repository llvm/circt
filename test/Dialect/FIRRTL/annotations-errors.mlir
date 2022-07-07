// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -split-input-file %s -verify-diagnostics

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
  // expected-error @+2 {{targeted by a DontTouchAnotation with target "~DontTouchOnNonReferenceTarget|Submodule"}}
  // expected-error @+1 {{targeted by a DontTouchAnotation with target "~DontTouchOnNonReferenceTarget|DontTouchOnNonReferenceTarget>submodule"}}
  firrtl.module @Submodule() {}
  firrtl.module @DontTouchOnNonReferenceTarget() {
    firrtl.instance submodule @Submodule()
  }
}
