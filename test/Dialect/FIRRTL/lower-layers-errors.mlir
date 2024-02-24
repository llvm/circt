// RUN: circt-opt -firrtl-lower-layers -verify-diagnostics %s

firrtl.circuit "DuplicateMarkDUTAnnotation" {
  // expected-note @below {{the first DUT was found here}}
  firrtl.module @Foo() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {}
  // expected-error @below {{is marked with a 'sifive.enterprise.firrtl.MarkDUTAnnotation', but 'Foo' also had such an annotation}}
  firrtl.module @DuplicateMarkDUTAnnotation() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance foo @Foo()
  }
}
