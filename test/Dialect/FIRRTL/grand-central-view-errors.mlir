// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-grand-central))' -split-input-file -verify-diagnostics %s


// View has insufficient operands.

// In the future, verifier on firrtl.view can catch this.
firrtl.circuit "TooFewOperands" {
  firrtl.module @TooFewOperands() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // expected-error @below {{op more ground types needed (2 so far) than view has operands (1)}}
    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          name = "foo"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "multi\nline\ndescription\nof\nbar",
          name = "bar"
        }
      ]
    }>, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----
// View has too many operands.

// In the future, verifier on firrtl.view can catch this.
firrtl.circuit "TooManyOperands" {
  firrtl.module @TooManyOperands() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // expected-error @below {{op has too many operands: 3 operands but only 2 were needed}}
    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          name = "foo"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "multi\nline\ndescription\nof\nbar",
          name = "bar"
        }
      ]
    }>, %c0_ui1, %c0_ui1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  }
}
