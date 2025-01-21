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

// -----
// View has "id" field on ground type. 

// In the future, verifier on firrtl.view can catch this.
firrtl.circuit "HasID" {
  firrtl.module @HasID() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // expected-error @below {{op has 'id' field which is only for old annotation encoding}}
    // expected-note @below {{id within GroundType attribute: {class = "sifive.enterprise.grandcentral.AugmentedGroundType", description = "description of foo", id = 1 : i64, name = "foo"}}}
    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          id = 1,
          name = "foo"
        }
      ]
    }>, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----
// Attribute missing "class" field.

// In the future, verifier on firrtl.view can catch this.
firrtl.circuit "MissingClass" {
  firrtl.module @MissingClass() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // expected-error @below {{missing 'class' key in {description = "description of foo", name = "foo"}}}
    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          // (Missing class key)
          description = "description of foo",
          name = "foo"
        }
      ]
    }>, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----
// Top-level attribute missing "defName" field.

// In the future, verifier on firrtl.view can catch this.
firrtl.circuit "MissingDefName" {
  firrtl.module @MissingDefName() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // expected-error @below {{op missing 'defName' at top-level}}
    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          name = "foo"
        }
      ]
    }>, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----
// Top-level attribute missing "elements" field.

// In the future, verifier on firrtl.view can catch this.
firrtl.circuit "MissingElements" {
  firrtl.module @MissingElements() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // expected-error @below {{op missing 'elements' at top-level}}
    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView"
    }>, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----
// Invalid / unsupported "class" in element.

// In the future, verifier on firrtl.view can catch this.
firrtl.circuit "InvalidClass" {
  firrtl.module @InvalidClass() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // expected-error @below {{op has an invalid AugmentedType class 'invalid class'}}
    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "invalid class",
          name = "foo"
        }
      ]
    }>, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----
// Invalid augmented "class" in element.

// In the future, verifier on firrtl.view can catch this.
firrtl.circuit "InvalidAug" {
  firrtl.module @InvalidAug() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

    // expected-error @below {{op has an invalid AugmentedType 'GroundTorp'}}
    firrtl.view "GroundView", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundTorp",
          name = "foo"
        }
      ]
    }>, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----
// Bundle missing 'defName' or `element

firrtl.circuit "InvalidBundleDefNameElements" {
  firrtl.module @InvalidBundleDefNameElements() {
    // expected-error @below {{has an invalid AugmentedBundleType that does not contain 'defName' and 'elements' fields: {class = "sifive.enterprise.grandcentral.AugmentedBundleType", description = "description of foo"}}
    firrtl.view "View", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          description = "description of foo",
          name = "foo"
        }
      ]
    }>
  }
}

// -----
// Bundle missing 'name'.

firrtl.circuit "InvalidBundleNoName" {
  firrtl.module @InvalidBundleNoName() {
    // expected-error @below {{missing 'name' field in element of bundle: {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "foo", description = "description of foo", elements = []}}
    firrtl.view "View", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          description = "description of foo",
          defName = "foo",
          elements = []
        }
      ]
    }>
  }
}

// -----
// Vector missing 'name'.

firrtl.circuit "InvalidVectorNoName" {
  firrtl.module @InvalidVectorNoName() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // expected-error @below {{has an invalid AugmentedVectorType that does not contain 'name' and 'elements' fields: {class = "sifive.enterprise.grandcentral.AugmentedVectorType", description = "description of foo", elements = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", description = "description of bar", name = "UNUSED"}]}}
    firrtl.view "View", <{
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          description = "description of foo",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              description = "description of bar",
              name = "UNUSED"
            }
          ]
        }
      ]
    }>, %c0_ui1 : !firrtl.uint<1>
  }
}
