// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' -verify-diagnostics --split-input-file %s

firrtl.circuit "UnknownIntrinsic" {
  firrtl.module @UnknownIntrinsic(in %data: !firrtl.uint<32>) {
    %0 = firrtl.wire : !firrtl.uint<32>
    // expected-error @below {{unknown intrinsic}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "unknown_intrinsic" %0 : (!firrtl.uint<32>) -> ()
    firrtl.matchingconnect %0, %data : !firrtl.uint<32>
  }
}

// -----

firrtl.circuit "InvalidCGOperand" {
    firrtl.module @InvalidCGOperand(in %clk: !firrtl.clock, in %en: !firrtl.uint<2>) {
      // expected-error @below {{circt.clock_gate input 1 not size 1}}
      // expected-error @below {{failed to legalize}}
      %0 = firrtl.int.generic "circt.clock_gate" %clk, %en : (!firrtl.clock, !firrtl.uint<2>) -> !firrtl.clock
    }
}

// -----

firrtl.circuit "MissingParam" {
    firrtl.module @MissingParam(in %clk: !firrtl.clock, in %en: !firrtl.uint<2>) {
      // expected-error @below {{circt_plusargs_test is missing parameter FORMAT}}
      // expected-error @below {{failed to legalize}}
      %0 = firrtl.int.generic "circt_plusargs_test" : () -> !firrtl.uint<1>
    }
}

// -----

firrtl.circuit "ViewNotBundle" {
  firrtl.module public @ViewNotBundle() {
    // expected-error @below {{'info' must be augmented bundle}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "circt_view" <name: none = "view", info: none = "{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}"> : () -> ()
  }
}

// -----

firrtl.circuit "ViewNotJSON" {
  firrtl.module public @ViewNotJSON() {
    // expected-error @below {{error parsing view JSON}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "circt_view" <name: none = "view", info: none = "boop"> : () -> ()
  }
}

// -----

firrtl.circuit "ViewNoDefname" {
  firrtl.module public @ViewNoDefname() {
    // expected-error @below {{View 'info' did not contain required key 'defName'}}
    // expected-note @below {{The full 'info' attribute is reproduced here: {class = "sifive.enterprise.grandcentral.AugmentedBundleType"}}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "circt_view" <name: none = "view", info: none = "{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\"}"> : () -> ()
  }
}

// -----

firrtl.circuit "ViewDefnameNotString" {
  firrtl.module public @ViewDefnameNotString() {
    // expected-error @below {{View 'info' did not contain the correct type for key 'defName'}}
    // expected-note @below {{The full 'info' attribute is reproduced here:}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "circt_view" <name: none = "view", info: none = "{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": 5}"> : () -> ()
  }
}

// -----

firrtl.circuit "ViewNoElementsField" {
  firrtl.module public @ViewNoElementsField() {
    // expected-error @below {{View 'info' did not contain required key 'elements'}}
    // expected-note @below {{The full 'info' attribute is reproduced here:}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "circt_view" <name: none = "view", info: none = "{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"MyView\"}"> : () -> ()
  }
}

// -----

firrtl.circuit "ViewInvalidElement" {
  firrtl.module public @ViewInvalidElement() {
    // expected-error @below {{View 'info' attribute with path '.elements[0]' contained an unexpected type (expected a DictionaryAttr)}}
    // expected-note @below {{The received element was: 5 : i64}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "circt_view" <name: none = "view", info: none = "{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"MyView\", \"elements\": [5]}"> : () -> ()
  }
}

// -----

firrtl.circuit "ViewElementEmptyDict" {
  firrtl.module public @ViewElementEmptyDict() {
    // expected-error @below {{View 'info' with path '.elements[0]' did not contain required key 'name'}}
    // expected-note @below {{The full 'info' attribute is reproduced here:}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "circt_view" <name: none = "view", info: none = "{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"MyView\", \"elements\": [{}]}"> : () -> ()
  }
}
