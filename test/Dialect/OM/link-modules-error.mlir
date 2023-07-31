// RUN: circt-opt %s --verify-diagnostics -pass-pipeline='builtin.module(om-link-modules)' --split-input-file

module {
  // Raise an error if there is no definition.
  module {
    // expected-error @+1 {{OM linker error: class "A" is declared as an external class but there is no definition}}
    om.class.extern @A() {}
  }
  module {
    // expected-note @+1 {{class "A" is declared here as well}}
    om.class.extern @A() {}
  }
}

// -----

module {
  // Raise an error if there are multiple definitions.
  module {
    // expected-error @+1 {{OM linker error: class "A" is declared as an external class but there are multiple definitions}}
    om.class.extern @A() {}
  }
  module {
    // expected-note @+1 {{class "A" is declared here as well}}
    om.class.extern @A() {}
  }
  module {
    // expected-note @+1 {{class "A" is defined here}}
    om.class @A() {}
  }
  module {
    // expected-note @+1 {{class "A" is defined here}}
    om.class @A() {}
  }
}

// -----

module {
  // Check that there is an verification error if types mismatch after linking.
  module {
    om.class.extern @A(%arg: i1) {
      om.class.extern.field @a: i1
    }
    om.class @UseA(%arg: i1) {
      // expected-error @+1 {{'om.object' op actual parameter type ('i1') doesn't match formal parameter type ('i2')}}
      %0 = om.object @A(%arg) : (i1) -> !om.class.type<@A>
    }
  }
  module {
    om.class @A(%arg: i2) {
      om.class.field @a, %arg: i2
    }
  }
}
