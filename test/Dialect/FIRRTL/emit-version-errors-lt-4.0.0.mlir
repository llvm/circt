// Tests for --firrtl-version: ops requiring >= 4.0.0 error when targeting 3.3.0.
// RUN: circt-translate --export-firrtl --firrtl-version=3.3.0 \
// RUN:   --verify-diagnostics --split-input-file %s

// enabled layers on a module require >= 4.0.0
firrtl.circuit "EnabledLayers" {
  firrtl.layer @A bind {}
  // expected-error @below {{'firrtl.module' op enabled layers requires FIRRTL 4.0.0}}
  firrtl.module @EnabledLayers() attributes {layers = [@A]} {}
}

// -----

// generic intrinsic expression requires >= 4.0.0.  The intrinsic is consumed
// by a firrtl.node so it is emitted inline; the error fires on the intrinsic op.
firrtl.circuit "GenericIntrinsic" {
  firrtl.module @GenericIntrinsic(in %clk : !firrtl.clock) {
    // expected-error @below {{'firrtl.int.generic' op generic intrinsics requires FIRRTL 4.0.0}}
    %0 = firrtl.int.generic "circt_sizeof" %clk : (!firrtl.clock) -> !firrtl.uint<32>
    %n = firrtl.node %0 : !firrtl.uint<32>
  }
}

// -----

// formal test declaration requires >= 4.0.0.  The circuit must contain a
// module matching its name (@FormalTest) for FIRRTL verification; the
// referenced target module (@FormalTop) is placed after the formal op so the
// emitter errors on formal first.
firrtl.circuit "FormalTest" {
  // expected-error @below {{'firrtl.formal' op formal tests requires FIRRTL 4.0.0}}
  firrtl.formal @myFormalDecl, @FormalTop {}
  firrtl.module @FormalTest() {}
  firrtl.module @FormalTop() {}
}

// -----

// List<T> property type requires >= 4.0.0.  list.create is consumed by
// propassign (valid at 3.3.0 >= 3.1.0); the error fires on the list.create op.
firrtl.circuit "ListCreate" {
  firrtl.module @ListCreate(out %l : !firrtl.list<integer>) {
    %0 = firrtl.integer 42
    // expected-error @below {{'firrtl.list.create' op Lists requires FIRRTL 4.0.0}}
    %l0 = firrtl.list.create %0 : !firrtl.list<integer>
    firrtl.propassign %l, %l0 : !firrtl.list<integer>
  }
}
