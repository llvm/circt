// Check explicit verification pass.
// RUN: circt-opt -hw-verify-irn -verify-diagnostics -split-input-file %s
// Check verification occurs in firtool pipeline.
// RUN: firtool -verify-diagnostics -split-input-file %s

// #3526
hw.module @B() {}

hw.module @A() {
  // expected-note @below {{see existing inner symbol definition here}}
  hw.instance "h" sym @A @B() -> () 
  // expected-error @below {{redefinition of inner symbol named 'A'}}
  hw.instance "h" sym @A @B() -> () 
}

// -----

// expected-error @below {{operation with symbol: #hw.innerNameRef<@A::@invalid> was not found}}
hw.hierpath private @test [@A::@invalid]

hw.module @A() {
}
