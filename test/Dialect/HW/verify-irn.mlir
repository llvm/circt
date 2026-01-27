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

// -----

// expected-error @below {{'hw.hierpath' op instance path is incorrect. Expected one of "XMRRefA", "XMRRefB" or "XMRRefC". Instead found: "XMRRefD"}}
hw.hierpath private @ref [@XMRRefOp::@foo, @XMRRefD::@a]

hw.module @XMRRefA() {
  %a = sv.wire sym @a : !hw.inout<i2>
}
hw.module @XMRRefB() {
  %a = sv.wire sym @a : !hw.inout<i2>
}
hw.module @XMRRefC() {
  %a = sv.wire sym @a : !hw.inout<i2>
}
hw.module @XMRRefD() {
  %a = sv.wire sym @a : !hw.inout<i2>
}
hw.module @XMRRefOp() {
  hw.instance_choice "foo" sym @foo option "bar" @XMRRefA or @XMRRefB if "B" or @XMRRefC if "C"() -> ()
}

// -----

// expected-error @below {{inner symbol reference #hw.innerNameRef<@VerbatimInnerRef::@nonexistent> could not be found}}
sv.verbatim "// {{0}}" {symbols = [#hw.innerNameRef<@VerbatimInnerRef::@nonexistent>]}
hw.module @VerbatimInnerRef() {
  hw.output
}

// -----

// expected-error @below {{inner symbol reference #hw.innerNameRef<@VerbatimExprInnerRef::@nonexistent> could not be found}}
%0 = sv.verbatim.expr "MACRO" : () -> i32 {symbols = [#hw.innerNameRef<@VerbatimExprInnerRef::@nonexistent>]}
hw.module @VerbatimExprInnerRef() {
  hw.output
}

// -----

// expected-error @below {{inner symbol reference #hw.innerNameRef<@VerbatimExprSEInnerRef::@nonexistent> could not be found}}
%0 = sv.verbatim.expr.se "MACRO" : () -> i32 {symbols = [#hw.innerNameRef<@VerbatimExprSEInnerRef::@nonexistent>]}
hw.module @VerbatimExprSEInnerRef() {
  hw.output
}
