// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-full-reset))' --verify-diagnostics --split-input-file %s

//===----------------------------------------------------------------------===//
// Full Reset
//===----------------------------------------------------------------------===//

// -----
// Reset annotation cannot target module
firrtl.circuit "top" {
  // expected-error @+1 {{'FullResetAnnotation' cannot target module; must target port or wire/node instead}}
  firrtl.module @top() attributes {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} {
  }
}

// -----
// Reset annotation resetType must match type of signal
firrtl.circuit "top" {
  firrtl.module @top() {
    // expected-error @below {{'FullResetAnnotation' with resetType == 'async' must target async reset, but targets '!firrtl.uint<1>'}}
    %innerReset = firrtl.wire {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.uint<1>
    // expected-error @below {{'FullResetAnnotation' with resetType == 'sync' must target sync reset, but targets '!firrtl.asyncreset'}}
    %innerReset2 = firrtl.wire {annotations = [{class = "circt.FullResetAnnotation", resetType = "sync"}]} : !firrtl.asyncreset
    // expected-error @below {{'FullResetAnnotation' with resetType == 'sync' must target sync reset, but targets '!firrtl.uint<2>'}}
    %innerReset3 = firrtl.wire {annotations = [{class = "circt.FullResetAnnotation", resetType = "sync"}]} : !firrtl.uint<2>
  }
}


// -----
// Reset annotation cannot target reset signals which are inferred to the wrong type
firrtl.circuit "top" {
  firrtl.module @top() {
   // expected-error @below {{'FullResetAnnotation' with resetType == 'async' must target async reset, but targets '!firrtl.uint<1>'}}
    %innerReset = firrtl.wire {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.uint<1>
    %invalid = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.matchingconnect %innerReset, %invalid : !firrtl.uint<1>

    // expected-error @below {{'FullResetAnnotation' with resetType == 'sync' must target sync reset, but targets '!firrtl.asyncreset'}}
    %innerReset2 = firrtl.wire {annotations = [{class = "circt.FullResetAnnotation", resetType = "sync"}]} : !firrtl.asyncreset
    %asyncWire = firrtl.wire : !firrtl.asyncreset
    firrtl.matchingconnect %innerReset2, %asyncWire : !firrtl.asyncreset
  }
}

// -----
// Ignore reset annotation cannot target port
firrtl.circuit "top" {
  // expected-error @+1 {{ExcludeFromFullResetAnnotation' cannot target port/wire/node; must target module instead}}
  firrtl.module @top(in %reset: !firrtl.asyncreset) attributes {portAnnotations =[[{class = "circt.ExcludeFromFullResetAnnotation"}]]} {
  }
}

// -----
// Ignore reset annotation cannot target wire/node
firrtl.circuit "top" {
  firrtl.module @top() {
    // expected-error @+1 {{ExcludeFromFullResetAnnotation' cannot target port/wire/node; must target module instead}}
    %0 = firrtl.wire {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} : !firrtl.asyncreset
    // expected-error @+1 {{ExcludeFromFullResetAnnotation' cannot target port/wire/node; must target module instead}}
    %1 = firrtl.node %0 {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} : !firrtl.asyncreset
    // expected-error @+1 {{reset annotations must target module, port, or wire/node}}
    %2 = firrtl.asUInt %0 {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : (!firrtl.asyncreset) -> !firrtl.uint<1>
    // expected-error @+1 {{reset annotations must target module, port, or wire/node}}
    %3 = firrtl.asUInt %0 {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} : (!firrtl.asyncreset) -> !firrtl.uint<1>
  }
}

// -----
// Cannot have multiple reset annotations on a module
firrtl.circuit "top" {
  // expected-error @+2 {{multiple reset annotations on module 'top'}}
  // expected-note @+1 {{conflicting "circt.FullResetAnnotation":}}
  firrtl.module @top(in %outerReset: !firrtl.asyncreset) attributes {portAnnotations = [[{class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    // expected-note @+1 {{conflicting "circt.FullResetAnnotation":}}
    %innerReset = firrtl.wire {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    // expected-note @+1 {{conflicting "circt.FullResetAnnotation":}}
    %anotherReset = firrtl.node %innerReset {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
  }
}

// -----
// A module in a domain which already has the reset port should error if the
// type of the port is wrong.
firrtl.circuit "Top" {
  // expected-error @below {{module 'Child' is in reset domain requiring port 'reset' to have type '!firrtl.asyncreset', but has type '!firrtl.uint<1>'}}
  firrtl.module @Child(in %clock: !firrtl.clock, in %reset : !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // expected-note @below {{reset domain rooted here}}
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %child_clock, %child_reset = firrtl.instance child @Child(in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
    firrtl.connect %child_clock, %clock : !firrtl.clock, !firrtl.clock
  }
}

// -----
// Multiple instances of same module cannot live in different reset domains
firrtl.circuit "Top" {
  // expected-error @+1 {{module 'Foo' instantiated in different reset domains}}
  firrtl.module @Foo(in %clock: !firrtl.clock) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // expected-note @+1 {{reset domain 'otherReset' of module 'Child' declared here:}}
  firrtl.module @Child(in %clock: !firrtl.clock, in %otherReset: !firrtl.asyncreset) attributes {portAnnotations = [[],[{class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    // expected-note @+1 {{instance 'child/inst' is in reset domain rooted at 'otherReset' of module 'Child'}}
    %inst_clock = firrtl.instance inst @Foo(in clock: !firrtl.clock)
    firrtl.connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
  }
  firrtl.module @Other(in %clock: !firrtl.clock) attributes {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} {
    %inst_clock = firrtl.instance inst @Foo(in clock: !firrtl.clock)
    firrtl.connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
  }
  // expected-note @+1 {{reset domain 'reset' of module 'Top' declared here:}}
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {portAnnotations = [[],[{class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    %child_clock, %child_otherReset = firrtl.instance child @Child(in clock: !firrtl.clock, in otherReset: !firrtl.asyncreset)
    %other_clock = firrtl.instance other @Other(in clock: !firrtl.clock)
    // expected-note @+1 {{instance 'foo' is in reset domain rooted at 'reset' of module 'Top'}}
    %foo_clock = firrtl.instance foo @Foo(in clock: !firrtl.clock)
    firrtl.connect %child_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %other_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %foo_clock, %clock : !firrtl.clock, !firrtl.clock
  }
}

// -----
// Invalid FullResetAnnotation resetType
firrtl.circuit "Top" {
  // expected-error @+1 {{'FullResetAnnotation' requires resetType == 'sync' | 'async', but got resetType == "potato"}}
  firrtl.module @Top(in %reset: !firrtl.asyncreset) attributes {portAnnotations = [[{class = "circt.FullResetAnnotation", resetType = "potato"}]]} {}
}
