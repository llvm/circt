// RUN:  circt-opt --hw-inline-input-only-modules --split-input-file --verify-diagnostics %s | FileCheck %s

// Check a module with no ports is empty.
module {
  // CHECK-NOT:   @InputOnly
  hw.module private @InputOnly() {}

  hw.module @Top() {
    // CHECK-NOT: @InputOnly
    hw.instance "inst" @InputOnly() -> ()
  }
}

// -----

// Check an "empty" module is inlined.
module {
  // CHECK-NOT:   @InputOnly
  hw.module private @InputOnly(in %clock: i1, in %cond: i1) {}

  hw.module @Top(in %clock: i1, in %a: i1, in %b: i1) {
    // CHECK-NOT: @InputOnly
    hw.instance "inst" @InputOnly(clock: %clock: i1, cond: %b: i1) -> ()
  }
}

// -----

// Check a module with IR in its body is inlined.
module {
  // CHECK-NOT:   @InputOnly
  hw.module private @InputOnly(in %x: i1) {
    %0 = comb.and %x : i1
  }

  hw.module @Top(in %clock: i1, in %x: i1) {
    // CHECK-NOT: @InputOnly
    // CHECK:     comb.and %x
    hw.instance "inst" @InputOnly(x: %x: i1) -> ()
  }
}

// -----

// Check that a module with IR which is "out of order" is inlined correctly.
module {
  // CHECK-NOT:   @InputOnly
  hw.module private @InputOnly(in %x: i1) {
    %0 = comb.and %1 : i1
    %1 = comb.and %x : i1
  }

  hw.module @Top(in %clock: i1, in %x: i1) {
    // CHECK-NOT: @InputOnly
    // CHECK:     %0 = comb.and %1 : i1
    // CHECK:     %1 = comb.and %x : i1
    hw.instance "inst" @InputOnly(x: %x: i1) -> ()
  }
}

// -----

// Check that if a module has bound-in instances, it cannot be inlined.
module {
  // This module cannot be inlined because it is bound in.
  // CHECK: @Bound()
  hw.module private @Bound() {}

  // This module must be inlined.
  // CHECK-NOT: @Component
  hw.module private @Component() {
    hw.instance "bound" sym @bound @Bound() -> () {doNotPrint}
  }

  // CHECK: @Top
  hw.module @Top() {
    // CHECK: @Bound
    hw.instance "component" @Component() -> ()
  }

  // CHECK: sv.bind <@Top::@bound>
  sv.bind <@Component::@bound>
}

// -----

// Check that bind statements are cloned appropriately.

// Check that if a module has bound-in instances, it cannot be inlined.
module {
  // CHECK: @Bound()
  hw.module private @Bound() {}

  // This module must be inlined.
  // CHECK-NOT: @Component
  hw.module private @Component() {
    hw.instance "bound" sym @bound @Bound() -> () {doNotPrint}
  }

  // CHECK: @Top
  hw.module @Top() {
    // CHECK: @Bound
    hw.instance "component1" @Component() -> ()
    // CHECK: @Bound
    hw.instance "component2" @Component() -> ()
  }

  // This bind statement is cloned out when component1 and 2 are inlined into top.
  // CHECK: sv.bind <@Top::@bound>
  // CHECK: sv.bind <@Top::@bound_0>
  sv.bind <@Component::@bound>
}

// -----

// Check that a module cannot be inlined if an instance has an inner sym, and
// that inner sym is not used by a non-bind operation.
module {
  // CHECK: @Component
  // expected-warning @below {{Component cannot be inlined because there is an instance with a symbol}}
  hw.module private @Component() {}

  // CHECK: @Top
  hw.module @Top() {
    // CHECK: "foo" sym @foo @Component
    // expected-note @below {{}}
    hw.instance "foo" sym @foo @Component() -> ()
  }
}

// -----

// @ShouldNotBeInlined cannot be inlined because there is a wire with an inner
// sym, which is referred by hierpath op.
module {
  hw.hierpath private @Foo [@ShouldNotBeInlined::@foo]

  hw.module private @ShouldNotBeInlined(in %clock: i1, in %a: i1) {
    // expected-warning @below {{module ShouldNotBeInlined is an input only module but cannot be inlined because signals are referred by name}}
    %w = sv.wire sym @foo: !hw.inout<i1>
    sv.always posedge %clock {
      sv.if %a {
        sv.assert %a, immediate message "foo"
      }
    }
    hw.output
  }

  hw.module @Top(in %clock: i1, in %a: i1) {
    hw.instance "foo" @ShouldNotBeInlined(clock: %clock: i1, a: %a: i1) -> ()
  }
}
