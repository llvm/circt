// UNSUPPORTED: system-windows
// RUN: circt-reduce %s --include=module-internal-name-sanitizer --include=module-name-sanitizer --test /usr/bin/env --test-arg true --keep-best=0 | FileCheck %s

// Test that module-name-sanitizer correctly renames modules, updates all
// SymbolRefAttr users (including sv.verbatim $symbols), and updates
// hw.hierpath namepath entries (which use InnerRefAttr, not SymbolRefAttr).
// Also tests that modules already carrying a metasyntactic name prefix are
// left unchanged (idempotency).

// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "A" {
  // hw.hierpath namepath entries use InnerRefAttr, not SymbolRefAttr, so they
  // require the NLATable rename step rather than SymbolTable::rename.
  // CHECK: hw.hierpath private @nla [@Foo::@sym_b, @Bar]
  hw.hierpath private @nla [@A::@sym_b, @B]
  // sv.verbatim $symbols entries use FlatSymbolRefAttr, so they are updated
  // by SymbolTable::rename.
  // CHECK: sv.verbatim "// ref: {{[{][{]0[}][}]}}" {symbols = [@Bar]}
  sv.verbatim "// ref: {{0}}" {symbols = [@B]}
  // CHECK-NEXT: firrtl.module private @Bar
  // CHECK-SAME:   in %clk: !firrtl.clock
  // CHECK-SAME:   in %clk_0: !firrtl.clock
  // CHECK-SAME:   in %rst: !firrtl.reset
  // CHECK-SAME:   in %rst_0: !firrtl.reset
  // CHECK-SAME:   out %ref: !firrtl.probe<uint<1>>
  // CHECK-SAME:   out %ref_0: !firrtl.rwprobe<uint<1>>
  // CHECK-SAME:   in %a: !firrtl.uint<1>
  // CHECK-SAME:   out %b: !firrtl.uint<1>
  firrtl.module private @B(
    in %clock: !firrtl.clock,
    in %clock2: !firrtl.clock,
    in %reset: !firrtl.reset,
    in %reset2: !firrtl.reset,
    out %someProbe: !firrtl.probe<uint<1>>,
    out %someOtherProbe: !firrtl.rwprobe<uint<1>>,
    in %x: !firrtl.uint<1>,
    out %y: !firrtl.uint<1>
  ) {
    // CHECK-NEXT: %reg = firrtl.reg
    // CHECK:      firrtl.regreset
    // CHECK-SAME:   {name = "reg"}
    %derp = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.const.uint<1>
    %herp = firrtl.regreset %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.reset, !firrtl.const.uint<1>, !firrtl.uint<1>
  }
  // CHECK:      firrtl.module @Foo
  // CHECK-SAME:   in %clk: !firrtl.clock
  // CHECK-SAME:   in %a: !firrtl.uint<1>
  // CHECK-SAME:   in %rst: !firrtl.asyncreset
  // CHECK-SAME:   out %b: !firrtl.uint<1>
  // CHECK-SAME:   out %ref: !firrtl.probe<uint<1>>
  // CHECK-SAME:   out %ref_0: !firrtl.rwprobe<uint<1>>
  firrtl.module @A(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.uint<1>,
    in %reset2: !firrtl.asyncreset,
    out %out: !firrtl.uint<1>,
    out %aProbe: !firrtl.probe<uint<1>>,
    out %bProbe: !firrtl.rwprobe<uint<1>>
  ) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK-NEXT: %wire = firrtl.wire
    // CHECK-NEXT: firrtl.wire {name = "wire"}
    %foo = firrtl.wire : !firrtl.uint<1>
    %bar = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %node = firrtl.node
    // CHECK-NEXT: firrtl.node {{.*}} {name = "node"}
    %baz = firrtl.node %bar : !firrtl.uint<1>
    %qux = firrtl.node %baz : !firrtl.uint<1>
    %b_clock, %b_clock2, %b_reset, %b_reset2,  %b_someProbe, %b_someOtherProbe,
      %b_x, %b_y = firrtl.instance b sym @sym_b @B(
        in clock: !firrtl.clock,
        in clock2: !firrtl.clock,
        in reset: !firrtl.reset,
        in reset2: !firrtl.reset,
        out someProbe: !firrtl.probe<uint<1>>,
        out someOtherProbe: !firrtl.rwprobe<uint<1>>,
        in x: !firrtl.uint<1>,
        out y: !firrtl.uint<1>
      )
    // CHECK: %Baz = firrtl.object @Baz()
    %obj = firrtl.object @MyClass()
  }
  // CHECK: firrtl.class @Baz() {
  firrtl.class @MyClass() {
  }
}

// Test reduction idempotency.  Modules that have already been renamed, should
// not be renamed again.  The reduction is one-shot, so it should not run more
// than once.  However, if a user runs it manually more than once, this should
// do something sane.
//
// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK:       firrtl.module @Foo()
// CHECK:       firrtl.module private @Bar()
// CHECK:       firrtl.module private @Baz()
firrtl.circuit "Foo" {
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
    firrtl.instance baz @Baz()
  }
  firrtl.module private @Bar() {}
  firrtl.module private @Baz() {}
}


// Test that symbol collisions are properly handled as opposed to causing a
// crash.
//
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo" {
  // CHECK-NEXT: firrtl.module @Qux_0
  firrtl.module @A() {
  }
  firrtl.extmodule @Qux()
  firrtl.extmodule @Foo()
}
