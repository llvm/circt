// RUN: circt-opt %s  --firrtl-lower-xmr -split-input-file |  FileCheck %s

// Test for same module lowering
// CHECK-LABEL: firrtl.circuit "xmr"
firrtl.circuit "xmr" {
  firrtl.module @xmr(out %o: !firrtl.uint<2>) {
    %w = firrtl.wire : !firrtl.uint<2>
    %1 = firrtl.ref.send %w : !firrtl.uint<2>
    %x = firrtl.ref.resolve %1 : !firrtl.ref<uint<2>>
    firrtl.strictconnect %o, %x : !firrtl.uint<2>
    // CHECK:  %w = firrtl.wire sym @xmr_sym   : !firrtl.uint<2>
    // CHECK-LITERAL:  %0 = firrtl.verbatim.expr "{{0}}" : () -> !firrtl.uint<2> {symbols = [#hw.innerNameRef<@xmr::@xmr_sym>]}
    // CHECK:  firrtl.strictconnect %o, %0 : !firrtl.uint<2>
  }
}

// -----

// Test the correct xmr path is generated
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1> {inner_sym = #firrtl<innerSym@xmr_sym>}
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %0 = firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@Top::@bar>, #hw.innerNameRef<@Bar::@barXMR>, #hw.innerNameRef<@XmrSrcMod::@xmr_sym>]}
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Test the correct xmr path to port is generated
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1> sym @xmr_sym) {
    %1 = firrtl.ref.send %pa : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %pa, %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.ref<uint<1>>)
    // CHECK: %bar_pa = firrtl.instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<1>)
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK-LITERAL: %0 = firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@Top::@bar>, #hw.innerNameRef<@Bar::@barXMR>, #hw.innerNameRef<@XmrSrcMod::@xmr_sym>]}
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Test for multiple readers and multiple instances
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  firrtl.module @XmrSrcMod(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1> {inner_sym = #firrtl<innerSym@xmr_sym>}
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Foo(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @fooXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
    %0 = firrtl.ref.resolve %xmr   : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %0 = firrtl.verbatim.expr "{{0}}.{{1}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@Foo::@fooXMR>, #hw.innerNameRef<@XmrSrcMod::@xmr_sym>]}
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.strictconnect %_a, %xmr   : !firrtl.ref<uint<1>>
    %0 = firrtl.ref.resolve %xmr   : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %0 = firrtl.verbatim.expr "{{0}}.{{1}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@Bar::@barXMR>, #hw.innerNameRef<@XmrSrcMod::@xmr_sym>]}
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.ref<uint<1>>)
    %foo_a = firrtl.instance foo sym @foo @Foo(out _a: !firrtl.ref<uint<1>>)
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    // CHECK:  firrtl.instance foo sym @foo  @Foo()
    // CHECK:  firrtl.instance xmr sym @xmr  @XmrSrcMod()
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.uint<1>
    %c = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %0 = firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@Top::@bar>, #hw.innerNameRef<@Bar::@barXMR>, #hw.innerNameRef<@XmrSrcMod::@xmr_sym>]}
    %1 = firrtl.ref.resolve %foo_a : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %1 = firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@Top::@foo>, #hw.innerNameRef<@Foo::@fooXMR>, #hw.innerNameRef<@XmrSrcMod::@xmr_sym>]}
    %2 = firrtl.ref.resolve %xmr_a : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %2 = firrtl.verbatim.expr "{{0}}.{{1}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@Top::@xmr>, #hw.innerNameRef<@XmrSrcMod::@xmr_sym>]}
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    firrtl.strictconnect %b, %1 : !firrtl.uint<1>
    firrtl.strictconnect %c, %2 : !firrtl.uint<1>
  }
}
