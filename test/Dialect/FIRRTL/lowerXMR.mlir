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

// Test for hierarchichal lowering
// CHECK-LABEL: firrtl.circuit "ForwardToInstance" {
firrtl.circuit "ForwardToInstance" {
  firrtl.module @Bar2(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @Bar2() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1> {inner_sym = #firrtl<innerSym@xmr_sym>}
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @xmr_sym  @Bar2()
    firrtl.strictconnect %_a, %bar_2 : !firrtl.ref<uint<1>>
  }
  firrtl.module @ForwardToInstance() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %0 = firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@ForwardToInstance::@xmr_sym>, #hw.innerNameRef<@Bar::@xmr_sym>, #hw.innerNameRef<@Bar2::@xmr_sym>]}
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Test for multiple readers
// CHECK-LABEL: firrtl.circuit "ForwardToInstance" {
firrtl.circuit "ForwardToInstance" {
  firrtl.module @Bar2(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @Bar2() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1> {inner_sym = #firrtl<innerSym@xmr_sym>}
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @xmr_sym  @Bar2()
    firrtl.strictconnect %_a, %bar_2 : !firrtl.ref<uint<1>>
    %0 = firrtl.ref.resolve %bar_2 : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %0 = firrtl.verbatim.expr "{{0}}.{{1}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@Bar::@xmr_sym>, #hw.innerNameRef<@Bar2::@xmr_sym>]}
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
  firrtl.module @ForwardToInstance() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    // CHECK-LITERAL:  %0 = firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@ForwardToInstance::@xmr_sym>, #hw.innerNameRef<@Bar::@xmr_sym>, #hw.innerNameRef<@Bar2::@xmr_sym>]}
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}
