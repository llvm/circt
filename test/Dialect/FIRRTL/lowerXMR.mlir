// RUN: circt-opt %s  --firrtl-lower-xmr -split-input-file |  FileCheck %s

// CHECK-LABEL: firrtl.circuit "xmr"
firrtl.circuit "xmr" {
  firrtl.module private @Test(out %x: !firrtl.ref<uint<2>>) {
    %w = firrtl.wire : !firrtl.uint<2>
    firrtl.ref.send %x, %w : !firrtl.ref<uint<2>>
  }
  firrtl.module @xmr() {
    %test_x = firrtl.instance test @Test(out x: !firrtl.ref<uint<2>>)
    %x = firrtl.ref.resolve %test_x : !firrtl.ref<uint<2>>
    // This should be lowered to:
    //  %0 = firrtl.verbatim.expr "{{0}}.{{1}}" : () -> !firrtl.uint<2>
    //  {symbols = [#hw.innerNameRef<@xmr::@xmr_sym>, #hw.innerNameRef<@Test::@xmr_sym>]}
  }
  // CHECK:   firrtl.module private @Test() {
  // CHECK:     %w = firrtl.wire sym @xmr_sym   : !firrtl.uint<2>
  // CHECK:   }
  // CHECK:   firrtl.module @xmr() {
  // CHECK:     firrtl.instance test sym @xmr_sym  @Test()
  // CHECK:     %0 = firrtl.verbatim.expr 
  // CHECK-SAME: > !firrtl.uint<2> {symbols = [#hw.innerNameRef<@xmr::@xmr_sym>, #hw.innerNameRef<@Test::@xmr_sym>]}
}

// -----

// CHECK-LABEL: firrtl.circuit "DUT" {
firrtl.circuit "DUT" {
  firrtl.module private @Submodule (out %ref_out1: !firrtl.ref<uint<1>>, out %ref_out2: !firrtl.ref<uint<4>>) {
  // CHECK:  firrtl.module private @Submodule() {
    %w_data1 = firrtl.wire : !firrtl.uint<1>
    firrtl.ref.send %ref_out1, %w_data1 : !firrtl.ref<uint<1>>
    %w_data2 = firrtl.wire : !firrtl.uint<4>
    firrtl.ref.send %ref_out2, %w_data2 : !firrtl.ref<uint<4>>
  // CHECK:    %w_data1 = firrtl.wire sym @xmr_sym   : !firrtl.uint<1>
  // CHECK:    %w_data2 = firrtl.wire sym @xmr_sym_0   : !firrtl.uint<4>
  }
  firrtl.module @DUT() {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
    %view_out1, %view_out2 = firrtl.instance sub @Submodule(out ref_out1: !firrtl.ref<uint<1>>, out ref_out2: !firrtl.ref<uint<4>>)
    %view_in1, %view_in2 = firrtl.instance MyView_companion @MyView_companion(in ref_in1: !firrtl.uint<1>, in ref_in2: !firrtl.uint<4>)

    %1 = firrtl.ref.resolve %view_out1 : !firrtl.ref<uint<1>>
    //  %0 = firrtl.verbatim.expr "{{0}}.{{1}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@DUT::@xmr_sym>, #hw.innerNameRef<@Submodule::@xmr_sym>]}
    %2 = firrtl.ref.resolve %view_out2 : !firrtl.ref<uint<4>>
    //  %1 = firrtl.verbatim.expr "{{0}}.{{1}}" : () -> !firrtl.uint<4> {symbols = [#hw.innerNameRef<@DUT::@xmr_sym>, #hw.innerNameRef<@Submodule::@xmr_sym_0>]}
    firrtl.strictconnect %view_in1, %1 : !firrtl.uint<1>
    firrtl.strictconnect %view_in2, %2 : !firrtl.uint<4>
    // CHECK:  %0 = firrtl.verbatim.expr
    // CHECK-SAME: !firrtl.uint<1> {symbols = [#hw.innerNameRef<@DUT::@xmr_sym>, #hw.innerNameRef<@Submodule::@xmr_sym>]}
    // CHECK:  %1 = firrtl.verbatim.expr
    // CHECK-SAME: !firrtl.uint<4> {symbols = [#hw.innerNameRef<@DUT::@xmr_sym>, #hw.innerNameRef<@Submodule::@xmr_sym_0>]}
    // CHECK:  firrtl.strictconnect %MyView_companion_ref_in1, %0 : !firrtl.uint<1>
    // CHECK:  firrtl.strictconnect %MyView_companion_ref_in2, %1 : !firrtl.uint<4>
  }

  firrtl.module private @MyView_companion (in %ref_in1: !firrtl.uint<1>, in %ref_in2: !firrtl.uint<4>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %_WIRE = firrtl.wire sym @_WIRE : !firrtl.uint<1>
    firrtl.strictconnect %_WIRE, %c0_ui1 : !firrtl.uint<1>
    %iface = sv.interface.instance sym @__MyView_MyInterface__  : !sv.interface<@MyInterface>
  }

  sv.interface @MyInterface {
    sv.verbatim "// a wire called 'bool'" {symbols = []}
    sv.interface.signal @bool : i1
  }
}

// -----

// "Example 2"
// CHECK-LABEL: firrtl.circuit "SimpleRead" {
firrtl.circuit "SimpleRead" {
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
  // CHECK:  firrtl.module @Bar() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.ref.send %_a, %zero : !firrtl.ref<uint<1>>
    // CHECK:  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1> {inner_sym = #firrtl<innerSym@xmr_sym>}
  }
  firrtl.module @SimpleRead() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    // CHECK:  firrtl.instance bar sym @xmr_sym  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    //   %0 = firrtl.verbatim.expr "{{0}}.{{1}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@SimpleRead::@xmr_sym>, #hw.innerNameRef<@Bar::@xmr_sym>]}
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:  %0 = firrtl.verbatim.expr
    // CHECK-SAME: !firrtl.uint<1> {symbols = [#hw.innerNameRef<@SimpleRead::@xmr_sym>, #hw.innerNameRef<@Bar::@xmr_sym>]}
    // CHECK: firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: firrtl.circuit "ForwardToInstance" {
firrtl.circuit "ForwardToInstance" {
  firrtl.module @Bar2(out %_a: !firrtl.ref<uint<1>>) {
    // CHECK: firrtl.module @Bar2() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1> {inner_sym = #firrtl<innerSym@xmr_sym>}
    firrtl.ref.send %_a, %zero : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %_a, %bar_2 : !firrtl.ref<uint<1>>
  }
  firrtl.module @ForwardToInstance() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    //  %0 = firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}" : () -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@ForwardToInstance::@xmr_sym>, #hw.innerNameRef<@Bar::@xmr_sym>, #hw.innerNameRef<@Bar2::@xmr_sym>]}
    // CHECK:  -> !firrtl.uint<1> {symbols = [#hw.innerNameRef<@ForwardToInstance::@xmr_sym>, #hw.innerNameRef<@Bar::@xmr_sym>, #hw.innerNameRef<@Bar2::@xmr_sym>]}
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}
