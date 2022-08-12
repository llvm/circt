// RUN: circt-opt %s -split-input-file
// These tests are just for demonstrating RefOps, and expected to not error.

// Simple 1 level read from wire.
firrtl.circuit "xmr" {
  firrtl.module private @Test(out %x: !firrtl.ref<uint<2>>) {
    %w = firrtl.wire : !firrtl.uint<2>
    %1 = firrtl.ref.send %w : !firrtl.uint<2>
    firrtl.strictconnect %x, %1 : !firrtl.ref<uint<2>>
  }
  firrtl.module @xmr() {
    %test_x = firrtl.instance test @Test(out x: !firrtl.ref<uint<2>>)
    %x = firrtl.ref.resolve %test_x : !firrtl.ref<uint<2>>
  }
}

// -----

// Simple 1 level read from constant.
firrtl.circuit "SimpleRead" {
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @SimpleRead() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Forward module port to instance
firrtl.circuit "ForwardToInstance" {
  firrtl.module @Bar2(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %_a, %bar_2 : !firrtl.ref<uint<1>>
  }
  firrtl.module @ForwardToInstance() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Multiple readers, for a single remote value.
firrtl.circuit "ForwardToInstance" {
  firrtl.module @Bar2(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.strictconnect %_a, %1    : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %_a, %bar_2 : !firrtl.ref<uint<1>>
    // Reader 1
    %0 = firrtl.ref.resolve %bar_2 : !firrtl.ref<uint<1>>
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
  firrtl.module @ForwardToInstance() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    // Reader 2
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Two references passed by value.
firrtl.circuit "DUT" {
  firrtl.module private @Submodule (out %ref_out1: !firrtl.ref<uint<1>>, out %ref_out2: !firrtl.ref<uint<4>>) {
    %w_data1 = firrtl.wire : !firrtl.uint<1>
    %1 = firrtl.ref.send %w_data1 : !firrtl.uint<1>
    firrtl.strictconnect %ref_out1, %1 : !firrtl.ref<uint<1>>
    %w_data2 = firrtl.wire : !firrtl.uint<4>
    %2 = firrtl.ref.send %w_data2 : !firrtl.uint<4>
    firrtl.strictconnect %ref_out2, %2 : !firrtl.ref<uint<4>>
  }
  firrtl.module @DUT() {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
    %view_out1, %view_out2 = firrtl.instance sub @Submodule(out ref_out1: !firrtl.ref<uint<1>>, out ref_out2: !firrtl.ref<uint<4>>)
    %view_in1, %view_in2 = firrtl.instance MyView_companion @MyView_companion(in ref_in1: !firrtl.uint<1>, in ref_in2: !firrtl.uint<4>)

    %1 = firrtl.ref.resolve %view_out1 : !firrtl.ref<uint<1>>
    %2 = firrtl.ref.resolve %view_out2 : !firrtl.ref<uint<4>>
    firrtl.strictconnect %view_in1, %1 : !firrtl.uint<1>
    firrtl.strictconnect %view_in2, %2 : !firrtl.uint<4>
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
