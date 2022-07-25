// For now, just checking very basic cases parse properly.

// Lots of checking around RefType likely warranted:
// * firrtl.ref<uint> -- handling?
// 
// Errors:
// * nested ref
// * Use anywhere other than handful of approved places
// * use in an aggregate type (bundle/vector/etc)

// RUN: circt-opt %s -split-input-file

firrtl.circuit "xmr" {
  firrtl.module private @Test(in %x: !firrtl.ref<uint<2>>) {
    %w = firrtl.wire : !firrtl.uint<2>
    firrtl.xmr.write %w, %x : !firrtl.ref<uint<2>>
  }
  firrtl.module @xmr() {
    %test_x = firrtl.instance test @Test(in x: !firrtl.ref<uint<2>>)
    %x = firrtl.xmr.get %test_x : !firrtl.ref<uint<2>>
  }
}

// -----

firrtl.circuit "DUT" {
  firrtl.module private @Submodule (out %ref_out1: !firrtl.ref<uint<1>>, out %ref_out2: !firrtl.ref<uint<4>>) {
    %w_data1 = firrtl.wire : !firrtl.uint<1>
    firrtl.xmr.read %ref_out1, %w_data1 : !firrtl.ref<uint<1>>
    %w_data2 = firrtl.wire : !firrtl.uint<4>
    firrtl.xmr.read %ref_out2, %w_data2 : !firrtl.ref<uint<4>>
  }
  firrtl.module @DUT() {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
    %view_out1, %view_out2 = firrtl.instance sub @Submodule(out ref_out1: !firrtl.ref<uint<1>>, out ref_out2: !firrtl.ref<uint<4>>)
    %view_in1, %view_in2 = firrtl.instance MyView_companion @MyView_companion(in ref_in1: !firrtl.uint<1>, in ref_in2: !firrtl.uint<4>)

    %1 = firrtl.xmr.get %view_out1 : !firrtl.ref<uint<1>>
    %2 = firrtl.xmr.get %view_out2 : !firrtl.ref<uint<4>>
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

// -----

// "Example 1"

firrtl.circuit "Foo" {
  firrtl.module @Bar(in %_a: !firrtl.ref<uint<1>>) {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.xmr.write %a, %_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Foo() {
    %bar_a = firrtl.instance bar @Bar(in _a: !firrtl.ref<uint<1>>)

    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.xmr.get %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %0, %zero : !firrtl.uint<1>
  }
}

// -----

// "Example 2"
firrtl.circuit "Foo" {
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.xmr.read %_a, %zero : !firrtl.ref<uint<1>>
  }
  firrtl.module @Foo() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.xmr.get %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// "Example 1"

firrtl.circuit "Foo" {
  firrtl.module @Bar(in %_a: !firrtl.ref<uint<1>>) {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.xmr.write %a, %_a : !firrtl.ref<uint<1>>
  }
  firrtl.module @Foo() {
    %bar_a = firrtl.instance bar1 @Bar(in _a: !firrtl.ref<uint<1>>)
    // bar_b is unconnected.
    %bar_b = firrtl.instance bar2 @Bar(in _a: !firrtl.ref<uint<1>>)

    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.xmr.get %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %0, %zero : !firrtl.uint<1>
  }
}

// -----

// "Example 2"
firrtl.circuit "Foo" {
  firrtl.module @Bar2(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.xmr.read %_a, %zero : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %_a, %bar_2 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Foo() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.xmr.get %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}
