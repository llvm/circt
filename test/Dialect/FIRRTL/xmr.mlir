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
    %e = firrtl.xmr.end %x : !firrtl.ref<uint<2>>
  }
  firrtl.module @xmr() {
    %test_x = firrtl.instance test @Test(in x: !firrtl.ref<uint<2>>)
    %x = firrtl.xmr.get %test_x : !firrtl.ref<uint<2>>
  }
}

// -----

// From design doc

firrtl.circuit "DUT" {
  firrtl.module @DUT() {
    %w = firrtl.wire sym @w : !firrtl.uint<1>

    %view_in, %view_out = firrtl.instance MyView_companion @MyView_companion(in ref_in1: !firrtl.ref<uint<1>>, out ref_out1: !firrtl.ref<uint<1>>)

    %view_in_end = firrtl.xmr.end %view_in : !firrtl.ref<uint<1>>
    firrtl.strictconnect %view_in_end, %w : !firrtl.uint<1>
    %iface = sv.interface.instance sym @__MyView_MyInterface__  : !sv.interface<@MyInterface>
    // Sink of XMR
    %view_out_end = firrtl.xmr.end %view_out : !firrtl.ref<uint<1>>
    sv.interface.signal.assign %iface(@MyInterface::@bool) = %view_out_end : !firrtl.uint<1>
    // firrtl.strictconnect %view_out_end, %iface : !firrtl.uint<1>>
  }

  firrtl.module private @MyView_companion (in %ref_in1: !firrtl.ref<uint<1>>, out %ref_out1: !firrtl.ref<uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %_WIRE = firrtl.wire sym @_WIRE : !firrtl.uint<1>
    firrtl.strictconnect %_WIRE, %c0_ui1 : !firrtl.uint<1>

    %view_in, %view_out = firrtl.instance MyView_mapping @MyView_mapping(in ref_in1: !firrtl.ref<uint<1>>, out ref_out1: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %view_in, %ref_in1 : !firrtl.ref<uint<1>>
    firrtl.strictconnect %ref_out1, %view_out : !firrtl.ref<uint<1>>
  }

  firrtl.module @MyView_mapping(in %ref_in1: !firrtl.ref<uint<1>>, out %ref_out1: !firrtl.ref<uint<1>>) {
    %0 = firrtl.xmr.get %ref_in1 : !firrtl.ref<uint<1>>
    %1 = firrtl.xmr.get %ref_out1 : !firrtl.ref<uint<1>>
    firrtl.strictconnect %1, %0 : !firrtl.uint<1>
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
    %0 = firrtl.xmr.end %_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
  firrtl.module @Foo() {
    %bar_a = firrtl.instance bar @Bar(in _a: !firrtl.ref<uint<1>>)

    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.xmr.get %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %0, %zero : !firrtl.uint<1>
  }
}
