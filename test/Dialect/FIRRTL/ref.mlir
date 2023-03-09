// RUN: circt-opt %s -split-input-file
// RUN: firtool %s -split-input-file
// These tests are just for demonstrating RefOps, and expected to not error.

// Simple 1 level read from wire.
firrtl.circuit "xmr" {
  firrtl.module private @Test(out %x: !firrtl.ref<uint<2>>) {
    %w = firrtl.wire : !firrtl.uint<2>
    %zero = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.strictconnect %w, %zero : !firrtl.uint<2>
    %1 = firrtl.ref.send %w : !firrtl.uint<2>
    firrtl.ref.define %x, %1 : !firrtl.ref<uint<2>>
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
    firrtl.ref.define %_a, %1 : !firrtl.ref<uint<1>>
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
    firrtl.ref.define %_a, %1 : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    firrtl.ref.define %_a, %bar_2 : !firrtl.ref<uint<1>>
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
    firrtl.ref.define %_a, %1    : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    firrtl.ref.define %_a, %bar_2 : !firrtl.ref<uint<1>>
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
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %w_data1 = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %w_data1, %zero : !firrtl.uint<1>
    %1 = firrtl.ref.send %w_data1 : !firrtl.uint<1>
    firrtl.ref.define %ref_out1, %1 : !firrtl.ref<uint<1>>
    %w_data2 = firrtl.wire : !firrtl.uint<4>
    %zero4 = firrtl.constant 0 : !firrtl.uint<4>
    firrtl.strictconnect %w_data2, %zero4 : !firrtl.uint<4>
    %2 = firrtl.ref.send %w_data2 : !firrtl.uint<4>
    firrtl.ref.define %ref_out2, %2 : !firrtl.ref<uint<4>>
  }
  firrtl.module @DUT() {
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

// -----

// RefType of aggregates and RefSub. 
firrtl.circuit "RefTypeVector" {
  firrtl.module @RefTypeVector(in %bundle : !firrtl.bundle<a: uint<1>, b flip: uint<2>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<4>
    %z = firrtl.bitcast %zero : (!firrtl.uint<4>) -> !firrtl.vector<uint<1>,4>
    %1 = firrtl.ref.send %z : !firrtl.vector<uint<1>,4>
    %10 = firrtl.ref.sub %1[0] : !firrtl.ref<vector<uint<1>,4>>
    %11 = firrtl.ref.sub %1[1] : !firrtl.ref<vector<uint<1>,4>>
    %a = firrtl.ref.resolve %10 : !firrtl.ref<uint<1>>
    %b = firrtl.ref.resolve %11 : !firrtl.ref<uint<1>>
    %b1 = firrtl.ref.send %bundle : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    %12 = firrtl.ref.sub %b1[1] : !firrtl.ref<bundle<a: uint<1>, b: uint<2>>>
    %rb = firrtl.ref.resolve %12 : !firrtl.ref<uint<2>>
    %bundle_b = firrtl.subfield %bundle[b] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    %zero2 = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.strictconnect %bundle_b, %zero2 : !firrtl.uint<2>
  }
}

// -----

// https://github.com/llvm/circt/issues/3715
firrtl.circuit "Issue3715" {
  firrtl.module private @Test(in %p: !firrtl.uint<1>, out %x: !firrtl.ref<uint<2>>) {
    firrtl.when %p : !firrtl.uint<1> {
      %zero = firrtl.constant 1 : !firrtl.uint<2>
      %w = firrtl.wire : !firrtl.uint<2>
      %1 = firrtl.ref.send %w : !firrtl.uint<2>
      firrtl.ref.define %x, %1 : !firrtl.ref<uint<2>>
      firrtl.strictconnect %w, %zero : !firrtl.uint<2>
    }
  }
  firrtl.module @Issue3715(in %p: !firrtl.uint<1>) {
    %test_in, %test_x = firrtl.instance test @Test(in p: !firrtl.uint<1>, out x: !firrtl.ref<uint<2>>)
    firrtl.strictconnect %test_in, %p : !firrtl.uint<1>
    %x = firrtl.ref.resolve %test_x : !firrtl.ref<uint<2>>
  }
}

// -----

// Support using output port reference locally.
// https://github.com/llvm/circt/issues/3713

firrtl.circuit "UseRefsWithSinkFlow" {
  firrtl.module private @InChild(in %p: !firrtl.ref<uint<1>>) {
  }
  firrtl.module private @OutChild(in %x: !firrtl.uint, out %y: !firrtl.uint, out %p: !firrtl.ref<uint>) {
    %0 = firrtl.ref.send %x : !firrtl.uint
    firrtl.ref.define %p, %0 : !firrtl.ref<uint>
    %1 = firrtl.ref.resolve %p : !firrtl.ref<uint>
    firrtl.connect %y, %1 : !firrtl.uint, !firrtl.uint
  }
  firrtl.module @UseRefsWithSinkFlow(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>, out %z: !firrtl.uint<1>, out %zz: !firrtl.uint<1>, out %p: !firrtl.ref<uint<1>>) {
    %0 = firrtl.ref.send %x : !firrtl.uint<1>
    firrtl.ref.define %p, %0 : !firrtl.ref<uint<1>>
    %1 = firrtl.ref.resolve %p : !firrtl.ref<uint<1>>
    firrtl.strictconnect %y, %1 : !firrtl.uint<1>
    %ic_p = firrtl.instance ic interesting_name @InChild(in p: !firrtl.ref<uint<1>>)
    %2 = firrtl.ref.send %x : !firrtl.uint<1>
    firrtl.ref.define %ic_p, %2 : !firrtl.ref<uint<1>>
    %3 = firrtl.ref.resolve %ic_p : !firrtl.ref<uint<1>>
    firrtl.strictconnect %z, %3 : !firrtl.uint<1>
    %oc_x, %oc_y, %oc_p = firrtl.instance oc interesting_name @OutChild(in x: !firrtl.uint, out y: !firrtl.uint, out p: !firrtl.ref<uint>)
    firrtl.connect %oc_x, %x : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %zz, %oc_y : !firrtl.uint<1>, !firrtl.uint
  }
}
